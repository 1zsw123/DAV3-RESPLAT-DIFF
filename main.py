import os
import time
from pathlib import Path

import hydra
import torch
import wandb
import signal
import lightning.pytorch as pl
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from datetime import timedelta

from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf


class ETACallback(pl.Callback):
    """在终端打印整体训练进度和预计剩余时间。"""

    def __init__(self, max_steps: int, print_every: int = 50):
        self.max_steps = max_steps
        self.print_every = print_every
        self._start_time: float | None = None

    def on_train_start(self, trainer, pl_module):
        self._start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.print_every != 0:
            return
        if trainer.local_rank != 0:
            return

        elapsed = time.time() - self._start_time
        speed = step / elapsed                          # steps / sec
        remaining = (self.max_steps - step) / speed    # seconds
        h, m = divmod(int(remaining), 3600)
        m //= 60

        pct = 100.0 * step / self.max_steps
        bar_len = 30
        filled = int(bar_len * step / self.max_steps)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"\r[{bar}] {pct:5.1f}%  step {step}/{self.max_steps}"
            f"  {speed:.2f} it/s  ETA {h}h {m:02d}m",
            flush=True,
        )

import contextlib
from src.misc.weight_modify import checkpoint_filter_fn
from src.model.distiller import get_distiller

# Skip beartype runtime type-checking in SLURM jobs to avoid GPFS bottleneck.
# Set BEARTYPE_TYPECHECK=1 to re-enable during local debugging.
_use_beartype = os.environ.get("BEARTYPE_TYPECHECK", "0") == "1"
_hook_ctx = (
    install_import_hook(("src",), ("beartype", "beartype"))
    if _use_beartype
    else contextlib.nullcontext()
)
with _hook_ctx:
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.load_foundation_model import load_foundation_model
    from src.model.mip_splatting_refiner import MipSplattingRefiner
    from src.model.diffusion_head import DiffusionHead


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing: latest checkpoint.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=1,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
            filename="latest-{info/global_step:.0f}",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'
    # Best checkpoint by val/psnr_crop when border mask enabled, else val/psnr.
    # Crop metric is the one we actually train against (border-masked loss),
    # so it's the truer signal of model improvement under the new loss regime.
    import os as _os_ckpt
    _monitor_key = "val/psnr_crop" if _os_ckpt.environ.get("BORDER_MASK_ENABLE", "0") == "1" else "val/psnr"
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            save_top_k=1,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor=_monitor_key,
            mode="max",
            filename=("best-psnrcrop-{val/psnr_crop:.3f}-{info/global_step:.0f}"
                     if _monitor_key == "val/psnr_crop"
                     else "best-psnr-{val/psnr:.3f}-{info/global_step:.0f}"),
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # ETA progress callback (only when max_steps is set)
    if cfg.trainer.max_steps > 0:
        callbacks.append(ETACallback(max_steps=cfg.trainer.max_steps, print_every=50))

    # Clear GPU cache before each validation to prevent memory fragmentation OOM
    class ClearCacheCallback(Callback):
        def on_validation_epoch_start(self, trainer, pl_module):
            import torch
            torch.cuda.empty_cache()
    callbacks.append(ClearCacheCallback())

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            # static_graph=True freezes the unused-parameter set after the first
            # forward — this is the official fix for find_unused_parameters NCCL
            # desyncs caused by rank-dependent forward paths (different ranks
            # reaching different conditional branches → different unused set →
            # mismatched allreduce shapes → 30-min ALLREDUCE timeout).
            # Requires every step to use the same trainable-param subset, which
            # is now guaranteed since all loss early-returns are graph-connected.
            DDPStrategy(find_unused_parameters=True,
                        static_graph=False,  # was True → caused expect_autograd_hooks_ assert when model has dynamic forward path (diffusion head N-step inference)
                        broadcast_buffers=False,
                        gradient_as_bucket_view=True,
                        timeout=timedelta(minutes=30))
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    
    vggt, dino, lseg_feature_extractor, clip, feature_dim = load_foundation_model(cfg)
    cfg.model.encoder.feature_dim = feature_dim if cfg.train.feature_rendering_loss > 0 else 0

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu')
        if 'model' in ckpt_weights:
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        elif 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        elif isinstance(ckpt_weights, dict):
            new_ckpt = {}
            for key, value in ckpt_weights.items():
                if 'aggregator' in key:
                    new_ckpt[f'backbone.{key}'] = value
                if 'point_head' in key:
                    new_ckpt[key.replace('point_head', 'dpt_head')] = value
            missing_keys, unexpected_keys = encoder.load_state_dict(new_ckpt, strict=False)
            del new_ckpt
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")
        
        del ckpt_weights



    # Optionally instantiate Mip-Splatting per-scene Gaussian refiner.
    refiner = None
    if cfg.model.refiner is not None:
        refiner = MipSplattingRefiner(cfg.model.refiner)
        print(cyan(f"MipSplattingRefiner enabled ({cfg.model.refiner.num_opt_steps} opt steps/sample)."))

    # Optionally instantiate DiffusionHead flow-matching deblurrer.
    diffusion_head = None
    if cfg.model.diffusion_head is not None:
        # Pre-compute d_gauss so DiffusionHead sub-modules exist before
        # configure_optimizers is called (lazy d_gauss=0 would leave the
        # optimizer with zero DiffusionHead parameters).
        _sh = cfg.model.encoder.gaussian_adapter.sh_degree
        _d_gauss = 3 + 3 + 4 + 1 + 3 * (_sh + 1) ** 2  # means+log_scales+quat+logit_opa+harmonics
        diffusion_head = DiffusionHead(cfg.model.diffusion_head, d_gauss=_d_gauss)
        print(cyan(f"DiffusionHead enabled (d_gauss={_d_gauss})."))

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
        vggt=vggt,
        dino=dino,
        clip=clip,
        lseg_feature_extractor = lseg_feature_extractor,
        mode=cfg.mode,
        refiner=refiner,
        diffusion_head=diffusion_head,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
    torch.cuda.empty_cache()

    if cfg.mode == "train":
        # load_weights_only: load model weights from checkpoint but reset step/optimizer state.
        # Use for fine-tuning where training should start fresh from step 0.
        if getattr(cfg.checkpointing, "load_weights_only", False) and checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            # Filter out keys with shape mismatch so strict=False doesn't still hard-fail.
            model_state = model_wrapper.state_dict()
            filtered, skipped_shape = {}, []
            for k, v in state.items():
                if k in model_state and model_state[k].shape != v.shape:
                    skipped_shape.append(k)
                else:
                    filtered[k] = v
            if skipped_shape:
                print(f"[checkpointing] Skipped {len(skipped_shape)} shape-mismatched keys: {skipped_shape[:5]}...")
            missing, unexpected = model_wrapper.load_state_dict(filtered, strict=False)
            if missing:
                print(f"[checkpointing] Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"[checkpointing] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
            print(f"[checkpointing] Loaded weights-only from {checkpoint_path}, starting from step 0.")
            checkpoint_path = None
        # Baseline val before training: trainer.validate() runs val_step AND
        # logs metrics to wandb (sanity_val does not log). Triggered for ANY
        # ckpt load (weights-only OR full resume) so we always have a
        # baseline val data point to compare with subsequent vals.
        # IMPORTANT: pass ckpt_path so Lightning loads the ckpt before val
        # (otherwise full-resume baseline runs on fresh-init weights and
        # produces identical numbers across all runs — was a real bug).
        if cfg.checkpointing.load is not None:
            print("[main] Running baseline val on loaded ckpt before training...")
            trainer.validate(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()

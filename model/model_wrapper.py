import contextlib
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Dict, Optional, Protocol, runtime_checkable, Any
from itertools import accumulate

import moviepy as mpy
import torch
import torch.nn.functional as F
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
from torchmetrics import JaccardIndex, Accuracy
import os
from tqdm import tqdm
import numpy as np
import torch.distributed as dist

from ..dataset.data_module import get_data_shim


def _safe_clamp_aniso(cov: Tensor, max_log_delta: float = 2.5) -> Tensor:
    """Eigh-decompose covariance, clamp |log_scale - mean_log_scale| <= max_log_delta
    (per gaussian), recompose. Caps anisotropy ratio at exp(2*max_log_delta).
    Default max_log_delta=2.5 → max_ratio = exp(5) ≈ 148 (vs uncapped 440K).

    Stability notes (v7 hardening — v6 crashed via this code path):
      - eigh backward divides by (λ_i - λ_j), producing NaN on degenerate eigenvalues
        (exactly the regime aniso_hinge is pushing toward → unavoidable collision).
        Fix: detach eigvecs so backward only flows through eigvals (stable).
      - eigvals.clamp(min=1e-12).log() has backward = 1/eigval = 1e12, overflows bf16.
        Fix: raise floor to 1e-6 → grad capped at 1e6.
    """
    eigvals, eigvecs = torch.linalg.eigh(cov.float())  # [..., 3]
    eigvecs = eigvecs.detach()                          # block unstable backward path
    eigvals = eigvals.clamp(min=1e-6)                   # bf16-safe log gradient
    log_s = 0.5 * eigvals.log()                        # log scales (sqrt of eigvals)
    mean_log_s = log_s.mean(dim=-1, keepdim=True)
    delta = (log_s - mean_log_s).clamp(-max_log_delta, max_log_delta)
    new_log_s = mean_log_s + delta
    new_eigvals = (2 * new_log_s).exp()
    new_cov = eigvecs @ torch.diag_embed(new_eigvals) @ eigvecs.transpose(-1, -2)
    return new_cov.to(cov.dtype)
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss import LossLpips
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose
from ..misc.image_io import prep_image, save_image, save_video, visualize_attention_map
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from src.model.clip import clip
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .mip_splatting_refiner import MipSplattingRefiner
from .diffusion_head import DiffusionHead, gaussians_to_attrs
from .utils import  save_segmap, run_pca

import importlib as _importlib
import os as _os
_DAV3_TRANSFORM_PATH = _os.path.abspath(_os.path.join(
    _os.path.dirname(__file__), "..", "..",
    "Depth-Anything-3", "src", "depth_anything_3", "model", "utils", "transform.py"
))
try:
    _spec = _importlib.util.spec_from_file_location("dav3_transform", _DAV3_TRANSFORM_PATH)
    _dav3_mod = _importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_dav3_mod)
    _extri_intri_to_pose_enc = _dav3_mod.extri_intri_to_pose_encoding
    _HAS_POSE_ENC = True
except Exception:
    _HAS_POSE_ENC = False


# ── ProPE pose_enc helper ─────────────────────────────────────────────────────

def _compute_pose_enc(extrinsics4x4, intrinsics, h, w):
    """Convert [B,V,4,4] extrinsics + [B,V,3,3] intrinsics → [B,V,9] ProPE pose_enc.
    C3G stores normalized intrinsics (fx/W, fy/H, cx/W, cy/H); DAV3 expects pixel-scale."""
    if not _HAS_POSE_ENC:
        return None
    ext34 = extrinsics4x4[:, :, :3, :]          # [B,V,3,4]
    K_pixel = intrinsics.clone()
    K_pixel[..., 0, 0] = K_pixel[..., 0, 0] * w   # fx_norm * W → fx_pixel
    K_pixel[..., 1, 1] = K_pixel[..., 1, 1] * h   # fy_norm * H → fy_pixel
    K_pixel[..., 0, 2] = K_pixel[..., 0, 2] * w   # cx
    K_pixel[..., 1, 2] = K_pixel[..., 1, 2] * h   # cy
    return _extri_intri_to_pose_enc(ext34, K_pixel, image_size_hw=(h, w))


def _make_refiner_context(batch_context: dict, visualization_dump: dict, use_vggt: bool) -> dict:
    """Return context dict for MipSplatting TTO, optionally with VGGT-predicted poses."""
    if not use_vggt:
        return batch_context
    pred_ext = visualization_dump.get('vggt_pred_extrinsics')
    pred_intr = visualization_dump.get('vggt_pred_intrinsics')
    if pred_ext is None or pred_intr is None:
        return batch_context
    return {**batch_context, "extrinsics": pred_ext, "intrinsics": pred_intr}


# ── L_flow helpers ────────────────────────────────────────────────────────────

def _procrustes_align_gaussians(
    means: Float[Tensor, "batch gaussian 3"],
    depth: Float[Tensor, "batch view h w"],
    extrinsics_c2w: Float[Tensor, "batch view 4 4"],
    intrinsics_norm: Float[Tensor, "batch view 3 3"],
) -> tuple[Float[Tensor, "batch gaussian 3"], Float[Tensor, "batch 3 3"], Float[Tensor, "batch 3"], Float[Tensor, "batch"]]:
    """
    Align C3G Gaussian means to GT world frame using depth-based Procrustes.

    For each batch element:
      1. Unproject the first context view's depth map → GT point cloud P_gt [N, 3]
      2. Procrustes(means[b], P_gt) → (R, t, s)
      3. Apply: means_aligned = s * (R @ means.T).T + t

    Returns:
        means_aligned  [B, G, 3]
        R_batch        [B, 3, 3]
        t_batch        [B, 3]
        s_batch        [B]
    """
    B, G = means.shape[:2]
    device = means.device

    R_batch = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    t_batch = torch.zeros(3, device=device).unsqueeze(0).expand(B, -1).clone()
    s_batch = torch.ones(B, device=device)

    means_aligned = means.clone()

    for b in range(B):
        # Unproject depth of view 0
        d = depth[b, 0]          # [H, W]
        c2w = extrinsics_c2w[b, 0]   # [4, 4]
        K = intrinsics_norm[b, 0]     # [3, 3] normalised

        H, W = d.shape
        # Denormalise intrinsics
        fx, fy = K[0, 0] * W, K[1, 1] * H
        cx, cy = K[0, 2] * W, K[1, 2] * H

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        z = d.reshape(-1)
        mask = (z > 0.01) & (z < 10.0)
        if mask.sum() < 16:
            continue

        x_cam = ((xs.reshape(-1)[mask] - cx) / fx) * z[mask]
        y_cam = ((ys.reshape(-1)[mask] - cy) / fy) * z[mask]
        pts_cam = torch.stack([x_cam, y_cam, z[mask]], dim=-1)   # [N, 3]
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]    # [N, 3]

        # Sub-sample to same size as G for Procrustes
        N = min(G, pts_world.shape[0])
        idx_g  = torch.randperm(G, device=device)[:N]
        idx_pt = torch.randperm(pts_world.shape[0], device=device)[:N]

        src = means[b][idx_g].detach()        # [N, 3]
        tgt = pts_world[idx_pt]               # [N, 3]

        mu_src = src.mean(0)
        mu_tgt = tgt.mean(0)
        src_c  = src - mu_src
        tgt_c  = tgt - mu_tgt

        s = tgt_c.norm(dim=1).mean() / src_c.norm(dim=1).mean().clamp(min=1e-8)
        H_mat = (src_c * s).T @ tgt_c
        U, _, Vh = torch.linalg.svd(H_mat)
        det = torch.linalg.det(Vh.T @ U.T)
        D = torch.diag(torch.tensor([1., 1., det.item()], device=device))
        R = Vh.T @ D @ U.T

        t = mu_tgt - s * (R @ mu_src)

        R_batch[b] = R
        t_batch[b] = t
        s_batch[b] = s
        means_aligned[b] = s * (R @ means[b].T).T + t

    return means_aligned, R_batch, t_batch, s_batch


def _chamfer_distance(
    x: Float[Tensor, "batch n d"],
    y: Float[Tensor, "batch m d"],
) -> Float[Tensor, ""]:
    """
    Symmetric Chamfer distance (mean over batch).
    For each point in x find nearest in y, and vice versa.
    Complexity: O(B * N * M * D) — use small N, M in practice.
    """
    # x: [B, N, D], y: [B, M, D]
    # ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i·y_j
    xx = (x * x).sum(-1, keepdim=True)   # [B, N, 1]
    yy = (y * y).sum(-1, keepdim=True)   # [B, M, 1]
    xy = torch.bmm(x, y.transpose(1, 2)) # [B, N, M]
    dist2 = xx + yy.transpose(1, 2) - 2 * xy  # [B, N, M]
    dist2 = dist2.clamp(min=0)

    d_x2y = dist2.min(dim=2).values.mean()   # nearest-in-y for each x
    d_y2x = dist2.min(dim=1).values.mean()   # nearest-in-x for each y
    return (d_x2y + d_y2x) * 0.5


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    visualize_gaussian_token: int = -1
    forward_vfm: bool = False
    labels: list[str] = field(default_factory=lambda: ['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other'])
    color_hex_list: list[str] = field(default_factory=lambda: ['#000000', '#E6194B','#3CB44B','#FFE119','#4363D8','#F58231','#911EB4','#42D4F4','#808000'])

    # TTO (Test-Time Optimization) settings
    use_tto: bool = False
    tto_steps: int = 200
    tto_lr: float = 1e-3
    tto_geo_lr_scale: float = 0.1   # geometry (means/covariances) learns slower

@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    random_select_context_view: bool = False
    reproj_model: str = 'none' # 'vggt' or 'dino'
    feature_rendering_loss: float = 0.0
    freeze_encoder: bool = False  # set True to freeze all encoder params
    share_weight: float = 0.1    # weight for L_share (context-view reconstruction loss)
    context_view_loss: bool = True  # kept for backwards compat
    l_flow_weight: float = 0.0   # weight for L_flow Chamfer loss (0 = disabled)
    cam_loss_weight: float = 0.0 # weight for Design A camera pose_enc L2 loss
    use_vggt_poses_for_refiner: bool = False  # use VGGT-predicted poses instead of GT for MipSplatting TTO


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass
    
class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        vggt = None,
        dino = None,
        lseg_feature_extractor = None,
        clip = None,
        mode: str = "train",
        refiner: Optional[MipSplattingRefiner] = None,
        diffusion_head: Optional[DiffusionHead] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.refiner = refiner
        self.diffusion_head = diffusion_head

        # (sharp_refiner removed — canonical GT Gaussians are precomputed offline)

        if train_cfg.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            self.encoder.eval()  # also fix BN/dropout
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        self.mode=mode
            
        self.vggt=vggt
        self.lseg_feature_extractor = lseg_feature_extractor

        if dino is not None:
            self.dino_model = dino['model']
            self.dino_processor = dino['processor'] if 'processor' in dino else None
            self.latent_mean = dino['latent_mean'] if dino['latent_mean'] is not None else 0
            self.latent_var = dino['latent_var'] if dino['latent_var'] is not None else 1
        else:
            self.dino_model, self.dino_processor = None, None
            
        if clip is not None:
            self.clip_model = clip['model']
        else:
            self.clip_model = None
            
        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.miou = JaccardIndex(
            task="multiclass",
            num_classes=len(self.test_cfg.labels) + 1,
            ignore_index=0,
        )
        
        self.acc = Accuracy(
            task="multiclass",
            num_classes=len(self.test_cfg.labels) + 1,
            ignore_index=0,
        )

        self.per_image_ious = []
        self.per_image_accs = []
        self._val_psnr = []
        self._val_lpips = []
        self._val_ssim = []
        self._val_global_step = 0
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0


    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined

        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        if self.train_cfg.random_select_context_view:
            # Use a deterministic per-step seed so all DDP ranks pick the same
            # num_ctx_views without any explicit collective — the dist.broadcast
            # that was here shared the default NCCL communicator with DDP's
            # gradient all-reduce, causing a communicator ordering deadlock
            # after ~80 steps on single-node NVLink hardware.
            gen = torch.Generator()
            gen.manual_seed(self.global_step)
            num_ctx_views = torch.randint(
                2, batch['context']['extrinsics'].shape[1] + 1,
                size=(1,), generator=gen,
            ).item()
                        
            ctx = {}
            for key, value in batch['context'].items():
                if key == 'overlap':
                    ctx[key] = value
                    continue
                ctx[key] = value[:, :num_ctx_views, ...].contiguous()
            batch['context'] = ctx

            # 1-to-1: target views = context views (dataset_rsblur 已改为
            # tgt = ctx 一一对应). num_tgt_views == num_ctx_views.
            num_tgt_views = num_ctx_views
            n_tgt_orig = batch['target']['image'].shape[1]
            tgt = {}
            for key, value in batch['target'].items():
                if isinstance(value, torch.Tensor) and len(value.shape) > 1 and value.shape[1] == n_tgt_orig:
                    tgt[key] = value[:, :num_tgt_views, ...].contiguous()
                else:
                    tgt[key] = value
            batch['target'] = tgt

        # Run the model.
        visualization_dump = {}
        
        context_feature = self.forward_foundation_model(batch['context']['image']) if self.encoder.cfg.feature_dim else None
        _enc_ctx = torch.no_grad() if self.train_cfg.freeze_encoder else contextlib.nullcontext()
        with _enc_ctx:
            gaussians = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, context_feature = context_feature)

        # When DA3 predicts poses, propagate predicted context poses to target.
        # 1-to-1: target uses ALL context poses (dataset is now tgt = ctx).
        if getattr(self.encoder.cfg, 'use_pred_pose', False):
            batch["target"]["extrinsics"] = batch["context"]["extrinsics"].clone()
            batch["target"]["intrinsics"] = batch["context"]["intrinsics"].clone()

        # (depth_coarse pre-render removed: GT depth now comes from dataset)

        # Optional: refine Gaussians per-scene with Mip-Splatting optimization.
        if self.refiner is not None:
            gaussians = self.refiner(
                gaussians,
                _make_refiner_context(batch["context"], visualization_dump, self.train_cfg.use_vggt_poses_for_refiner),
            )

        # Optional: DiffusionHead flow-matching deblurrer (with camera conditioning).
        l_flow = torch.tensor(0.0, device=gaussians.means.device)
        l_cam  = torch.tensor(0.0, device=gaussians.means.device)
        if self.diffusion_head is not None:
            gaussians_blurry_pre = gaussians  # save x0 before refinement
            ctx_pose_enc = _compute_pose_enc(
                batch["context"]["extrinsics"], batch["context"]["intrinsics"], h, w
            )
            _dh_out = self.diffusion_head(
                gaussians,
                context_pose_enc=ctx_pose_enc,
                blurry_patch_tokens=visualization_dump.get('blurry_patch_tokens'),
            )
            # Design A returns (gaussians, pred_pose_enc); Design B returns gaussians
            if isinstance(_dh_out, tuple):
                gaussians, pred_pose_enc = _dh_out
                if ctx_pose_enc is not None and self.train_cfg.cam_loss_weight > 0:
                    l_cam = self.train_cfg.cam_loss_weight * F.mse_loss(
                        pred_pose_enc, ctx_pose_enc.to(pred_pose_enc.device)
                    )
                    self.log("loss/l_cam", l_cam, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            else:
                gaussians = _dh_out

            # ── Fix 4 (train): safe-clamp Gaussian anisotropy — same as val ──
            _aniso_delta_tr = float(os.environ.get('ANISO_CLAMP_DELTA', '0'))
            if _aniso_delta_tr > 0:
                gaussians = dc_replace(
                    gaussians,
                    covariances=_safe_clamp_aniso(gaussians.covariances, max_log_delta=_aniso_delta_tr),
                )

            # ── Debug stats for diagnosing ODE divergence (every 50 steps) ──
            # Must be called on ALL ranks symmetrically: self.log triggers an internal
            # metric reduce that uses the default NCCL communicator, so rank-0-only
            # logging causes a collective desync vs DDP gradient allreduce (rank 0
            # NumelIn=1 vs ranks 1-3 NumelIn=trainable params), reproduced as a
            # 30-min ALLREDUCE timeout at SeqNum=2775.
            if self.global_step % 50 == 0:
                with torch.no_grad():
                    _attrs = gaussians_to_attrs(gaussians)          # [B,G,D]
                    _quats = _attrs[:, :, 6:10]
                    _qnorm = _quats.norm(dim=-1)
                    _means = _attrs[:, :, 0:3]
                    self.log("debug/quats_norm_mean", _qnorm.mean(), on_step=True, logger=True, sync_dist=True)
                    self.log("debug/quats_norm_max",  _qnorm.max(),  on_step=True, logger=True, sync_dist=True)
                    self.log("debug/quats_norm_min",  _qnorm.min(),  on_step=True, logger=True, sync_dist=True)
                    self.log("debug/means_abs_max",   _means.abs().max(), on_step=True, logger=True, sync_dist=True)
                    self.log("debug/means_x",         _means[..., 0].mean(), on_step=True, logger=True, sync_dist=True)
                    self.log("debug/means_y",         _means[..., 1].mean(), on_step=True, logger=True, sync_dist=True)
                    self.log("debug/means_z",         _means[..., 2].mean(), on_step=True, logger=True, sync_dist=True)

            # ── L_flow: Chamfer(blurry_means, sharp_means) ───────────────────
            # Run frozen encoder on sharp context images (same views as blurry).
            # Same views → same internal coordinate frame → no alignment needed.
            flow_weight = self.train_cfg.l_flow_weight
            has_sharp = "sharp_image" in batch["context"]
            if flow_weight > 0 and has_sharp:
                _ENC_KEYS = {"image", "intrinsics", "extrinsics", "near", "far", "index"}
                sharp_ctx = {k: v for k, v in batch["context"].items() if k in _ENC_KEYS}
                sharp_ctx["image"] = batch["context"]["sharp_image"]

                with torch.no_grad():
                    gaussians_sharp = self.encoder(sharp_ctx, self.global_step)

                # Sub-sample to keep Chamfer tractable (512 pts each side)
                N_sub     = 512
                idx_blur  = torch.randperm(gaussians.means.shape[1],       device=gaussians.means.device)[:N_sub]
                idx_sharp = torch.randperm(gaussians_sharp.means.shape[1], device=gaussians.means.device)[:N_sub]
                l_flow = _chamfer_distance(
                    gaussians.means[:, idx_blur],
                    gaussians_sharp.means[:, idx_sharp],
                )
                self.log("loss/l_flow", l_flow, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        V_tgt = batch["target"]["image"].shape[1]
        V_ctx = batch["context"]["image"].shape[1]

        # Skip context renders when share_weight=0 to avoid wasting memory/compute
        # (e.g. 6 iPhone views rendered at 480×640 that are never used in any loss).
        if self.train_cfg.share_weight > 0 and V_ctx > 0:
            all_extrinsics = torch.cat([batch["target"]["extrinsics"], batch["context"]["extrinsics"]], dim=1)
            all_intrinsics = torch.cat([batch["target"]["intrinsics"], batch["context"]["intrinsics"]], dim=1)
            all_near       = torch.cat([batch["target"]["near"],       batch["context"]["near"]],       dim=1)
            all_far        = torch.cat([batch["target"]["far"],        batch["context"]["far"]],         dim=1)
        else:
            all_extrinsics = batch["target"]["extrinsics"]
            all_intrinsics = batch["target"]["intrinsics"]
            all_near       = batch["target"]["near"]
            all_far        = batch["target"]["far"]

        # Safety clamp: large 3D covariances cause tiles_touched uint32 overflow
        # in the rasterizer → 152 TB allocation.  DiffHead (freshly initialised)
        # can also produce outlier covariances on the first few steps.
        if getattr(self.encoder.cfg, 'use_pred_pose', False):
            from src.model.types import Gaussians as _Gaussians
            gaussians = _Gaussians(
                means=torch.nan_to_num(gaussians.means, nan=0.0, posinf=1e4, neginf=-1e4),
                covariances=torch.nan_to_num(gaussians.covariances).clamp(-0.1, 0.1),
                harmonics=torch.nan_to_num(gaussians.harmonics, nan=0.0),
                opacities=torch.nan_to_num(gaussians.opacities, nan=0.0).clamp(0.0, 1.0),
                feature=gaussians.feature,
            )

        output = self.decoder.forward(
            gaussians,
            all_extrinsics,
            all_intrinsics,
            all_near,
            all_far,
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
            global_step=self.global_step
        )

        # ── L_color: rendered target views vs sharp GT ────────────────────
        # Slice output to target views only for main photometric losses.
        output_tgt = dc_replace(output,
                                color=output.color[:, :V_tgt],
                                depth=output.depth[:, :V_tgt] if output.depth is not None else None,
                                feature=output.feature[:, :V_tgt] if output.feature is not None else None)

        sharp_gt = batch["target"]["image"]   # [B, V_tgt, 3, H, W] sharp frames [0,1]

        psnr_probabilistic = compute_psnr(
            rearrange(sharp_gt, "b v c h w -> (b v) c h w"),
            rearrange(output_tgt.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # ── L_color: context-view reconstruction (MSE + LPIPS, matches diagram) ──
        # Render at context poses, compare with blurry context GT.
        share_weight = self.train_cfg.share_weight
        if share_weight > 0 and V_ctx > 0:
            ctx_color_pred = output.color[:, V_tgt:]             # [B, V_ctx, 3, H, W]
            ctx_color_gt   = (batch["context"]["image"] + 1) / 2  # [0,1] blurry GT
            l_color_mse  = F.mse_loss(ctx_color_pred, ctx_color_gt)
            # LPIPS component: reuse the existing lpips loss instance
            lpips_fn = next((fn for fn in self.losses if isinstance(fn, LossLpips)), None)
            if lpips_fn is not None and self.global_step >= lpips_fn.cfg.apply_after_step:
                l_color_lpips = lpips_fn.lpips(
                    rearrange(ctx_color_pred, "b v c h w -> (b v) c h w"),
                    rearrange(ctx_color_gt,   "b v c h w -> (b v) c h w"),
                    normalize=True,
                ).mean()
            else:
                # Graph-connected zero (use ctx_color_pred from output) — DDP must
                # see the same trainable-param dependency on every rank.
                l_color_lpips = (ctx_color_pred * 0).sum()
            l_color = l_color_mse + lpips_fn.cfg.weight * l_color_lpips if lpips_fn is not None else l_color_mse
            self.log("loss/l_color", l_color, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        else:
            # Graph-connected zero so total_loss has a consistent grad path on every rank.
            l_color = (output.color * 0).sum()

        # ── Main losses (L_render): target-view photometric (MSE+LPIPS) + L_depth ──
        # All computed on TARGET views only; LossDepth reads batch["target"]["depth"].
        total_loss = share_weight * l_color + self.train_cfg.l_flow_weight * l_flow + l_cam
        for loss_fn in self.losses:
            loss = loss_fn.forward(output_tgt, batch, gaussians, self.global_step, target_image=sharp_gt)
            self.log(f"loss/{loss_fn.name}", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            total_loss = total_loss + loss
            
        if self.train_cfg.feature_rendering_loss > 0:
            B,CV,_,H,W = batch['context']['image'].shape
            B,TV,_,H,W = batch['target']['image'].shape
            feature = self.forward_foundation_model(torch.cat((batch["context"]["image"], batch['target']['image'] * 2 - 1), dim=1), interpolate=False)
            feature = torch.cat((feature[:,CV:], feature[:,:CV]), dim=1)        ## ordering: target -> context
            
            gaussian_feature = output.feature
            B,N,_,FH,FW = feature.shape
            
            gaussian_feature = F.interpolate(gaussian_feature.reshape(B*N,-1,H,W), size=(FH, FW), mode='bilinear', align_corners=False).reshape(B,N,-1,FH,FW)
            
            gaussian_feature = F.normalize(gaussian_feature, p=2, dim=2)
            feature = F.normalize(feature, p=2, dim=2)
            
            feature_rendering_loss = F.cosine_similarity(gaussian_feature, feature.detach(), dim=2)                
            feature_rendering_loss = (1 - feature_rendering_loss).mean()
            
            self.log("loss/feature_rendering_loss", feature_rendering_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            total_loss = total_loss + self.train_cfg.feature_rendering_loss * feature_rendering_loss

        # ── Hinge aniso loss: gradient pressure to keep Gaussians spherical ──
        # Clamp alone caps anisotropy at the ceiling but model can still push
        # everything toward the ceiling (visible needles re-emerge as training
        # progresses). Hinge gives non-zero gradient to Gaussians whose log
        # scale-ratio exceeds ANISO_HINGE_TAU, pushing them toward more spherical.
        # log_aniso = ln(max_scale / min_scale) per gaussian; tau=4 → max ratio exp(4)≈55.
        ANISO_HINGE_W = float(os.environ.get('ANISO_HINGE_WEIGHT', '0'))
        if ANISO_HINGE_W > 0:
            ANISO_HINGE_TAU = float(os.environ.get('ANISO_HINGE_TAU', '4.0'))
            with torch.enable_grad():  # ensure grad path
                # v7: floor 1e-6 (was 1e-12) — log backward = 1/eigval, 1e12 overflows bf16.
                _eigvals = torch.linalg.eigvalsh(gaussians.covariances.float()).clamp(min=1e-6)
                _log_s = 0.5 * _eigvals.log()
                _log_aniso = _log_s[..., -1] - _log_s[..., 0]    # log(max_scale/min_scale)
                l_aniso = torch.relu(_log_aniso - ANISO_HINGE_TAU).mean()
            total_loss = total_loss + ANISO_HINGE_W * l_aniso
            self.log("loss/aniso_hinge", l_aniso, on_step=True, on_epoch=False, logger=True)

        self.log("loss/total", total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        if self.global_rank == 0:
            self._epoch_loss_sum += total_loss.detach().item()
            self._epoch_loss_count += 1


        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
            and (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}",
                f"low_pass_filter = {self.decoder.low_pass_filter:.3f}",
            )
        self.log("info/global_step", self.global_step, on_step=True, on_epoch=False, prog_bar=True, logger=True)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    @torch.no_grad()
    def forward_foundation_model(self, input_image, interpolate=True, vggt_tracking=False):
        B, V, C, H, W = input_image.shape       ## [-1~1]
                
        with torch.no_grad():              
            if self.train_cfg.reproj_model == 'dinov2':
                context_feature = self.dino_model.get_intermediate_layers(input_image.reshape(B*V,C,H,W), reshape=True)[0].reshape(B,V,-1,H//14,W//14)
                
            elif 'dinov3' in self.train_cfg.reproj_model:
                context_feature = self.dino_model(**self.dino_processor((input_image.reshape(B*V,C,H,W) + 1)/2. * 255, return_tensors='pt').to(self.device))
                context_feature = rearrange(context_feature['last_hidden_state'][:,5:], 'b (h w) c -> b c h w', h=H//16, w=W//16)
                context_feature = context_feature.reshape(B,V,-1,H//16,W//16)
                
            elif self.train_cfg.reproj_model == 'lseg':
                context_feature = self.lseg_feature_extractor.extract_features(input_image.reshape(B*V,3,H,W))
                context_feature = context_feature.reshape(B,V,-1,H//2,W//2)
                
            elif 'vggt' in self.train_cfg.reproj_model:
                if self.train_cfg.reproj_model=='vggt_tracking':
                    context_feature = self.vggt(input_image)['feature']
                else:
                    aggregated_tokens_list, patch_start_idx = self.vggt(input_image, only_feature=True)
                    vggt_features = aggregated_tokens_list[-1][:,:,patch_start_idx:]
                    context_feature = rearrange(vggt_features, "b n (h w) c -> b n c h w", h=H//14, w=W//14)

            elif self.train_cfg.reproj_model == 'maskclip':
                input_image = (input_image + 1) / 2
                mean, std = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device), torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device)
                input_image = (input_image - mean[None,None,:,None,None]) / std[None,None,:,None,None]
                context_feature = self.clip_model(input_image.reshape(B*V,C,H,W))
                context_feature = context_feature.reshape(B,V,-1,H,W)            

        if interpolate:
            context_feature = F.interpolate(context_feature.reshape(B*V,-1,context_feature.shape[-2],context_feature.shape[-1]), size=(H//14, W//14), mode='bilinear', align_corners=False).reshape(B,V,-1,H//14,W//14)
                
        ## B, V, C, H, W
        return context_feature
    


    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, v, _, h, w = batch["target"]["image"].shape
        
        if h!=224 or w!=224:
            b, cv, _, ch, cw = batch['context']['image'].shape
            batch['context']['image'] = F.interpolate(batch['context']['image'].reshape(b*cv,3,ch,cw), size=(224,224), mode='bilinear', align_corners=False).reshape(b,cv,3,224,224)
        
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        if self.test_cfg.visualize_gaussian_token>=0:
            outputs = []
            def hook_fn(module, input, output):
                outputs.append(output)
            _ = self.encoder.gmae_decoder.layers[0][0].to_qkv.register_forward_hook(hook_fn)
            _ = self.encoder.gmae_decoder.layers[1][0].to_qkv.register_forward_hook(hook_fn)
            
        # Render Gaussians.
        context_feature = self.forward_foundation_model(batch['context']['image']) if self.encoder.cfg.feature_dim else None
        visualization_dump = {}
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                visualization_dump=visualization_dump,
                context_feature=context_feature
            )

        if self.test_cfg.visualize_gaussian_token>=0:
            num_heads = self.encoder.gmae_decoder.layers[0][0].heads
            gaussian_token_idx = self.test_cfg.visualize_gaussian_token
            
            name = get_cfg()["wandb"]["name"]
            path = self.test_cfg.output_path / name
            (scene,) = batch["scene"]
            os.makedirs(path / scene , exist_ok=True)
            
            visualize_attention_map(
                outputs[0],batch, num_heads, gaussian_token_idx, batch['context']['image'].shape[3:5], patch_size = self.encoder.patch_size, output_path= path / scene / f"{gaussian_token_idx}_layer1")
            
            visualize_attention_map(
                outputs[1], batch, num_heads, gaussian_token_idx, batch['context']['image'].shape[3:5], patch_size = self.encoder.patch_size, output_path= path / scene /  f"{gaussian_token_idx}_layer2")
            
            C0 = 0.28209479177387814
            gaussians.harmonics[:, gaussian_token_idx, :, 0] = (torch.tensor([[1,0,0]]) - 0.5) / C0
        
        # Mip-Splatting per-scene refinement (if enabled)
        if self.refiner is not None:
            with self.benchmarker.time("mip_refiner"):
                gaussians = self.refiner(
                    gaussians,
                    _make_refiner_context(batch["context"], visualization_dump, self.train_cfg.use_vggt_poses_for_refiner),
                )

        # DiffusionHead flow-matching deblurrer (if enabled, with camera conditioning)
        if self.diffusion_head is not None:
            with self.benchmarker.time("diffusion_head"):
                _b2, _, _, _h2, _w2 = batch["target"]["image"].shape
                _ctx_pose_enc2 = _compute_pose_enc(
                    batch["context"]["extrinsics"], batch["context"]["intrinsics"], _h2, _w2
                )
                _dh_out = self.diffusion_head(
                    gaussians,
                    context_pose_enc=_ctx_pose_enc2,
                    blurry_patch_tokens=visualization_dump.get('blurry_patch_tokens'),
                )
                gaussians = _dh_out[0] if isinstance(_dh_out, tuple) else _dh_out

        # TTO: refine Gaussians on context views before rendering
        if self.test_cfg.use_tto:
            with self.benchmarker.time("tto"):
                gaussians = self.test_step_tto(batch, gaussians, verbose=True)

        if self.test_cfg.align_pose and (not self.test_cfg.forward_vfm):
            output = self.test_step_align(batch, gaussians, verbose=True)
        else:
            with self.benchmarker.time("decoder", num_calls=v):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )

        # compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0] if "overlap" in batch["context"] else None
            overlap_tag = get_overlap_tag(overlap) if overlap is not None else "all"

            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            
            psnr = compute_psnr(rgb_gt, rgb_pred).mean()
            all_metrics = {
                f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr_ours": psnr,
            }
            methods = ['ours']

            self.log_dict(all_metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)

        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], output.color[0]):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in output.color[0]],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )
            
        projections = render_projections(gaussians,256,extra_label="",low_pass = self.decoder.low_pass_filter, draw_label=False)[0]
        save_image(projections[2], path / f"{scene}_projections.png")
        
        if self.test_cfg.save_compare or isinstance(self.logger, WandbLogger):
            context_img = inverse_normalize(batch["context"]["image"][0])
            error_map = (rgb_gt - rgb_pred.clamp(0,1)).abs()
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
                add_label(vcat(*error_map), "Error Map"),
            )
            if self.test_cfg.save_compare:
                save_image(comparison, path / f"{scene}_{psnr:.3f}.png")
            if isinstance(self.logger, WandbLogger) and batch_idx % 50 == 0:
                try:
                    self.logger.log_image(
                        "test/comparison",
                        [prep_image(comparison)],
                        step=batch_idx,
                        caption=[f"{scene} PSNR={psnr:.2f}"],
                    )
                except Exception as e:
                    print(f"[WARN] test log_image failed: {e}")
            
            if self.encoder.cfg.feature_dim:
                gaussian_feature = output.feature
                B,N,C,H,W = gaussian_feature.shape

                if 'dinov2' in self.train_cfg.reproj_model or 'vggt' in self.train_cfg.reproj_model:
                    B,V,C,H,W = batch['target']['image'].shape
                    target_image = F.interpolate(batch['target']['image'].reshape(B*V,C,H,W), size=(224,224), mode='bilinear', align_corners=False).reshape(B,V,C,224,224)
                else:
                    target_image = batch['target']['image']

                foundation_feature = self.forward_foundation_model((target_image * 2 - 1),interpolate=False)
                
                save_dir = path / scene / "seg"
                save_gt_dir = path / scene / "seg_gt"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_gt_dir.mkdir(parents=True, exist_ok=True)

                if 'dino' not in self.train_cfg.reproj_model and 'vggt' not in self.train_cfg.reproj_model:
                    pca_images, pca_vggt_images = [], []
                    V = gaussian_feature.shape[1]
                    
                    for i in range(foundation_feature.shape[1]):
                        pca_gaussian_img = run_pca(gaussian_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                        pca_images.append(pca_gaussian_img.squeeze(dim=0))
                        
                        pca_vggt_img = run_pca(foundation_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                        pca_vggt_images.append(pca_vggt_img.squeeze(dim=0))
                    cocnat_pca_images = run_pca(gaussian_feature[0], (H,W))
                    cocnat_pca_vggt_images = run_pca(foundation_feature[0], (H,W)) 
                    
                    preds = []
                    targets = []
                    for i, (index, g_upfeat) in enumerate(zip(batch["target"]["index"][0], gaussian_feature)):
                        if self.test_cfg.forward_vfm:
                            g_upfeat = foundation_feature[i]
                            if 'lseg' in self.train_cfg.reproj_model:
                                g_upfeat = self.lseg_feature_extractor.scratch.output_conv(g_upfeat)
                        
                        if 'text' in batch['target'].keys():
                            labelset = batch['target']['text']
                            labelset = [label[0] for label in labelset]
                        else:  
                            labelset = self.test_cfg.labels
                        
                        if self.train_cfg.reproj_model == 'lseg':
                            pred = self.lseg_feature_extractor.decode_feature(g_upfeat, labelset=labelset)
                        else:
                            pred = self.clip_decode_feature(g_upfeat, labelset=labelset)
                                                    
                        pred = torch.argmax(pred, dim=1) + 1

                        target = batch["target"]["label"][0]
                        targets.append(target)
                        
                        iou_val = self.miou(pred.flatten(), target.flatten())
                        acc_val = self.acc(pred.flatten(), target.flatten())

                        self.log("test/miou", iou_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                        self.log("test/acc", acc_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

                        print(f"IoU: {iou_val.item()}, Acc: {acc_val.item()}")

                        self.per_image_ious.append(iou_val.item())
                        self.per_image_accs.append(acc_val.item())
                        
                        preds.append(pred)

                    seg_preds = []
                    seg_tgts = []
                            
                    for pred in preds:
                        for index, seg_pred in zip(batch["target"]["index"][0], pred):
                            labels = self.test_cfg.labels[:8]
                            seg_pred_vis = save_segmap(gaussian_feature, seg_pred, index, save_dir, labels, self.test_cfg.color_hex_list)
                            seg_preds.append(seg_pred_vis)
                            
                    for target in targets:
                        for index, seg_tgt in zip(batch["target"]["index"][0], target):
                            labels = self.test_cfg.labels[:8]
                            seg_tgt_vis = save_segmap(gaussian_feature, seg_tgt, index, save_gt_dir, labels, self.test_cfg.color_hex_list)
                            seg_tgts.append(seg_tgt_vis)
                            


                    
                    comparison = hcat(
                        add_label(vcat(*context_img), "Context"),
                        add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                        add_label(vcat(*pca_vggt_images), "Feature (VFM)"),
                        add_label(vcat(*pca_images), "Feature (Pred)"),
                        add_label(vcat(*cocnat_pca_vggt_images), "Feature_cat (VFM)"),
                        add_label(vcat(*cocnat_pca_images), "Feature_cat (Pred)"),
                        add_label(vcat(*seg_preds), "Segmentation (Pred)"),
                        add_label(vcat(*seg_tgts), "Segmentation (GT)"),
                    )
                    save_image(comparison, path / f"{scene}_{psnr:.3f}_pca.png")
                    
                else:
                    pca_images, pca_vggt_images = [], []
                    
                    V = gaussian_feature.shape[1]
                    
                    for i in range(foundation_feature.shape[1]):
                        pca_gaussian_img = run_pca(gaussian_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                        pca_images.append(pca_gaussian_img.squeeze(dim=0))
                        
                        pca_vggt_img = run_pca(foundation_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                        pca_vggt_images.append(pca_vggt_img.squeeze(dim=0))
                        
                    cocnat_pca_images = run_pca(gaussian_feature[0], (H,W))
                    cocnat_pca_vggt_images = run_pca(foundation_feature[0], (H,W)) 

                    for i in range(len(cocnat_pca_images)):
                        save_image(cocnat_pca_images[i], path / f"{scene}_cocnat_pca{i}.png")

                    comparison = hcat(
                        add_label(vcat(*context_img), "Context"),
                        add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                        add_label(vcat(*pca_vggt_images), "Feature (VFM)"),
                        add_label(vcat(*pca_images), "Feature (Pred)"),
                        add_label(vcat(*cocnat_pca_vggt_images), "Feature_cat (VFM)"),
                        add_label(vcat(*cocnat_pca_images), "Feature_cat (Pred)"),
                    )
                    save_image(comparison, path / f"{scene}_{psnr:.3f}_pca.png")
                    
    @torch.no_grad()
    def clip_decode_feature(self, image_features, labelset=''):
        imshape = image_features.shape      # B C H W
        
        text = clip.tokenize(labelset)
        
        text = text.to(image_features.device)
        if 'maskclip' in self.train_cfg.reproj_model:
            text_features = self.clip_model.model.model.encode_text(text)
        else:
            text_features = self.clip_model.encode_text(text)
        image_features = image_features.permute(0,2,3,1).reshape(-1, image_features.shape[1])
        
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = image_features.half() @ text_features.t()
        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        return out

                    
    # image-level iou and acc
    def on_test_epoch_end(self):
        mean_iou = sum(self.per_image_ious) / len(self.per_image_ious) if self.per_image_ious else 0.0
        mean_acc = sum(self.per_image_accs) / len(self.per_image_accs) if self.per_image_accs else 0.0

        print("mIoU:", mean_iou)
        print("Acc:", mean_acc)

        self.log("test/mIoU", mean_iou, prog_bar=True)
        self.log("test/Acc", mean_acc, prog_bar=True)

        # Reset lists for next epoch
        self.per_image_ious.clear()
        self.per_image_accs.clear()

    def test_step_align(self, batch, gaussians, verbose=False):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()
            
            if verbose:
                logger = tqdm(range(self.test_cfg.pose_align_steps))
            else:
                logger = range(self.test_cfg.pose_align_steps)
                
            prev_loss = None
            patience_counter = 0
            patience_limit = 10 
            
            with self.benchmarker.time("optimize"):
                for i in logger:
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                    )

                    # Compute and log loss.
                    total_loss = 0

                    
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step, target_image=batch["target"]["image"])
                        total_loss = total_loss + loss
                        
                    if verbose:
                        logger.set_description(f"pose optim step {i}; loss = {total_loss:.6f}")

                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

                    if prev_loss is not None:
                        delta = abs(total_loss.item() - prev_loss)
                        if delta < 0.00001:
                            patience_counter += 1
                            if patience_counter >= patience_limit and i >= 100:
                                break
                        else:
                            patience_counter = 0
                    prev_loss = total_loss.item()


        # Render Gaussians.
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        del pose_optimizer

        return output

    def test_step_tto(self, batch, gaussians, verbose=True):
        """
        Test-Time Optimization (TTO):
        Fix the encoder, optimise the 2048 Gaussian parameters directly
        using the CONTEXT views as supervision signal.
        Returns refined Gaussians ready for novel-view rendering.
        """
        from .types import Gaussians as GaussiansType

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, cv, _, h, w = batch["context"]["image"].shape

        with torch.set_grad_enabled(True):
            # ---- make Gaussian parameters learnable ----
            means       = nn.Parameter(gaussians.means.clone().detach())
            harmonics   = nn.Parameter(gaussians.harmonics.clone().detach())
            opacities   = nn.Parameter(gaussians.opacities.clone().detach())
            covariances = nn.Parameter(gaussians.covariances.clone().detach())

            lr     = self.test_cfg.tto_lr
            geo_lr = lr * self.test_cfg.tto_geo_lr_scale

            optimizer = torch.optim.Adam([
                {"params": [harmonics],   "lr": lr},      # appearance
                {"params": [opacities],   "lr": lr},      # opacity
                {"params": [means],       "lr": geo_lr},  # geometry (slower)
                {"params": [covariances], "lr": geo_lr},
            ])

            steps = self.test_cfg.tto_steps
            logger = tqdm(range(steps), desc="TTO") if verbose else range(steps)

            prev_loss = None
            patience_counter = 0

            for i in logger:
                optimizer.zero_grad()

                opt_gaussians = GaussiansType(
                    means=means,
                    covariances=covariances,
                    harmonics=harmonics,
                    opacities=opacities,
                    feature=gaussians.feature,
                )

                # Render context views and compute loss against GT context images
                ctx_output = self.decoder.forward(
                    opt_gaussians,
                    batch["context"]["extrinsics"],
                    batch["context"]["intrinsics"],
                    batch["context"]["near"],
                    batch["context"]["far"],
                    (h, w),
                )

                loss = F.mse_loss(ctx_output.color, batch["context"]["image"])

                if verbose:
                    logger.set_description(f"TTO step {i}; loss = {loss.item():.6f}")

                loss.backward()
                optimizer.step()

                # Early stopping
                if prev_loss is not None and abs(loss.item() - prev_loss) < 1e-6:
                    patience_counter += 1
                    if patience_counter >= 20 and i >= 50:
                        break
                else:
                    patience_counter = 0
                prev_loss = loss.item()

        # Return refined (detached) Gaussians
        refined = GaussiansType(
            means=means.detach(),
            covariances=covariances.detach(),
            harmonics=harmonics.detach(),
            opacities=opacities.detach(),
            feature=gaussians.feature,
        )
        del optimizer
        return refined

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {}

        context_feature = self.forward_foundation_model(batch['context']['image']) if self.encoder.cfg.feature_dim else None

        gaussians = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump, context_feature = context_feature)

        if getattr(self.encoder.cfg, 'use_pred_pose', False):
            # Replace target poses with predicted (context) poses. The dataset's
            # mapping from tgt to ctx may be either tgt = ctx[:n_tgt] (e.g.
            # blur_replica) or tgt = ctx[::2] (e.g. dataset_rsblur). Use the
            # explicit "index" field to recover the correct mapping rather than
            # assuming a fixed pattern (the prior ctx[:, :n_tgt] heuristic was
            # wrong for RSBlur and rendered tgt at sequential ctx[0..n_tgt]
            # poses while GT was at ctx[0,2,4,...] poses → val/psnr undercount).
            n_tgt = batch["target"]["image"].shape[1]
            ctx_idx = batch["context"]["index"][0].tolist()
            tgt_idx = batch["target"]["index"][0].tolist()
            ctx_pos_for_tgt = [ctx_idx.index(t) for t in tgt_idx[:n_tgt]]
            batch["target"]["extrinsics"] = batch["context"]["extrinsics"][:, ctx_pos_for_tgt].clone()
            batch["target"]["intrinsics"] = batch["context"]["intrinsics"][:, ctx_pos_for_tgt].clone()
            # Diagnostic: optionally force tgt intrinsics to identity-like
            # (f=1.0, cx=cy=0.5) instead of DA3-predicted (which can vary 0.1-10).
            # If FORCE_IDENTITY_INTRINSICS=1 and val/psnr jumps significantly,
            # confirms intrinsics mismatch is contributing to artifacts.
            import os as _os_diag2
            if _os_diag2.environ.get('FORCE_IDENTITY_INTRINSICS', '0') == '1':
                _K_id = torch.eye(3, device=batch["target"]["intrinsics"].device,
                                  dtype=batch["target"]["intrinsics"].dtype)
                _K_id[0, 0] = 1.0; _K_id[1, 1] = 1.0
                _K_id[0, 2] = 0.5; _K_id[1, 2] = 0.5
                batch["target"]["intrinsics"] = _K_id.view(1, 1, 3, 3).expand(
                    batch["target"]["intrinsics"].shape).contiguous()
            # DIAGNOSTIC: print predicted intrinsics for first batch to see if
            # DA3-predicted (fx, fy, cx, cy) differs from identity (1, 1, 0.5, 0.5)
            # in a way that explains the "bottom cropping" appearance.
            if batch_idx == 0 and self.global_rank == 0:
                # ── DIAG A: render camera == target camera? ──
                ctx_idx_full = batch["context"]["index"][0].cpu().tolist()
                tgt_idx_full = batch["target"]["index"][0].cpu().tolist()
                ctx_t = batch["context"]["extrinsics"][0, :, :3, 3]  # [V_ctx, 3]
                render_t = batch["target"]["extrinsics"][0, :, :3, 3]  # [V_tgt, 3]
                print(f"[DIAG A] ctx_idx={ctx_idx_full[:8]}")
                print(f"[DIAG A] tgt_idx={tgt_idx_full[:8]}  (subset of ctx)")
                print(f"[DIAG A] ctx_pos_for_tgt={ctx_pos_for_tgt[:8]}  (positions in ctx list)")
                print(f"[DIAG A] ctx_t (translations) per view:")
                for i, v in enumerate(ctx_t[:8].cpu().tolist()):
                    print(f"           ctx[{i:>2}] t=[{v[0]:+7.3f}, {v[1]:+7.3f}, {v[2]:+7.3f}]")
                print(f"[DIAG A] render_t (target extrinsics used by decoder):")
                for i, v in enumerate(render_t[:8].cpu().tolist()):
                    src = ctx_pos_for_tgt[i] if i < len(ctx_pos_for_tgt) else '?'
                    print(f"           render[{i:>2}] t=[{v[0]:+7.3f}, {v[1]:+7.3f}, {v[2]:+7.3f}]  ←supposedly ctx[{src}]")
                # Sanity check: render_t[i] should equal ctx_t[ctx_pos_for_tgt[i]]
                ok = True
                for i in range(min(len(ctx_pos_for_tgt), render_t.shape[0])):
                    expected = ctx_t[ctx_pos_for_tgt[i]]
                    actual = render_t[i]
                    if (expected - actual).abs().max().item() > 1e-5:
                        print(f"[DIAG A] MISMATCH at tgt[{i}]: expected {expected.cpu().tolist()}, got {actual.cpu().tolist()}")
                        ok = False
                print(f"[DIAG A] render==ctx[ctx_pos_for_tgt] check: {'PASS' if ok else 'FAIL'}")

                # ── DIAG B: translation variance across burst ──
                t_std = ctx_t.std(dim=0)  # [3] std per axis
                t_range = ctx_t.max(dim=0).values - ctx_t.min(dim=0).values  # [3]
                print(f"[DIAG B] translation std  per axis (xyz): "
                      f"[{t_std[0].item():.4f}, {t_std[1].item():.4f}, {t_std[2].item():.4f}]")
                print(f"[DIAG B] translation range per axis (xyz): "
                      f"[{t_range[0].item():.4f}, {t_range[1].item():.4f}, {t_range[2].item():.4f}]")
                print(f"[DIAG B] y_range = {t_range[1].item():.4f}  (>0.05 = catastrophic for static rig)")

                # ── DIAG C: Gaussian anisotropy distribution ──
                # Already computed eigvals above; reuse
                _eigvals = torch.linalg.eigvalsh(gaussians.covariances.float()).clamp(min=1e-12)
                _scales = _eigvals.sqrt()
                _log_aniso = (_scales[..., -1].log() - _scales[..., 0].log()).flatten()  # log(max/min) per gaussian
                print(f"[DIAG C] log_aniso mean={_log_aniso.mean().item():.3f}  "
                      f"p50={torch.quantile(_log_aniso, 0.50).item():.3f}  "
                      f"p95={torch.quantile(_log_aniso, 0.95).item():.3f}  "
                      f"p99={torch.quantile(_log_aniso, 0.99).item():.3f}  "
                      f"max={_log_aniso.max().item():.3f}")
                print(f"[DIAG C] (recall: log_aniso=ln(max_scale/min_scale); 0=spherical, 5+=needle)")

                # Old INTR/EXT dump kept for reference
                _K = batch["target"]["intrinsics"][0]
                print(f"[INTR] fx={_K[..., 0, 0].mean().item():.3f} fy={_K[..., 1, 1].mean().item():.3f} "
                      f"cx={_K[..., 0, 2].mean().item():.3f} cy={_K[..., 1, 2].mean().item():.3f}")

        # Log coarse Gaussian count (before any refinement).
        n_gaussians_coarse = gaussians.means.shape[1]
        self.log("info/n_gaussians_coarse", float(n_gaussians_coarse), on_step=True, logger=True)

        # Apply MipSplattingRefiner in val if present (enabled for eval-only runs).
        if self.refiner is not None:
            _, _, _, h_ref, w_ref = batch["target"]["image"].shape
            gaussians = self.refiner(
                gaussians,
                _make_refiner_context(batch["context"], visualization_dump, self.train_cfg.use_vggt_poses_for_refiner),
                h=h_ref, w=w_ref,
            )
            n_gaussians_refined = gaussians.means.shape[1]
            self.log("info/n_gaussians_refined", float(n_gaussians_refined), on_step=True, logger=True)

        # Apply DiffusionHead (skip MipSplattingRefiner in val: too slow for rank-0-only step).
        if self.diffusion_head is not None:
            _b, _, _, _h, _w = batch["target"]["image"].shape
            _ctx_pose_enc = _compute_pose_enc(
                batch["context"]["extrinsics"], batch["context"]["intrinsics"], _h, _w
            )
            _dh_out = self.diffusion_head(
                gaussians,
                context_pose_enc=_ctx_pose_enc,
                blurry_patch_tokens=visualization_dump.get('blurry_patch_tokens'),
            )
            gaussians = _dh_out[0] if isinstance(_dh_out, tuple) else _dh_out

        # ── Fix 4 (val): safe-clamp Gaussian anisotropy ──────────────────
        # Cap |log_scale - mean_log_scale| <= ANISO_CLAMP_DELTA per gaussian
        # so max_scale/min_scale <= exp(2 * delta). Direct hard constraint on
        # needle Gaussians without retraining (effective when AT inference,
        # the model has learned to produce needles but we want to render with
        # bounded anisotropy).
        _aniso_delta = float(os.environ.get('ANISO_CLAMP_DELTA', '0'))
        if _aniso_delta > 0:
            from dataclasses import replace as _dc_replace
            gaussians = _dc_replace(
                gaussians,
                covariances=_safe_clamp_aniso(gaussians.covariances, max_log_delta=_aniso_delta),
            )

        # ── DIAGNOSTIC: Gaussian anisotropy ──────────────────────────────
        # Compute eigenvalues of each Gaussian's covariance and log the
        # anisotropy ratio = max_scale / min_scale (per gaussian). A
        # spherical Gaussian has ratio=1; a "needle" (degenerate Gaussian)
        # has ratio >> 1. If this ratio increases over training, model is
        # learning needle-shaped Gaussians as a defensive strategy against
        # pose noise, producing visible needle artifacts at val viewpoints.
        with torch.no_grad():
            cov = gaussians.covariances  # [B, G, 3, 3]
            # eigvalsh on symmetric SPD matrix; outputs sorted ascending.
            eigvals = torch.linalg.eigvalsh(cov.float())  # [B, G, 3]
            eigvals = eigvals.clamp(min=1e-12)
            scales = eigvals.sqrt()                       # [B, G, 3]
            scale_max = scales[..., -1]                    # largest
            scale_min = scales[..., 0]                     # smallest
            anisotropy = scale_max / scale_min             # [B, G]
            log_aniso = torch.log10(anisotropy.clamp(min=1.0))  # log10 ratio
            self.log("diag/aniso_mean", anisotropy.mean(), on_step=False, on_epoch=True, sync_dist=True, logger=True)
            self.log("diag/aniso_max",  anisotropy.max(),  on_step=False, on_epoch=True, sync_dist=True, logger=True)
            self.log("diag/log_aniso_mean", log_aniso.mean(), on_step=False, on_epoch=True, sync_dist=True, logger=True)
            self.log("diag/log_aniso_p99",  torch.quantile(log_aniso.flatten(), 0.99), on_step=False, on_epoch=True, sync_dist=True, logger=True)
            self.log("diag/scale_max_mean", scale_max.mean(), on_step=False, on_epoch=True, sync_dist=True, logger=True)
            self.log("diag/scale_min_mean", scale_min.mean(), on_step=False, on_epoch=True, sync_dist=True, logger=True)
            if batch_idx == 0 and self.global_rank == 0:
                print(f"[DIAG step={self.global_step}] aniso mean={anisotropy.mean().item():.2f} max={anisotropy.max().item():.2f} | "
                      f"log10_aniso mean={log_aniso.mean().item():.3f} p99={torch.quantile(log_aniso.flatten(), 0.99).item():.3f}")

        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            "depth",
        )
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        # ── DIAG: coverage proxy via near-bg fraction (bg=[0,0,0] black) ──
        # Distinguishes "true camera crop" (alpha exists everywhere, content
        # shifted) from "Gaussian coverage collapse" (no Gaussians cover bottom).
        # near-bg = all RGB channels < 0.05 → effectively black → no Gaussian.
        if batch_idx == 0 and self.global_rank == 0:
            with torch.no_grad():
                _rgb = output.color[0]                          # [V, 3, H, W]
                _near_bg = (_rgb.abs() < 0.05).all(dim=1)       # [V, H, W] mask
                _H = _rgb.shape[-2]
                _bottom_mask = torch.zeros_like(_near_bg, dtype=torch.bool)
                _bottom_mask[..., int(0.8 * _H):, :] = True
                _top_mask = torch.zeros_like(_near_bg, dtype=torch.bool)
                _top_mask[..., :int(0.2 * _H), :] = True
                _full_bg = _near_bg.float().mean().item()
                _bot_bg = (_near_bg & _bottom_mask).float().sum().item() / _bottom_mask.float().sum().clamp(min=1).item()
                _top_bg = (_near_bg & _top_mask).float().sum().item() / _top_mask.float().sum().clamp(min=1).item()
                print(f"[COVER] near-bg fraction: full={_full_bg:.4f} bottom20%={_bot_bg:.4f} top20%={_top_bg:.4f}")
                print(f"[COVER] bottom/full ratio = {(_bot_bg/max(_full_bg,1e-6)):.2f}  (>>1 = bottom coverage collapse, ≈1 = uniform coverage shift)")

        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # Compute validation metrics — all ranks, sync_dist aggregates automatically.
        rgb_gt = batch["target"]["image"][0]

        # ── DIAG: pred-vs-gt vs pred-vs-ctx, plus shuffled-target sanity test ──
        # Toggle with DIAG_PRED_DIST=1. Logs once per val (first batch, rank 0).
        import os as _os_d
        if _os_d.environ.get("DIAG_PRED_DIST", "0") == "1" and batch_idx == 0 and self.global_rank == 0:
            try:
                _ctx = batch["context"]["image"][0]                     # [V, 3, H, W] blur
                _gt  = rgb_gt                                            # [V, 3, H, W] sharp
                _pred = rgb_pred.clamp(0, 1)
                # Align ctx view count to gt/pred view count if they differ (use_pred_pose
                # path subsamples target; with use_pred_pose=false they should match).
                _Vp = _pred.shape[0]
                _Vc = _ctx.shape[0]
                if _Vc != _Vp:
                    _ctx_aligned = _ctx[:_Vp]
                else:
                    _ctx_aligned = _ctx
                _d_pg = (_pred - _gt).abs().mean().item()
                _d_pc = (_pred - _ctx_aligned).abs().mean().item()
                _d_cg = (_ctx_aligned - _gt).abs().mean().item()
                # Shuffle GT views within sample. Skip view 0 → wrap perm to ensure all
                # views actually move to a different position.
                import torch as _torch_d
                _V = _gt.shape[0]
                _perm = _torch_d.randperm(_V)
                while (_perm == _torch_d.arange(_V)).all() and _V > 1:
                    _perm = _torch_d.randperm(_V)
                _gt_shuf = _gt[_perm]
                _d_p_shuf = (_pred - _gt_shuf).abs().mean().item()
                _d_g_shuf = (_gt - _gt_shuf).abs().mean().item()
                print(f"[DIAG_PRED_DIST] step={self.global_step}  scene={batch.get('scene', '?')}")
                print(f"  |pred - gt|       = {_d_pg:.5f}    <-- want LOW")
                print(f"  |pred - ctx|      = {_d_pc:.5f}    <-- want HIGH(er than pred-gt)")
                print(f"  |ctx  - gt|       = {_d_cg:.5f}    <-- baseline gap (the deblur task)")
                print(f"  |pred - gt_shuf|  = {_d_p_shuf:.5f}  <-- want HIGHER than |pred-gt|")
                print(f"  |gt   - gt_shuf|  = {_d_g_shuf:.5f}  <-- floor of view-level differences")
                if _d_pg < _d_pc:
                    print(f"  → pred CLOSER to GT than to ctx ({_d_pc - _d_pg:+.5f}) — model is deblurring ✓")
                else:
                    print(f"  → pred CLOSER to ctx than to GT ({_d_pg - _d_pc:+.5f}) — model is copying input ✗")
                if _d_p_shuf > _d_pg:
                    print(f"  → shuffle test PASS: pred matches its OWN gt ({_d_p_shuf - _d_pg:+.5f}) — supervision real ✓")
                else:
                    print(f"  → shuffle test FAIL: pred matches shuffled gt as well — view-level supervision unclear ✗")
            except Exception as _e_d:
                print(f"[DIAG_PRED_DIST] failed: {_e_d}")
        # ── end DIAG ──
        # Apply dynamic-region mask (gated via env var VAL_USE_MASK=1; default on).
        import os as _os
        _use_mask = _os.environ.get("VAL_USE_MASK", "1") == "1"
        _mask = batch["target"].get("mask", None) if _use_mask else None
        if _mask is not None:
            _m = _mask[0].to(rgb_pred.device, dtype=rgb_pred.dtype)
            rgb_pred_m = rgb_pred * _m
            rgb_gt_m = rgb_gt * _m
        else:
            rgb_pred_m, rgb_gt_m = rgb_pred, rgb_gt
        psnr = compute_psnr(rgb_gt_m, rgb_pred_m).mean()
        lpips = compute_lpips(rgb_gt_m, rgb_pred_m).mean()
        ssim = compute_ssim(rgb_gt_m, rgb_pred_m).mean()
        self.log("val/psnr",  psnr,  on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log("val/lpips", lpips, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log("val/ssim",  ssim,  on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        # Additional center-cropped val metrics (BORDER_MASK_ENABLE=1) for diagnostic
        # comparison against full-image metrics. Eval/papers should still use full.
        from ..loss.border_mask import crop_box as _bm_crop_box
        _H, _W = rgb_pred_m.shape[-2:]
        _t, _b, _l, _r = _bm_crop_box(_H, _W)
        if (_t, _b, _l, _r) != (0, _H, 0, _W):
            _pred_c = rgb_pred_m[..., _t:_b, _l:_r]
            _gt_c = rgb_gt_m[..., _t:_b, _l:_r]
            psnr_c = compute_psnr(_gt_c, _pred_c).mean()
            lpips_c = compute_lpips(_gt_c, _pred_c).mean()
            ssim_c = compute_ssim(_gt_c, _pred_c).mean()
            self.log("val/psnr_crop",  psnr_c,  on_step=False, on_epoch=True, sync_dist=True, logger=True)
            self.log("val/lpips_crop", lpips_c, on_step=False, on_epoch=True, sync_dist=True, logger=True)
            self.log("val/ssim_crop",  ssim_c,  on_step=False, on_epoch=True, sync_dist=True, logger=True)

        # Only log comparison images for the first 4 samples per val run (rank 0 only).
        if batch_idx >= 4 or self.global_rank != 0:
            return

        # Construct comparison image.
        context_img = inverse_normalize(batch["context"]["image"][0])
        context_img_depth = vis_depth_map(gaussian_means)
        max_ctx_vis = context_img.shape[0]  # show all context views
        context = []
        for i in range(min(context_img.shape[0], max_ctx_vis)):
            context.append(context_img[i])
            context.append(context_img_depth[i])
        comparison = hcat(
            add_label(vcat(*context), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
        )

        try:
            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )
        except Exception as e:
            print(f"[WARN] log_image failed: {e}")
        
        if self.train_cfg.feature_rendering_loss > 0:
            context_output = self.decoder.forward(
                gaussians,
                batch["context"]["extrinsics"],
                batch["context"]["intrinsics"],
                batch["context"]["near"],
                batch["context"]["far"],
                (h, w),
                "depth",
            )
            
            gaussian_feature = output.feature
            B,N,C,H,W = gaussian_feature.shape
            
            context_gaussian_feature = context_output.feature
            B,CN,C,H,W = context_gaussian_feature.shape
            gaussian_feature = torch.cat((context_gaussian_feature, gaussian_feature), dim=1)
            
            foundation_feature = self.forward_foundation_model(torch.cat((batch["context"]["image"], batch['target']['image'] * 2 - 1), dim=1))
            context_foundation_features = self.forward_foundation_model(batch["context"]["image"])
            
            
            pca_images, pca_vggt_images = [], []
            for i in range(foundation_feature.shape[1]):
                pca_gaussian_img = run_pca(gaussian_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                pca_images.append(pca_gaussian_img.squeeze(dim=0))
                
                pca_vggt_img = run_pca(foundation_feature[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                pca_vggt_images.append(pca_vggt_img.squeeze(dim=0))
                
            cocnat_pca_images = run_pca(gaussian_feature[0], (H,W))
            cocnat_pca_vggt_images = run_pca(foundation_feature[0], (H,W)) 
            
            pca_context_images = []   
            for i in range(context_foundation_features.shape[1]):
                pca_context_img = run_pca(context_foundation_features[0, i].unsqueeze(dim=0), (H,W))  # (C, H, W)
                pca_context_images.append(pca_context_img.squeeze(dim=0))
                
            context[1] = pca_context_images[0]
            context[3] = pca_context_images[1]
            
            comparison = hcat(
                add_label(vcat(*context), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*pca_vggt_images), "Feature (VFM)"),
                add_label(vcat(*pca_images), "Feature (Prediction)"),
                add_label(vcat(*cocnat_pca_vggt_images), "Feature_cat (VFM)"),
                add_label(vcat(*cocnat_pca_images), "Feature_cat (Prediction)"),
            )
            
            self.logger.log_image(
                f"PCA",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )

        # Render projections and construct projection image.
        try:
            projections = hcat(
                    *render_projections(
                        gaussians,
                        256,
                        extra_label="",
                        low_pass = self.decoder.low_pass_filter,
                    )[0]
                )
            self.logger.log_image(
                "projection",
                [prep_image(add_border(projections))],
                step=self.global_step,
            )
        except Exception as e:
            print(f"[WARN] projection log_image failed: {e}")

        # Draw cameras.
        try:
            cameras = hcat(*render_cameras(batch, 256))
            self.logger.log_image(
                "cameras", [prep_image(add_border(cameras))], step=self.global_step
            )
        except Exception as e:
            print(f"[WARN] cameras log_image failed: {e}")

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        # self.render_video_interpolation(batch)
        # self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._epoch_loss_sum = 0.0
            self._epoch_loss_count = 0
            return
        # val/psnr, val/lpips, val/ssim are logged per-step in validation_step with
        # sync_dist=True; Lightning aggregates and logs them automatically.
        # Only log loss/epoch here (training metric accumulated on rank 0).
        if self._epoch_loss_count > 0 and self.global_rank == 0:
            import wandb as _wandb
            if _wandb.run is not None:
                _wandb.log({"loss/epoch": self._epoch_loss_sum / self._epoch_loss_count},
                           step=self.global_step)
            self._epoch_loss_sum = 0.0
            self._epoch_loss_count = 0

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        if self.encoder.cfg.feature_dim:
            context_feature = self.forward_foundation_model(batch['context']['image'])
        else:
            context_feature = None
        
        gaussians = self.encoder(batch["context"], self.global_step, context_feature = context_feature)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=30)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def on_before_optimizer_step(self, optimizer):
        """v7 belt-and-suspenders: zero out any NaN/Inf grads before the step.
        Catches surprises that slip past gradient clipping (e.g. eigh backward
        on degenerate covariances). Logs once per occurrence so we know if it
        fires often (frequent fires = real problem upstream)."""
        nan_params = 0
        for p in self.parameters():
            if p.grad is not None:
                bad = torch.isnan(p.grad) | torch.isinf(p.grad)
                if bad.any():
                    p.grad[bad] = 0.0
                    nan_params += 1
        if nan_params > 0:
            self.log("debug/nan_grad_params", float(nan_params), on_step=True, prog_bar=True, logger=True)
            print(f"[NAN GUARD] step {self.global_step}: zeroed NaN/Inf grads in {nan_params} params")

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # When freeze_encoder=True, requires_grad is already False for encoder
            # params, so the guard above handles it.  This explicit check is a
            # belt-and-suspenders safeguard in case something re-enables grad.
            if self.train_cfg.freeze_encoder and name.startswith("encoder."):
                continue

            if "gaussian_param_head" in name or "intrinsic_encoder" in name or 'dpt_gs_head' in name or 'gmae' in name or "diffusion_head" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        if warm_up_steps > 0:
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def tensor_mem_mb(t):
    return t.nelement() * t.element_size() / 1024**2
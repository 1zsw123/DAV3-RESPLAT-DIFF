"""
EncoderDA3ReSplat: DA3-GIANT + ReSplat combined encoder for C3G deblur pipeline.

Pipeline:
  1. DA3-GIANT (frozen): blurry images (+ optional GT poses) → depth maps + backbone tokens
  2. ReSplat (frozen or fine-tuned): blurry images → Gaussians, using DA3 depth as depth_init.
  3. Returns: ReSplat Gaussians + DA3 patch tokens in visualization_dump.

use_pred_pose=False (default, Replica/ScanNet++):
  DA3 receives GT extrinsics/intrinsics → pose-conditioned depth → GS render for depth_init.

use_pred_pose=True (RSBlur, no GT poses):
  DA3 predicts its own extrinsics/intrinsics from images alone → predicted depth used directly
  as depth_init. Predicted poses overwrite context["extrinsics"/"intrinsics"] in-place so that
  ReSplat and the downstream renderer both use consistent predicted geometry.
"""

from __future__ import annotations
import sys, types as _types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# ── DA3 path ────────────────────────────────────────────────────────────────────
_DA3_SRC = Path(__file__).resolve().parents[3] / "Depth-Anything-3" / "src"
if str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))
for _k in ("moviepy", "moviepy.editor"):
    if _k not in sys.modules:
        sys.modules[_k] = _types.ModuleType(_k)

# ── ReSplat path ─────────────────────────────────────────────────────────────────
_RESPLAT_SRC = Path(__file__).resolve().parents[3] / "resplat" / "src"
if str(_RESPLAT_SRC) not in sys.path:
    sys.path.insert(0, str(_RESPLAT_SRC))

from ..types import Gaussians
from .encoder import Encoder

_DA3_CKPT_DEFAULT   = "/gpfs/scratch1/shared/qzhang1/da3_pretrained/DA3-GIANT"
_RESPLAT_CKPT_DEFAULT = "/gpfs/scratch1/shared/qzhang1/resplat_pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


@dataclass
class _GaussianAdapterStub:
    sh_degree: int = 3  # matches ReSplat's sh_degree


@dataclass
class EncoderDA3ReSplatCfg:
    name: Literal["da3_resplat"]
    da3_checkpoint: str = _DA3_CKPT_DEFAULT
    resplat_checkpoint: str = _RESPLAT_CKPT_DEFAULT
    freeze_da3: bool = True
    freeze_resplat: bool = True
    pretrained_weights: Optional[str] = None
    gaussian_adapter: _GaussianAdapterStub = field(default_factory=_GaussianAdapterStub)
    use_pred_pose: bool = False  # True → DA3 predicts poses (no GT poses required)


# ── Depth rendering from DA3 Gaussians ──────────────────────────────────────────

def render_da3_depth(da3_gs, w2c_norm: Tensor, K_norm: Tensor,
                     tgt_h: int, tgt_w: int) -> Tensor:
    """
    Render per-view depth maps from DA3 Gaussians for use as ReSplat depth_init.
    Uses DA3's own gs_renderer.

    Returns [B, V, H, W] depth in the normalised world frame scale.
    """
    from depth_anything_3.model.utils.gs_renderer import render_3dgs
    B, V = w2c_norm.shape[:2]
    with torch.no_grad():
        _rgb, depth = render_3dgs(
            extrinsics=w2c_norm.view(B * V, 4, 4),
            intrinsics=K_norm.view(B * V, 3, 3),
            image_shape=(tgt_h, tgt_w),
            gaussian=da3_gs,
            num_view=B * V,
        )
    # depth: [B*V, 1, H, W] or [B*V, H, W] — normalise to [B, V, H, W]
    if depth.dim() == 4:
        depth = depth.squeeze(1)
    return depth.view(B, V, tgt_h, tgt_w)


# ── Encoder class ────────────────────────────────────────────────────────────────

class EncoderDA3ReSplat(Encoder):
    """
    DA3-GIANT + ReSplat combined encoder.

    Step 1: DA3-GIANT (frozen)
        - Encode blurry context images with GT poses
        - Output: pixel-aligned Gaussians + depth maps per view + backbone tokens
    Step 2: ReSplat (frozen by default)
        - Reconstruct Gaussians from blurry images
        - Use DA3 depth as depth_init for point-cloud lifting
        - Output: multi-view consistent Gaussians

    visualization_dump receives DA3 backbone tokens for DiffusionHead:
        blurry_patch_tokens  [B, V, N, C]   (DA3 patch tokens, 4-D)
    (DiffusionHead projects these to its d_tok via a learned linear layer.)
    """

    def __init__(self, cfg: EncoderDA3ReSplatCfg) -> None:
        super().__init__(cfg)

        # ── Load DA3-GIANT ─────────────────────────────────────────────────────
        from depth_anything_3.api import DepthAnything3
        da3_api = DepthAnything3.from_pretrained(cfg.da3_checkpoint)
        da3_api.eval()
        self.da3 = da3_api.model
        if self.da3.gs_head is None:
            raise RuntimeError(f"DA3 checkpoint has no GS head: {cfg.da3_checkpoint}")
        if cfg.freeze_da3:
            for p in self.da3.parameters():
                p.requires_grad_(False)

        # ── Load ReSplat ───────────────────────────────────────────────────────
        import sys as _sys
        from omegaconf import OmegaConf, DictConfig
        _RESPLAT_ROOT = _RESPLAT_SRC.parent
        if str(_RESPLAT_ROOT) not in _sys.path:
            _sys.path.insert(0, str(_RESPLAT_ROOT))
        # C3G's `src` package is already cached in sys.modules; temporarily evict it
        # so Python resolves `from src.model.encoder.encoder_resplat` against resplat/
        # instead of C3G/src. encoder_resplat.py uses only relative imports internally
        # so it is safe to restore C3G's src.* immediately after.
        _c3g_src = {k: _sys.modules.pop(k)
                    for k in list(_sys.modules) if k == 'src' or k.startswith('src.')}
        try:
            from src.model.encoder.encoder_resplat import EncoderReSplat, EncoderReSplatCfg  # noqa: E402
            from dacite import Config as _DaciteCfg, from_dict as _from_dict
        finally:
            # Stash resplat's src.* under resplat_src.* then restore C3G's src.*
            for _k in [k for k in list(_sys.modules) if k == 'src' or k.startswith('src.')]:
                _sys.modules['resplat_src' + _k[3:]] = _sys.modules.pop(_k)
            _sys.modules.update(_c3g_src)

        _rs_cfg_yaml = _RESPLAT_ROOT / "config" / "model" / "encoder" / "resplat.yaml"
        rs_cfg_raw: DictConfig = OmegaConf.load(str(_rs_cfg_yaml))
        # Apply DL3DV-specific overrides required by the pretrained base checkpoint.
        _dl3dv_overrides = OmegaConf.create({
            "shim_patch_size": 16,
            "upsample_factor": 8,
            "lowest_feature_resolution": 8,
        })
        rs_cfg_raw = OmegaConf.merge(rs_cfg_raw, _dl3dv_overrides)
        rs_cfg = _from_dict(
            EncoderReSplatCfg,
            OmegaConf.to_container(rs_cfg_raw, resolve=True),
            config=_DaciteCfg(strict=False),
        )
        self.resplat = EncoderReSplat(rs_cfg)

        # .pth weights have keys prefixed with "encoder."
        ckpt = torch.load(cfg.resplat_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = self.resplat.load_state_dict(enc_state, strict=False)
        assert len(missing) == 0, f"[EncoderDA3ReSplat] Missing encoder keys: {missing}"
        # Only refinement (update_*) keys are expected to be unused — anything else is a bug.
        bad_unexpected = [k for k in unexpected if not k.startswith("update_")]
        assert not bad_unexpected, \
            f"[EncoderDA3ReSplat] Unexpected non-refinement keys: {bad_unexpected[:10]}"
        if cfg.freeze_resplat:
            for p in self.resplat.parameters():
                p.requires_grad_(False)
            self.resplat.eval()

    @torch.no_grad()
    def _normalise_poses(self, w2c: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """DA3-convention: first-view-relative + median translation=1."""
        transform = torch.inverse(w2c[:, 0:1]).expand_as(w2c)
        w2c_n = w2c @ transform
        c2w_n = torch.inverse(w2c_n)
        dists = c2w_n[:, :, :3, 3].norm(dim=-1)
        median_dist = torch.median(dists, dim=1).values.clamp(min=0.1)
        w2c_n = w2c_n.clone()
        w2c_n[:, :, :3, 3] /= median_dist.view(-1, 1, 1)
        return w2c_n, transform, median_dist

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        context_feature=None,
    ) -> Gaussians:
        images = context["image"]   # [B, V, 3, H, W]  in [0,1]
        B, V, _, H, W = images.shape
        device = images.device

        # ImageNet normalise (shared by both paths)
        mean = _IMAGENET_MEAN.to(images)
        std  = _IMAGENET_STD.to(images)
        imgs_da3 = (images - mean) / std

        # DA3 (ViT-G, patch=14) needs H and W divisible by 14.
        # Cap at 336×448 (14-multiples, preserving 3:4 aspect ratio) to avoid OOM
        # at full resolution — DA3 only extracts geometric priors, not fine detail.
        # ReSplat (patch=16) receives the full-resolution context["image"] below.
        _DA3_PATCH = 14
        _DA3_MAX_H, _DA3_MAX_W = 336, 448  # 14×24, 14×32; same 3:4 ratio as 480×640
        H_da3 = min(H, _DA3_MAX_H)
        W_da3 = min(W, _DA3_MAX_W)
        H_da3 = (H_da3 // _DA3_PATCH) * _DA3_PATCH
        W_da3 = (W_da3 // _DA3_PATCH) * _DA3_PATCH
        if H_da3 != H or W_da3 != W:
            imgs_da3 = torch.nn.functional.interpolate(
                imgs_da3.view(B * V, 3, H, W),
                size=(H_da3, W_da3), mode="bilinear", align_corners=False,
            ).view(B, V, 3, H_da3, W_da3)

        # use_pred_pose can be overridden per-call via context["_use_pred_pose"]
        _use_pred = context.pop("_use_pred_pose", self.cfg.use_pred_pose)

        if _use_pred:
            depth_init = self._forward_pred_pose(
                imgs_da3, context, B, V, H, W, device, visualization_dump
            )
        else:
            depth_init = self._forward_gt_pose(
                imgs_da3, context, B, V, H, W, visualization_dump
            )

        # If DA3 was resized to a different resolution, resize depth_init back to (H, W)
        # so ReSplat's UNet skip connections match the context image features.
        if depth_init is not None:
            _, _, dH, dW = depth_init.shape
            if dH != H or dW != W:
                depth_init = torch.nn.functional.interpolate(
                    depth_init, size=(H, W), mode="bilinear", align_corners=False
                )

        # ── Step 2: ReSplat forward ────────────────────────────────────────────
        # ReSplat's UNet requires H and W to be multiples of 32.
        # Floor-crop images to the nearest multiple of 32 before passing in.
        # Gaussians live in 3D world space so the small border crop is harmless.
        _RS_ALIGN = 32
        H_rs = (H // _RS_ALIGN) * _RS_ALIGN
        W_rs = (W // _RS_ALIGN) * _RS_ALIGN
        if H_rs != H or W_rs != W:
            context_rs = {k: v[:, :, :, :H_rs, :W_rs] if (
                isinstance(v, torch.Tensor) and v.dim() == 5 and v.shape[-2] == H and v.shape[-1] == W
            ) else v for k, v in context.items()}
            if depth_init is not None:
                depth_init = depth_init[:, :, :H_rs, :W_rs]
        else:
            context_rs = context

        with torch.no_grad() if self.cfg.freeze_resplat else torch.enable_grad():
            rs_out = self.resplat.forward(
                context_rs,
                global_step=global_step,
                visualization_dump=visualization_dump,
                depth_init=depth_init,
            )

        rs_gs = rs_out["gaussians"] if isinstance(rs_out, dict) else rs_out
        B_g, G = rs_gs.means.shape[:2]
        feat = torch.zeros(B_g, G, 0, device=rs_gs.means.device, dtype=rs_gs.means.dtype)

        # Guard against NaN/Inf in Gaussian parameters.  Large 3D covariances
        # produce huge 2D projections when depth is small → tiles_touched
        # overflows uint32 → prefix sum wraps → num_rendered cast to size_t
        # gives 152 TB allocation.  Typical indoor covariance eigenvalues are
        # 1e-6 to 0.01; clamping to ±0.1 is safe and prevents overflow.
        means = torch.nan_to_num(rs_gs.means, nan=0.0, posinf=1e4, neginf=-1e4)
        covs  = torch.nan_to_num(rs_gs.covariances, nan=0.0, posinf=0.1, neginf=-0.1)
        covs  = covs.clamp(-0.1, 0.1)
        opacs = torch.nan_to_num(rs_gs.opacities, nan=0.0).clamp(0.0, 1.0)
        harmo = torch.nan_to_num(rs_gs.harmonics, nan=0.0)

        return Gaussians(
            means=means,
            covariances=covs,
            harmonics=harmo,
            opacities=opacs,
            feature=feat,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward_gt_pose(
        self, imgs_da3, context, B, V, H, W, visualization_dump
    ):
        """DA3 forward with GT poses → GS-rendered depth_init."""
        c2w       = context["extrinsics"]  # [B, V, 4, 4] C2W
        intr_norm = context["intrinsics"]  # [B, V, 3, 3] normalised

        w2c = torch.inverse(c2w.view(B * V, 4, 4)).view(B, V, 4, 4)
        w2c_n, _transform, _median_dist = self._normalise_poses(w2c)

        # K_px must match the resolution DA3 actually sees (H_da3, W_da3),
        # not the original (H, W) — DA3 is hard-capped at 336x448 and the image
        # is bilinearly resized to that. Using the wrong scale here breaks the
        # camera intrinsics by up to 2.7x at full Replica resolution.
        H_da3, W_da3 = imgs_da3.shape[-2], imgs_da3.shape[-1]
        K_px = intr_norm.clone()
        K_px[..., 0, 0] *= W_da3;  K_px[..., 0, 2] *= W_da3
        K_px[..., 1, 1] *= H_da3;  K_px[..., 1, 2] *= H_da3

        # ── CHECKPOINTS to prove GT data flows end-to-end ──
        # Print one-time on first call so every train + val sees these in log.
        if not getattr(self, "_gt_path_logged", False):
            print(f"[CHECK_GT_PATH] STAGE 1 (context @ encoder entry):")
            print(f"  c2w[0,0] (GT from dataset):\n{c2w[0,0].detach().cpu().numpy()}")
            print(f"  intr_norm[0,0] (GT normalised K): "
                  f"fx={intr_norm[0,0,0,0].item():.4f} fy={intr_norm[0,0,1,1].item():.4f} "
                  f"cx={intr_norm[0,0,0,2].item():.4f} cy={intr_norm[0,0,1,2].item():.4f}")
            print(f"[CHECK_GT_PATH] STAGE 2 (after inverse → w2c):")
            print(f"  w2c[0,0]:\n{w2c[0,0].detach().cpu().numpy()}")
            print(f"  w2c[0,0] @ c2w[0,0] should = identity (sanity):")
            _id_check = (w2c[0,0] @ c2w[0,0]).detach().cpu().numpy()
            print(f"  {_id_check}")
            print(f"[CHECK_GT_PATH] STAGE 3 (after _normalise_poses):")
            print(f"  w2c_n[0,0] (frame-0-rel, scale-norm):\n{w2c_n[0,0].detach().cpu().numpy()}")
            print(f"  median_dist[0]={_median_dist[0].item():.4f}  (=1.0 means clamp engaged or trivial)")
            print(f"[CHECK_GT_PATH] STAGE 4 (K_px at DA3 input resolution {H_da3}x{W_da3}):")
            print(f"  K_px[0,0]: fx={K_px[0,0,0,0].item():.2f} fy={K_px[0,0,1,1].item():.2f} "
                  f"cx={K_px[0,0,0,2].item():.2f} cy={K_px[0,0,1,2].item():.2f}")
            print(f"[CHECK_GT_PATH] STAGE 5 (DA3 forward input — GT-w2c + GT-K, ZERO DA3 pred):")
            print(f"  da3.forward(extrinsics=w2c_n[GT], intrinsics=K_px[GT], ...)")
            print(f"[CHECK_GT_PATH] use_pred_pose=False → _forward_gt_pose path → NO write-back to context['extrinsics' or 'intrinsics'] → ReSplat / decoder read GT directly.")
            self._gt_path_logged = True

        with torch.no_grad():
            da3_out = self.da3.forward(
                imgs_da3,
                extrinsics=w2c_n,
                intrinsics=K_px,
                infer_gs=True,
                return_tokens=True,
            )

        if visualization_dump is not None:
            if hasattr(da3_out, "patch_tokens") and da3_out.patch_tokens is not None:
                visualization_dump["blurry_patch_tokens"] = da3_out.patch_tokens

        depth_init = None
        if da3_out.gaussians is not None:
            try:
                depth_norm = render_da3_depth(
                    da3_out.gaussians, w2c_n, intr_norm.view(B, V, 3, 3), H, W
                )
                depth_init = depth_norm * _median_dist.view(B, 1, 1, 1)
            except Exception as e:
                print(f"[EncoderDA3ReSplat] depth render failed: {e}, using ReSplat depth")

        del da3_out
        torch.cuda.empty_cache()
        return depth_init

    def _forward_pred_pose(
        self, imgs_da3, context, B, V, H, W, device, visualization_dump
    ):
        """DA3 forward without GT poses → predicted poses + direct depth_init.

        DA3 output conventions:
          da3_out.extrinsics  [B, V, 3, 4]  W2C  (affine_inverse of predicted C2W)
          da3_out.intrinsics  [B, V, 3, 3]  pixel-scale K
          da3_out.depth       [B*V, H, W] or [B, V, H, W]  (DA3-internal scale)

        We convert to C3G conventions and overwrite context in-place so that
        both ReSplat and the renderer downstream use the predicted geometry.

        Stability addition (2026-05-02): **scene-scale lift**. DA3
        internally normalises poses to median-trans = 1 (unit scale).
        ReSplat was pretrained on dataset-scale (meter) inputs, so feeding
        it unit-scale depth puts it well out-of-distribution and risks
        rasterizer blowup at Replica's near=0.01. We multiply predicted
        translations and depth by `scene_scale = (near + far) / 2` (≈ 5 m
        for replica), placing them in the meter-scale range ReSplat expects.

        Plus defensive NaN/Inf scrubbing and value clamps for the rare
        cases where DA3 emits absurd outputs.
        """
        from depth_anything_3.utils.geometry import as_homogeneous

        with torch.no_grad():
            da3_out = self.da3.forward(
                imgs_da3,
                extrinsics=None,
                intrinsics=None,
                infer_gs=False,
                return_tokens=True,
            )

        if visualization_dump is not None:
            if hasattr(da3_out, "patch_tokens") and da3_out.patch_tokens is not None:
                visualization_dump["blurry_patch_tokens"] = da3_out.patch_tokens

        # ── Predicted poses ─────────────────────────────────────────────────
        w2c_pred_34 = da3_out.extrinsics                          # [B, V, 3, 4]
        bottom = torch.tensor(
            [0., 0., 0., 1.], device=device
        ).view(1, 1, 1, 4).expand(B, V, 1, 4)
        w2c_pred = torch.cat([w2c_pred_34, bottom], dim=2)        # [B, V, 4, 4] W2C

        # Guard against degenerate DA3 outputs (NaN/Inf, runaway translations).
        w2c_pred = torch.nan_to_num(w2c_pred, nan=0.0, posinf=1e3, neginf=-1e3)
        w2c_pred[..., :3, 3] = w2c_pred[..., :3, 3].clamp(min=-1e3, max=1e3)
        c2w_pred = torch.inverse(w2c_pred.view(B * V, 4, 4)).view(B, V, 4, 4)
        c2w_pred = torch.nan_to_num(c2w_pred, nan=0.0, posinf=1e3, neginf=-1e3)

        # ── (2) Scene-scale lift: bring unit-scale predictions to meter scale
        # so ReSplat sees inputs in its training distribution. Use the
        # batch's near/far if available; default to 5.0 (replica typical).
        ctx_near = context.get("near", None)
        ctx_far  = context.get("far",  None)
        if ctx_near is not None and ctx_far is not None:
            scene_scale = ((ctx_near.float().mean() + ctx_far.float().mean()) * 0.5).item()
        else:
            scene_scale = 5.0
        scene_scale = max(0.1, min(scene_scale, 100.0))   # safety clamp

        c2w_pred[..., :3, 3] = c2w_pred[..., :3, 3] * scene_scale

        # Predicted intrinsics: pixel-scale → normalised, then guarded.
        # CRITICAL: DA3 outputs K in pixel-space at the DA3 INPUT resolution
        # (H_da3, W_da3), not the dataset (H, W). They differ when DA3's 336×448
        # cap clipped/resized the input. Normalising by (H, W) is silently
        # correct only when H_da3 == H and W_da3 == W; for any larger dataset
        # resolution it breaks K. Use the actual DA3 input dims here.
        H_da3, W_da3 = imgs_da3.shape[-2], imgs_da3.shape[-1]
        K_px_pred  = torch.nan_to_num(
            da3_out.intrinsics.clone(), nan=0.0, posinf=1e4, neginf=0.0
        )                                                          # [B, V, 3, 3] @ DA3 res
        K_norm_pred = K_px_pred.clone()
        K_norm_pred[..., 0, 0] /= W_da3;  K_norm_pred[..., 0, 2] /= W_da3
        K_norm_pred[..., 1, 1] /= H_da3;  K_norm_pred[..., 1, 2] /= H_da3
        K_norm_pred[..., 0, 0].clamp_(min=0.1, max=10.0)
        K_norm_pred[..., 1, 1].clamp_(min=0.1, max=10.0)
        K_norm_pred[..., 0, 2].clamp_(min=0.0, max=1.0)
        K_norm_pred[..., 1, 2].clamp_(min=0.0, max=1.0)
        # DIAG: dump DA3-raw pixel intrinsics + image dims to verify normalization correctness
        import os as _os_dbg
        if _os_dbg.environ.get('DUMP_DA3_INTR', '0') == '1':
            print(f"[DA3-RAW] image (H,W) used by DA3 = ({imgs_da3.shape[-2]}, {imgs_da3.shape[-1]})")
            print(f"[DA3-RAW] dataset image (H,W) = ({H}, {W})")
            print(f"[DA3-RAW] K_pixel (DA3 raw): fx_px={K_px_pred[..., 0, 0].mean().item():.2f} "
                  f"fy_px={K_px_pred[..., 1, 1].mean().item():.2f} "
                  f"cx_px={K_px_pred[..., 0, 2].mean().item():.2f} "
                  f"cy_px={K_px_pred[..., 1, 2].mean().item():.2f}")
            print(f"[DA3-RAW] K_normalized (after /W /H): fx={K_norm_pred[..., 0, 0].mean().item():.4f} "
                  f"fy={K_norm_pred[..., 1, 1].mean().item():.4f}")
            # If fx_px != fy_px significantly → DA3 thinks pixels are non-square, suspicious
            _ratio = (K_px_pred[..., 0, 0] / K_px_pred[..., 1, 1].clamp(min=1e-6)).mean().item()
            print(f"[DA3-RAW] fx_px/fy_px ratio = {_ratio:.4f}  (=1.0 for square pixels)")

        # Diagnostic override (CTX_POSE_OVERRIDE env var):
        #   'identity' → replace all ctx poses with identity (all views = same camera).
        #   'first'    → broadcast first ctx pose to all ctx views (collapses pose diversity).
        # Used to test whether DA3's noisy pose predictions on near-static RSBlur
        # bursts (where ground truth pose diff ≈ 0) are the root cause of val drift.
        import os as _os_diag
        _pose_override = _os_diag.environ.get('CTX_POSE_OVERRIDE', '')
        if _pose_override == 'identity':
            c2w_pred = torch.eye(4, device=c2w_pred.device, dtype=c2w_pred.dtype).expand(B, V, 4, 4).contiguous()
        elif _pose_override == 'first':
            c2w_pred = c2w_pred[:, 0:1].expand_as(c2w_pred).contiguous()
        # Fix 2: pose smoothing — average XYZ translation across burst (keep
        # rotation per-view since DA3 rotations may capture real micro-jitter).
        # RSBlur rig is physically stationary → ctx translations should be
        # near-constant (handheld jitter at most). DA3 predicts x_range=14,
        # y=0.87, z=2.14 unit across burst — catastrophic.
        # SMOOTH_CTX_TRANSLATION=1 forces mean translation; this is a PHYSICAL
        # PRIOR for static-rig data, not a hack.
        if _os_diag.environ.get('SMOOTH_CTX_TRANSLATION', _os_diag.environ.get('SMOOTH_CTX_POSE', '0')) == '1':
            mean_t = c2w_pred[..., :3, 3].mean(dim=1, keepdim=True)  # [B, 1, 3]
            c2w_pred = c2w_pred.clone()
            c2w_pred[..., :3, 3] = mean_t.expand(-1, V, -1)
        # Smooth rotation: force all ctx to use FIRST view's rotation. RSBlur
        # rig has near-fixed orientation across burst; DA3 predicts per-view
        # pitch (forward_y up to 0.038 rad ≈ 2°) which renders less ground
        # → "bottom cropped" appearance. SMOOTH_CTX_ROTATION=1 broadcasts the
        # first-view rotation to all views, eliminating per-view pitch jitter.
        if _os_diag.environ.get('SMOOTH_CTX_ROTATION', '0') == '1':
            first_R = c2w_pred[:, 0:1, :3, :3]                # [B, 1, 3, 3]
            c2w_pred = c2w_pred.clone()
            c2w_pred[:, :, :3, :3] = first_R.expand(-1, V, -1, -1)
        # Optional GT injection: keep pred-pose forward path (preserves GoPro
        # ckpt's training distribution) but replace pred poses/intrinsics with
        # GT before downstream sees them. context still holds dataset-provided
        # GT here (encoder hasn't read context["extrinsics"] yet in pred-pose).
        # Configured via env vars (no schema change):
        #   INJECT_GT_INTRINSICS=1  → use ground-truth K from dataset
        #   INJECT_GT_EXTRINSICS=1  → align GT c2w to pred frame-0 + scene_scale
        import os as _os_inj
        _inj_K = _os_inj.environ.get('INJECT_GT_INTRINSICS', '0') == '1'
        _inj_E = _os_inj.environ.get('INJECT_GT_EXTRINSICS', '0') == '1'

        if _inj_K and "intrinsics" in context:
            gt_K = context["intrinsics"]
            # gt_K shape may be [B, V, 3, 3] (normalised) or pixel; assume normalised.
            if gt_K.shape == K_norm_pred.shape:
                # One-time verification print so user can confirm in log that
                # GT intrinsics actually replaced DA3-predicted ones.
                if not getattr(self, "_inj_K_logged", False):
                    pred_sample = K_norm_pred[0, 0].detach().cpu().tolist()
                    gt_sample   = gt_K[0, 0].detach().cpu().tolist()
                    print(f"[INJECT_GT_INTRINSICS] verified active.")
                    print(f"[INJECT_GT_INTRINSICS] DA3 pred K (view 0): "
                          f"fx={pred_sample[0][0]:.4f} fy={pred_sample[1][1]:.4f} "
                          f"cx={pred_sample[0][2]:.4f} cy={pred_sample[1][2]:.4f}")
                    print(f"[INJECT_GT_INTRINSICS] GT K (view 0):       "
                          f"fx={gt_sample[0][0]:.4f} fy={gt_sample[1][1]:.4f} "
                          f"cx={gt_sample[0][2]:.4f} cy={gt_sample[1][2]:.4f}")
                    self._inj_K_logged = True
                K_norm_pred = gt_K.to(K_norm_pred.dtype)

        # ── Ray-based R re-derivation per the paper formulation ────────────
        # H = K · R is DA3's "homography" found via DLT from its ray map.
        # We have R_pred and K_pred from DA3, and inject GT K above. With K
        # fixed to GT, recompute R from the SAME ray-direction homography:
        #     R_new = SVD-orthogonalize( K_gt^{-1} · K_pred · R_pred )
        # so the injected pose stays consistent with DA3's internal ray map
        # (i.e. its Gaussians) while honouring GT intrinsics.
        if _inj_E and _inj_K and "intrinsics" in context:
            R_pred = w2c_pred[..., :3, :3]                      # [B, V, 3, 3]
            t_pred = w2c_pred[..., :3, 3:4]                     # [B, V, 3, 1]

            # K_pred from DA3: pixel-form at DA3 INPUT resolution (H_da3, W_da3),
            # NOT dataset (H, W). They differ when DA3 caps the input.
            K_pred_px = K_px_pred.to(R_pred.dtype)              # [B, V, 3, 3] pixel @ DA3 res
            # GT K from context is normalised; must be converted to pixel using
            # the SAME (H_da3, W_da3) so K_gt and K_pred share a coord system.
            # Otherwise K_gt^{-1} K_pred mixes pixel scales → wrong R_raw.
            H_da3, W_da3 = imgs_da3.shape[-2], imgs_da3.shape[-1]
            K_gt_norm = context["intrinsics"].to(R_pred.dtype)  # [B, V, 3, 3] normalised
            K_gt_px = K_gt_norm.clone()
            K_gt_px[..., 0, 0] = K_gt_px[..., 0, 0] * W_da3
            K_gt_px[..., 0, 2] = K_gt_px[..., 0, 2] * W_da3
            K_gt_px[..., 1, 1] = K_gt_px[..., 1, 1] * H_da3
            K_gt_px[..., 1, 2] = K_gt_px[..., 1, 2] * H_da3

            # H = K_pred · R_pred  (the ray-direction homography DA3 produced)
            H_pred = K_pred_px @ R_pred                         # [B, V, 3, 3]

            # R_raw = K_gt^{-1} · H_pred
            R_raw = torch.linalg.solve(K_gt_px, H_pred)

            # Project to nearest orthogonal matrix via SVD (Procrustes).
            # Force det = +1 (no reflection).
            U, S_sv, Vh = torch.linalg.svd(R_raw)
            R_new = U @ Vh
            det = torch.linalg.det(R_new)                       # [B, V]
            sign = torch.sign(det).clamp(min=-1.0, max=1.0)
            sign_fix = torch.ones_like(R_new[..., 0])
            sign_fix[..., -1] = sign
            R_new = U @ torch.diag_embed(sign_fix) @ Vh

            # Build w2c_new with new rotation, keeping DA3's original t_pred
            # (= mean ray origins in DA3's convention).
            w2c_new = torch.eye(4, device=R_new.device, dtype=R_new.dtype).expand(B, V, 4, 4).contiguous().clone()
            w2c_new[..., :3, :3] = R_new
            w2c_new[..., :3, 3:4] = t_pred

            # Convert to c2w and apply scene_scale (mirroring the original lift)
            c2w_new = torch.inverse(w2c_new.view(B * V, 4, 4)).view(B, V, 4, 4)
            c2w_new[..., :3, 3] = c2w_new[..., :3, 3] * scene_scale

            # One-time verification print
            if not getattr(self, "_inj_ray_logged", False):
                R_diff = R_pred[0, 0] @ R_new[0, 0].transpose(-1, -2)
                tr = torch.diagonal(R_diff, dim1=-1, dim2=-2).sum(-1).clamp(-1.0, 3.0)
                ang_deg = torch.acos(((tr - 1) / 2).clamp(-1.0, 1.0)) * 180.0 / 3.14159265
                print(f"[INJECT_RAY_BASED] verified active.")
                print(f"[INJECT_RAY_BASED] K_pred (px): "
                      f"fx={K_pred_px[0,0,0,0].item():.2f} fy={K_pred_px[0,0,1,1].item():.2f} "
                      f"cx={K_pred_px[0,0,0,2].item():.2f} cy={K_pred_px[0,0,1,2].item():.2f}")
                print(f"[INJECT_RAY_BASED] K_gt   (px): "
                      f"fx={K_gt_px[0,0,0,0].item():.2f} fy={K_gt_px[0,0,1,1].item():.2f} "
                      f"cx={K_gt_px[0,0,0,2].item():.2f} cy={K_gt_px[0,0,1,2].item():.2f}")
                print(f"[INJECT_RAY_BASED] R_raw det range over views: "
                      f"min={det.min().item():.4f} max={det.max().item():.4f}")
                print(f"[INJECT_RAY_BASED] singular values of R_raw (view 0,0): {S_sv[0,0].tolist()}")
                print(f"[INJECT_RAY_BASED] R_pred vs R_new angular diff (view 0,0): {ang_deg.item():.4f} deg")
                self._inj_ray_logged = True

            c2w_pred = c2w_new
        elif _inj_E and "extrinsics" in context:
            # Fallback: SE(3) substitution if INJECT_GT_INTRINSICS not set
            gt_c2w = context["extrinsics"].to(c2w_pred.dtype)
            gt_inv0 = torch.inverse(gt_c2w[:, 0:1])
            gt_rel = gt_inv0 @ gt_c2w
            pred_t_mag = c2w_pred[:, :, :3, 3].norm(dim=-1)
            gt_t_mag   = gt_rel[:,   :, :3, 3].norm(dim=-1)
            pred_med = torch.median(pred_t_mag[:, 1:], dim=1).values.clamp(min=0.01)
            gt_med   = torch.median(gt_t_mag[:, 1:],   dim=1).values.clamp(min=0.01)
            ratio = (pred_med / gt_med).clamp(max=100.0).view(-1, 1, 1)
            gt_rel = gt_rel.clone()
            gt_rel[:, :, :3, 3] = gt_rel[:, :, :3, 3] * ratio
            c2w_pred = gt_rel

        # Overwrite context in-place with (possibly overridden) predicted poses.
        context["extrinsics"] = c2w_pred
        context["intrinsics"] = K_norm_pred

        # ── Depth, lifted to scene scale ────────────────────────────────────
        depth = da3_out.depth
        if depth is not None:
            if depth.dim() == 3:
                depth = depth.view(B, V, H, W)
            depth = torch.nan_to_num(depth, nan=0.0, posinf=10.0, neginf=0.0)
            depth = depth * scene_scale
            depth = depth.clamp(min=0.0, max=ctx_far.float().max().item() * 2.0
                                if ctx_far is not None else 50.0)
            depth_init = depth.detach()
        else:
            depth_init = None

        del da3_out
        torch.cuda.empty_cache()
        return depth_init

    def get_data_shim(self):
        return lambda batch: batch

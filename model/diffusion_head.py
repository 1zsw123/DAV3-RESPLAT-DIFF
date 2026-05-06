"""
DiffusionHead: flow-matching-based Gaussian deblurrer.

Adapts Nova3R's PointJointFMDecoderV2 and TransformerEncoder to operate on
14-dimensional Gaussian attribute vectors
(means 3 + log_scales 3 + quaternions 4 + harmonics_flat 3*d_sh + logit_opa 1).

Pipeline:
  1. Flatten blurry Gaussian attrs → x_0  [B, G, D_gauss]
  2. Encode  x_0 → cond_tokens            [B, K, d_tok]
  3. Euler ODE (num_steps steps):
       velocity = FMDenoiser(x_t, t, cond_tokens)   [B, G, D_gauss]
       x_{t+dt} = x_t + dt * velocity
  4. Convert x_1 back → Gaussians

Training: single Euler step (num_steps=1), supervised by photometric rendering loss.
Inference: multi-step for better quality.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .types import Gaussians
from .mip_splatting_refiner_utils import _cov_to_log_scales_quat, _scales_quat_to_cov, _matrix_to_quat, _quat_to_matrix

# ── Add nova3r to path (idempotent) ──────────────────────────────────────────
_NOVA3R_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "nova3r")
)
for _p in (_NOVA3R_ROOT, os.path.join(_NOVA3R_ROOT, "nova3r")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nova3r.heads.pts3d_encoder.transformer_encoder import TransformerEncoder
from nova3r.heads.pts3d_decoder.flowm_decoder_point_joint_v2 import PointJointFMDecoderV2

# ── DAV3 pose_enc utility (ProPE-style: T+quat+fov_h+fov_w = 9-dim) ─────────
_DAV3_TRANSFORM = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "Depth-Anything-3", "src", "depth_anything_3", "model", "utils", "transform.py")
)
try:
    _spec = importlib.util.spec_from_file_location("dav3_transform", _DAV3_TRANSFORM)
    _dav3 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_dav3)
    _extri_intri_to_pose_enc = _dav3.extri_intri_to_pose_encoding
    _HAS_POSE_ENC = True
except Exception:
    _HAS_POSE_ENC = False


# ─────────────────────────────── Config ──────────────────────────────────────

@dataclass
class DiffusionHeadCfg:
    # ---- Gaussian encoder (TransformerEncoder) ----
    enc_k: int = 256            # number of shape tokens (K)
    enc_df: int = 128           # inner dim of shape tokens
    enc_df_out: int = 128       # encoder output dim  (= decoder dim_in)
    enc_d_point: int = 64       # point embedding dim
    enc_cross_depth: int = 4    # cross-attention layers in encoder
    enc_self_depth: int = 2     # self-attention layers in encoder

    # ---- FM denoiser (PointJointFMDecoderV2) ----
    dec_dim_model: int = 64     # internal attention dim of denoiser
    dec_cross_depth: int = 3    # cross/self-attention depth in denoiser
    dec_num_virtual: int = 256  # virtual tracks in denoiser

    # ---- Training/inference ----
    train_num_steps: int = 1    # Euler steps during training (1 = fast)
    infer_num_steps: int = 10   # Euler steps during inference

    # ---- Camera conditioning ----
    use_cam_cond: bool = True     # inject GT camera pose_enc as conditioning tokens (ProPE 9-dim)
    use_vggt_priors: bool = True  # inject VGGT camera_token + patch_token as extra conditioning

    # ---- Design A: camera prediction output ----
    use_cam_pred: bool = False  # predict per-view pose_enc [B,V,9] and supervise with GT


# ────────────────────────────── Camera Encoder ───────────────────────────────

class CamEncoder(nn.Module):
    """
    Encode camera parameters → conditioning token per view.

    Uses ProPE-style 9-dim pose_enc: T(3) + quat(4) + fov_h(1) + fov_w(1).
    This is non-Euclidean-aware: rotation as unit quaternion, FoV scale-invariant.
    T may still be scale-dependent if baseline operates in random scale —
    for prototype experiments this is acceptable; normalize in future work.

    Accepts pre-computed pose_enc [B, V, 9] (preferred) OR raw extrinsics+intrinsics
    with image size for on-the-fly conversion.
    """
    _D_POSE: int = 9  # T(3)+quat(4)+fov_h+fov_w

    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(self._D_POSE, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

    def forward(
        self,
        pose_enc: Float[Tensor, "batch view 9"],
    ) -> Float[Tensor, "batch view d"]:
        return self.proj(pose_enc)   # [B, V, d_out]


# ──────────────────────────── Gaussian ↔ attrs ───────────────────────────────

def gaussians_to_attrs(gaussians: Gaussians) -> Float[Tensor, "batch gaussian d"]:
    """
    Pack all Gaussian parameters into a flat [B, G, D_gauss] tensor.

    Layout:  [means(3) | log_scales(3) | quaternions(4) | logit_opa(1) | dc_color(3) | higher_sh_flat(3*(d_sh-1))]
    Geometry section (indices 0:14): means + log_scales + quats + logit_opa + dc_color(RGB).
      DC color included so DiffHead can correct blurry appearance (SH degree-0 term).
    Higher SH section (indices 14:): pass-through, never touched by ODE.
    """
    B, G = gaussians.means.shape[:2]

    # Covariance → (log_scales, quaternions)  in batch mode
    eigenvalues, Q = torch.linalg.eigh(gaussians.covariances)  # [B,G,3], [B,G,3,3]
    eigenvalues = eigenvalues.clamp(min=1e-10)
    log_scales = eigenvalues.sqrt().log().clamp(-10, 10)        # [B,G,3]

    # Ensure Q is a proper rotation (det = +1)
    dets = torch.linalg.det(Q)                                  # [B,G]
    Q = Q.clone()
    Q[:, :, :, 0] = Q[:, :, :, 0] * dets.sign().unsqueeze(-1)  # flip col-0 if det<0

    quats = _matrix_to_quat(Q.reshape(B * G, 3, 3)).reshape(B, G, 4)  # [B,G,4]

    # Opacities [B,G] → logit [B,G,1]  (index 10)
    opc = gaussians.opacities.clamp(1e-6, 1.0 - 1e-6)
    logit_opa = torch.log(opc / (1.0 - opc)).unsqueeze(-1)      # [B,G,1]

    d_sh = gaussians.harmonics.shape[-1]
    if d_sh > 1:
        # DC color = harmonics[:,:,:,0] [B,G,3]  (geometry section, index 11:14)
        dc_color    = gaussians.harmonics[:, :, :, 0]              # [B,G,3]
        # Higher SH (degree ≥ 1) pass-through, index 14:
        higher_flat = gaussians.harmonics[:, :, :, 1:].reshape(B, G, -1)  # [B,G,3*(d_sh-1)]
        return torch.cat([gaussians.means, log_scales, quats, logit_opa, dc_color, higher_flat], dim=-1)
    else:
        # d_sh==1: all harmonics are DC, all in geometry section
        dc_color = gaussians.harmonics[:, :, :, 0]                # [B,G,3]
        return torch.cat([gaussians.means, log_scales, quats, logit_opa, dc_color], dim=-1)


def attrs_to_gaussians(
    attrs: Float[Tensor, "batch gaussian d"],
    template: Gaussians,
) -> Gaussians:
    """
    Unpack a [B, G, D_gauss] tensor back to Gaussians, using *template* for
    shapes (num SH channels, feature tensor).

    Layout:  [means(3) | log_scales(3) | quats(4) | logit_opa(1) | dc_color(3) | higher_sh_flat(3*(d_sh-1))]
    Geometry section (0:14): geometry + DC color.  Higher SH (14:): pass-through.
    """
    B, G = attrs.shape[:2]
    n_colors = template.harmonics.shape[-2]   # 3
    d_sh     = template.harmonics.shape[-1]   # (sh_degree+1)^2

    # Split geometry section
    means      = attrs[:, :, 0:3]
    log_scales = attrs[:, :, 3:6]
    quats      = attrs[:, :, 6:10]
    logit_opa  = attrs[:, :, 10]               # [B,G]
    dc_color   = attrs[:, :, 11:14]            # [B,G,3]  DC color (now part of geometry ODE)

    if d_sh > 1:
        higher_flat = attrs[:, :, 14:14 + n_colors * (d_sh - 1)]  # [B,G,3*(d_sh-1)]
        higher_sh   = higher_flat.reshape(B, G, n_colors, d_sh - 1)
        harmonics   = torch.cat([dc_color.unsqueeze(-1), higher_sh], dim=-1)  # [B,G,3,d_sh]
    else:
        harmonics = dc_color.unsqueeze(-1)     # [B,G,3,1] — only DC

    # (log_scales, quats) → covariance
    scales  = torch.exp(log_scales.clamp(-10, 10))   # [B,G,3] clamp prevents extreme scales
    q_norm  = F.normalize(quats, p=2, dim=-1)  # [B,G,4]
    # Guard against zero quaternions (shouldn't happen after normalize, but safety)
    q_norm  = torch.nan_to_num(q_norm, nan=0.0)
    q_norm[:, :, 0] = q_norm[:, :, 0] + (q_norm.norm(dim=-1) < 1e-6).float()  # fallback to identity
    q_norm  = F.normalize(q_norm, p=2, dim=-1)
    R       = _quat_to_matrix(q_norm.reshape(B * G, 4)).reshape(B, G, 3, 3)
    S2      = torch.diag_embed(scales.reshape(B * G, 3) ** 2).reshape(B, G, 3, 3)
    cov     = R @ S2 @ R.transpose(-1, -2)    # [B,G,3,3]
    cov     = torch.nan_to_num(cov, nan=1e-6) # safety

    opacities = torch.sigmoid(logit_opa)       # [B,G]

    return Gaussians(
        means=means,
        covariances=cov,
        harmonics=harmonics,
        opacities=opacities,
        feature=template.feature,              # pass-through unchanged
    )


# ─────────────────────────────── Module ──────────────────────────────────────

class DiffusionHead(nn.Module):
    """
    Flow-matching Gaussian deblurrer.

    Given blurry Gaussians (from the Mip-Splatting refiner), runs a learned
    velocity field to produce "sharp" Gaussians that render photometrically
    closer to sharp GT images.

    Training: one Euler step, supervised entirely by the downstream photometric
    loss (no per-Gaussian GT needed).
    Inference: multi-step Euler ODE for higher quality.
    """

    def __init__(self, cfg: DiffusionHeadCfg, d_gauss: int) -> None:
        """
        Args:
            cfg:     hyperparameters
            d_gauss: Gaussian attribute vector dimension (auto-computed from
                     the first batch; pass 0 to defer, and call init_dims later)
        """
        super().__init__()
        self.cfg     = cfg
        self.d_gauss = d_gauss

        if d_gauss > 0:
            self._build(d_gauss)

    def _build(self, d_gauss: int) -> None:
        """Construct sub-modules once d_gauss is known."""
        cfg = self.cfg

        # ── Gaussian attribute encoder ─────────────────────────────────────
        self.gaussian_encoder = TransformerEncoder(
            input_dim   = d_gauss,
            k           = cfg.enc_k,
            df          = cfg.enc_df,
            df_out      = cfg.enc_df_out,
            d_point     = cfg.enc_d_point,
            cross_depth = cfg.enc_cross_depth,
            self_depth  = cfg.enc_self_depth,
        )

        # ── Camera conditioning encoder (ProPE 9-dim pose_enc → d_tok) ───
        if cfg.use_cam_cond:
            self.cam_encoder = CamEncoder(d_out=cfg.enc_df_out)

        # ── VGGT 2D prior projector (DA3 ViT-G patch tokens, dim=3072) ─────
        # vggt_cam_projector removed: encoder never dumps blurry_camera_tokens.
        if cfg.use_vggt_priors:
            _D_VGGT = 3072  # DA3-GIANT ViT-G hidden dim, confirmed via config.json
            self.vggt_patch_projector = nn.Linear(_D_VGGT, cfg.enc_df_out)

        # ── Design A: camera prediction head ──────────────────────────────
        # Predicts 9-dim pose_enc (T3+quat4+fovh+fovw) per view from
        # shape context + per-view cam tokens.
        if cfg.use_cam_pred:
            self.camera_pred_head = nn.Sequential(
                nn.Linear(cfg.enc_df_out * 2, cfg.enc_df_out),
                nn.GELU(),
                nn.Linear(cfg.enc_df_out, 9),
            )

        # ── Flow-matching velocity predictor ──────────────────────────────
        # ODE operates on: means(3)+log_scales(3)+quats(4)+logit_opa(1)+dc_color(3)=14.
        # query_dim / output_dim are always 14 regardless of full d_gauss so that
        # x_geom (sliced to 14 dims in forward) matches the denoiser's expected input.
        _D_GEOM = 14
        self.fm_denoiser = PointJointFMDecoderV2(
            has_conf          = False,
            dim_in            = cfg.enc_df_out,   # must match encoder output
            output_dim        = _D_GEOM,
            query_dim         = _D_GEOM,
            cross_depth       = cfg.dec_cross_depth,
            self_depth        = cfg.dec_cross_depth,   # must be equal (assert inside)
            dim_model         = cfg.dec_dim_model,
            num_virtual_tracks= cfg.dec_num_virtual,
            use_sdpa          = True,
            use_num_view_cond = False,
        )
        # PointJointFMDecoderV2.virual_tracks is defined in __init__ but the line
        # that uses it in forward() is commented out → dead parameter → DDP hangs
        # waiting for a gradient that never arrives.  Disable grad so DDP ignores it.
        if hasattr(self.fm_denoiser, "virual_tracks"):
            self.fm_denoiser.virual_tracks.requires_grad_(False)

        self.d_gauss = d_gauss

        # ── Zero-init output projections → velocity≈0 at training start ──────
        self._zero_init_output_projections(_D_GEOM)

    def _zero_init_output_projections(self, d_geom: int) -> None:
        """Zero-initialize any Linear layer whose output dim == d_geom."""
        for module in self.fm_denoiser.modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] == d_geom:
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        gaussians_blurry: Gaussians,
        num_steps: Optional[int] = None,
        context_pose_enc: Optional[Float[Tensor, "batch view 9"]] = None,
        blurry_patch_tokens:  Optional[Float[Tensor, "batch view num_patches d_da3"]] = None,
    ):
        """
        Args:
            gaussians_blurry:    [B, G, ...] from C3G encoder (blurry)
            num_steps:           Euler steps; None → train/infer cfg
            context_pose_enc:    [B, V, 9] ProPE pose_enc (T+quat+fovh+fovw), pre-computed
            blurry_patch_tokens: [B, V, N, C] DA3 patch tokens per view (from encoder dump)
        Returns:
            Design B: gaussians_sharp  (Gaussians)
            Design A: (gaussians_sharp, pred_pose_enc [B,V,9])
        """
        # Lazy build: if d_gauss was 0 at init, we build on first forward
        if not hasattr(self, "gaussian_encoder"):
            x0_tmp = gaussians_to_attrs(gaussians_blurry)
            self._build(x0_tmp.shape[-1])   # build with actual full d_gauss
            self.to(gaussians_blurry.means.device)

        if num_steps is None:
            num_steps = self.cfg.train_num_steps if self.training else self.cfg.infer_num_steps

        # ── 1. Flatten Gaussians → attribute vector ────────────────────────
        x0 = gaussians_to_attrs(gaussians_blurry)   # [B, G, D]  coarse (= starting point)
        x = x0.clone()
        B, G, D = x.shape

        # ── 2. Encode blurry Gaussians as conditioning ─────────────────────
        cond_tokens = self.gaussian_encoder(x0)     # [B, K, d_tok]
        _cam_tokens_stored = None                   # saved for Design A camera pred head

        # ── 2b. GT camera conditioning (ProPE 9-dim pose_enc → d_tok) ────────
        if (
            self.cfg.use_cam_cond
            and hasattr(self, "cam_encoder")
            and context_pose_enc is not None
        ):
            cam_tokens = self.cam_encoder(context_pose_enc.to(x.device))   # [B, V, d_tok]
            _cam_tokens_stored = cam_tokens
            cond_tokens = torch.cat([cond_tokens, cam_tokens], dim=1)      # [B, K+V, d_tok]

        # ── 2c. 2D prior tokens (DA3 per-patch tokens) ───────────────────────
        # blurry_patch_tokens:  [B, V, N, C]    — per-patch spatial tokens (full spatial resolution)
        #   → projected per-patch, then concatenated as V*N extra conditioning tokens.
        if self.cfg.use_vggt_priors:
            if blurry_patch_tokens is not None:
                pt = blurry_patch_tokens.to(x.device)
                if pt.dim() == 3:
                    # Legacy [B, V, C] (mean-pooled) — project as V tokens
                    vggt_patch = self.vggt_patch_projector(pt)           # [B, V, d_tok]
                else:
                    # [B, V, N, C] — project per patch, flatten to V*N tokens
                    B_, V_, N_, C_ = pt.shape
                    vggt_patch = self.vggt_patch_projector(
                        pt.reshape(B_, V_ * N_, C_)
                    )                                                     # [B, V*N, d_tok]
                cond_tokens = torch.cat([cond_tokens, vggt_patch], dim=1)

        # ── 3. Geometry + DC color attr vector for the ODE ────────────────
        # ODE refines: means(3) + log_scales(3) + quats(4) + logit_opa(1) + dc_color(3) = 14 dims.
        # DC color included so DiffHead can correct blurry SH appearance.
        # Higher SH (degree ≥ 1) remain pass-through (rarely contribute to deblurring).
        D_geom = 14   # means(3)+log_scales(3)+quats(4)+logit_opa(1)+dc_color(3)
        x_geom = x[:, :, :D_geom].clone()   # [B, G, 14]

        # ── 4. Euler ODE on geometry only ─────────────────────────────────
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((B, G), t_val, device=x_geom.device, dtype=x_geom.dtype)

            velocity = self.fm_denoiser(
                [cond_tokens],
                query_points=x_geom,
                timestep=t,
            )                           # [B, G, D_geom]

            # All 5 Gaussian attrs are refined: means, log_scales, quats, logit_opa, dc_color.
            # Only means velocity is clamped to prevent exploding translations.
            v_means  = velocity[:, :, 0:3].clamp(-0.3, 0.3)
            velocity = torch.cat([v_means, velocity[:, :, 3:]], dim=-1)

            x_new = x_geom + dt * velocity

            # Normalize quats to unit norm — out-of-place.
            q = x_new[:, :, 6:10] / (x_new[:, :, 6:10].norm(dim=-1, keepdim=True) + 1e-8)
            x_geom = torch.cat([x_new[:, :, :6], q, x_new[:, :, 10:]], dim=-1)

        # ── 5. Reconstruct full attr vector: refined geometry + original harmonics ──
        x_out = x0.clone()
        x_out[:, :, :D_geom] = x_geom

        # ── 6. Convert back to Gaussians ──────────────────────────────────
        gaussians_sharp = attrs_to_gaussians(x_out, gaussians_blurry)

        # ── 7. Design A: predict per-view camera pose_enc ─────────────────
        # Combine global shape context with per-view cam_tokens → predict 9-dim pose_enc.
        # cam_tokens are available only when use_cam_cond=True.
        if self.cfg.use_cam_pred and hasattr(self, "camera_pred_head") and _cam_tokens_stored is not None:
            global_shape = cond_tokens[:, :self.cfg.enc_k].mean(dim=1, keepdim=True)  # [B,1,d]
            global_shape = global_shape.expand(-1, _cam_tokens_stored.shape[1], -1)   # [B,V,d]
            pred_input   = torch.cat([_cam_tokens_stored, global_shape], dim=-1)       # [B,V,2d]
            pred_pose_enc = self.camera_pred_head(pred_input)                          # [B,V,9]
            return gaussians_sharp, pred_pose_enc

        return gaussians_sharp

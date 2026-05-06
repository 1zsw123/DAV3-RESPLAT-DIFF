"""
MipSplattingRefiner: per-scene test-time Gaussian refinement using
MipSplatting's anti-aliased CUDA rasterizer.

Key difference from plain C3G TTO:
  - Uses MipSplatting's 3D Smoothing Filter + 2D Mip Filter (kernel_size)
    instead of C3G's 2D screen-space low-pass clamp (low_pass_filter).
  - The 3D smoothing filter adds a minimum isotropic covariance
    Σ_smooth = Σ + kernel_size² I, preventing close-range aliasing.
  - The 2D mip filter integrates the projected Gaussian over the pixel footprint,
    preventing far-range aliasing.
  These are the core anti-aliasing contributions of MipSplatting (Barron et al., 2023).

Pipeline:
  coarse Gaussians (C3G encoder)
      ↓  _refine_single  (Adam, num_opt_steps steps)
  refined Gaussians  →  C3G decoder  →  photometric loss

Reference:
  Barron et al., "Zip-NeRF" / MipSplatting (2023)
  https://github.com/autonomousvision/mip-splatting
"""

from __future__ import annotations

import glob as _glob
import importlib.util as _ilu
import os
import sys
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .types import Gaussians
from .mip_splatting_refiner_utils import (
    _cov_to_log_scales_quat,
    _scales_quat_to_cov,
    _matrix_to_quat,
    _quat_to_matrix,
)

from ..geometry.projection import get_fov

# ── Load MipSplatting _C extension directly (avoids sys.modules conflict) ────
# C3G's own diff_gaussian_rasterization (no kernel_size) gets JIT-compiled and
# cached in sys.modules before this file is imported. A plain sys.path.insert
# + import returns that cached version instead of MipSplatting's. We bypass
# the module registry by loading _C.so directly via importlib.
_MIP_C_PATTERN = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "mip-splatting",
    "submodules", "diff-gaussian-rasterization",
    "diff_gaussian_rasterization", "_C*.so",
))
_mip_C_paths = _glob.glob(_MIP_C_PATTERN)
if not _mip_C_paths:
    raise ImportError(
        f"MipSplatting _C extension not found at {_MIP_C_PATTERN}.\n"
        "Run: cd mip-splatting/submodules/diff-gaussian-rasterization "
        "&& python setup.py build_ext --inplace"
    )
# Module name must match PyInit__C symbol exported by the .so
_mip_C_spec = _ilu.spec_from_file_location("_C", _mip_C_paths[0])
_mip_C = _ilu.module_from_spec(_mip_C_spec)
sys.modules["_mip_gaussian_rast._C"] = _mip_C   # register under private key
_mip_C_spec.loader.exec_module(_mip_C)


# ── Inline wrappers (adapted from MipSplatting __init__.py) ──────────────────

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    kernel_size: float
    subpixel_offset: torch.Tensor
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, means2D, sh, colors_precomp, opacities,
                scales, rotations, cov3Ds_precomp, raster_settings):
        args = (
            raster_settings.bg, means3D, colors_precomp, opacities,
            scales, rotations, raster_settings.scale_modifier, cov3Ds_precomp,
            raster_settings.viewmatrix, raster_settings.projmatrix,
            raster_settings.tanfovx, raster_settings.tanfovy,
            raster_settings.kernel_size, raster_settings.subpixel_offset,
            raster_settings.image_height, raster_settings.image_width,
            sh, raster_settings.sh_degree, raster_settings.campos,
            raster_settings.prefiltered, raster_settings.debug,
        )
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = \
            _mip_C.rasterize_gaussians(*args)
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations,
                              cov3Ds_precomp, radii, sh,
                              geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        num_rendered = ctx.num_rendered
        rs = ctx.raster_settings
        (colors_precomp, means3D, scales, rotations, cov3Ds_precomp,
         radii, sh, geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors
        args = (
            rs.bg, means3D, radii, colors_precomp, scales, rotations,
            rs.scale_modifier, cov3Ds_precomp,
            rs.viewmatrix, rs.projmatrix, rs.tanfovx, rs.tanfovy,
            rs.kernel_size, rs.subpixel_offset,
            grad_out_color, sh, rs.sh_degree, rs.campos,
            geomBuffer, num_rendered, binningBuffer, imgBuffer, rs.debug,
        )
        # CUDA backward returns:
        #   (dL_dmeans2D, dL_dcolors, dL_dopacities, dL_dmeans3D,
        #    dL_dcov3D, dL_dsh, dL_dscales, dL_drotations)
        # Must reorder to match forward input order:
        #   (means3D, means2D, sh, colors_precomp, opacities,
        #    scales, rotations, cov3Ds_precomp, raster_settings)
        (dL_dmeans2D, dL_dcolors, dL_dopacities, dL_dmeans3D,
         dL_dcov3D, dL_dsh, dL_dscales, dL_drotations) = \
            _mip_C.rasterize_gaussians_backward(*args)
        return (dL_dmeans3D, dL_dmeans2D, dL_dsh, dL_dcolors,
                dL_dopacities, dL_dscales, dL_drotations, dL_dcov3D, None)


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings) -> None:
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, means2D, opacities, shs=None,
                colors_precomp=None, scales=None, rotations=None,
                cov3D_precomp=None):
        rs = self.raster_settings
        shs            = shs            if shs            is not None else torch.Tensor([])
        colors_precomp = colors_precomp if colors_precomp is not None else torch.Tensor([])
        scales         = scales         if scales         is not None else torch.Tensor([])
        rotations      = rotations      if rotations      is not None else torch.Tensor([])
        cov3D_precomp  = cov3D_precomp  if cov3D_precomp  is not None else torch.Tensor([])
        return _RasterizeGaussians.apply(
            means3D, means2D, shs, colors_precomp, opacities,
            scales, rotations, cov3D_precomp, rs,
        )


# ─────────────────────────────── Config ──────────────────────────────────────

def _compute_filter_3d(
    means: torch.Tensor,        # [G, 3]  world-space Gaussian centres
    extrinsics: torch.Tensor,   # [V, 4, 4]  C2W  (C3G convention)
    intrinsics: torch.Tensor,   # [V, 3, 3]  normalised K (fx=K[0,0]/width)
    near: torch.Tensor,         # [V]
    h: int,
    w: int,
    filter_boost: float = 1.0,
    _print_depth: bool = False,
) -> torch.Tensor:
    """
    Compute per-Gaussian 3D filter radius (world space) as in MipSplatting.
    filter_3D[g] = min_depth_across_views / max_focal_px * sqrt(0.2) * filter_boost
    Returns tensor [G, 1].
    """
    G = means.shape[0]
    device = means.device
    distance = torch.full((G,), 1e6, device=device)
    valid_any = torch.zeros(G, dtype=torch.bool, device=device)
    max_focal = 0.0

    for i in range(extrinsics.shape[0]):
        # extrinsics[i] is C2W; invert to get W2C
        W2C = extrinsics[i].inverse()             # [4, 4]
        R = W2C[:3, :3]                            # [3, 3]
        T = W2C[:3, 3]                             # [3]
        xyz_cam = means @ R.T + T.unsqueeze(0)    # [G, 3]
        depth   = xyz_cam[:, 2]                   # [G]

        # focal length in pixels
        focal_x = float(intrinsics[i, 0, 0]) * w
        focal_y = float(intrinsics[i, 1, 1]) * h
        focal   = max(focal_x, focal_y)
        if focal > max_focal:
            max_focal = focal

        valid = depth > float(near[i])
        if _print_depth and valid.any():
            print(f"  [DIAG filter] view{i}: depth[valid] "
                  f"min={depth[valid].min():.4f} max={depth[valid].max():.4f} "
                  f"focal_px={focal:.1f}", flush=True)
        distance[valid] = torch.min(distance[valid], depth[valid])
        valid_any |= valid

    distance[~valid_any] = distance[valid_any].max() if valid_any.any() else 1.0
    filter_3d = distance / max(max_focal, 1.0) * (0.2 ** 0.5) * filter_boost
    return filter_3d.unsqueeze(-1)   # [G, 1]


def _get_projection_matrix(
    near: torch.Tensor,   # [1]
    far: torch.Tensor,    # [1]
    fov_x: torch.Tensor,  # [1]
    fov_y: torch.Tensor,  # [1]
) -> torch.Tensor:
    """OpenGL-style projection matrix (same formula as C3G's get_projection_matrix)."""
    tan_x = (0.5 * fov_x).tan()
    tan_y = (0.5 * fov_y).tan()
    top   =  tan_y * near;  bottom = -top
    right =  tan_x * near;  left   = -right
    P = torch.zeros(4, 4, dtype=torch.float32, device=near.device)
    P[0, 0] = 2 * near / (right - left)
    P[1, 1] = 2 * near / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = far / (far - near)
    P[2, 3] = -(far * near) / (far - near)
    return P


@dataclass
class MipSplattingRefinerCfg:
    # Optimization
    num_opt_steps: int = 3000         # gradient steps per sample (full TTO)
    lr_xyz: float = 1.6e-4
    lr_scale: float = 5e-3
    lr_rotation: float = 1e-3
    lr_color: float = 2.5e-3
    lr_opacity: float = 5e-2
    # MipSplatting renderer
    kernel_size: float = 0.1          # 3D smoothing filter σ
    filter_boost: float = 1.0         # multiply filter_3D by this factor (compensate for near<<1 scenes)
    make_scale_invariant: bool = True
    background_color: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Densification — matches mip-splatting train.py defaults
    densify_from_iter: int = 500
    densify_until_iter: int = 15000   # Phase A: densify+prune (GS count grows)
    densification_interval: int = 100
    densify_grad_threshold: float = 0.0002
    opacity_reset_interval: int = 3000
    min_opacity: float = 0.005
    percent_dense: float = 0.01
    max_screen_size: int = 20         # Phase B (>densify_until_iter): Adam + compute_3D_filter only
    max_gaussians: int = 24000        # hard cap on GS count during densification


# ─────────────────────────────── Module ──────────────────────────────────────

class MipSplattingRefiner(nn.Module):
    """
    Per-scene test-time Gaussian refinement with MipSplatting's rasterizer.

    No persistent learnable parameters.  For each sample in a batch:
      1. Wrap Gaussian attributes as nn.Parameters.
      2. Run Adam for cfg.num_opt_steps steps, rendering with MipSplatting's
         anti-aliased rasterizer (3D smoothing + 2D mip filter).
      3. Return refined Gaussians (detached).
    """

    def __init__(self, cfg: MipSplattingRefinerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        context: dict,
        h: int = 0,
        w: int = 0,
    ) -> Gaussians:
        B = gaussians.means.shape[0]
        refined_list = []

        for b in range(B):
            feat = gaussians.feature[b] if gaussians.feature is not None else None
            refined = self._refine_single(
                means=gaussians.means[b].detach().clone(),
                covariances=gaussians.covariances[b].detach().clone(),
                harmonics=gaussians.harmonics[b].detach().clone(),
                opacities=gaussians.opacities[b].detach().clone(),
                feature=feat.detach().clone() if feat is not None else None,
                extrinsics=context["extrinsics"][b],
                intrinsics=context["intrinsics"][b],
                near=context["near"][b],
                far=context["far"][b],
                images=context["image"][b],
            )
            refined_list.append(refined)

        # Densification may produce different G counts per sample; pad to max.
        max_g = max(r.means.shape[0] for r in refined_list)

        def _pad(t: torch.Tensor, target: int) -> torch.Tensor:
            n = target - t.shape[0]
            if n == 0:
                return t
            pad = torch.zeros((n, *t.shape[1:]), dtype=t.dtype, device=t.device)
            return torch.cat([t, pad], dim=0)

        padded_opa = []
        for r in refined_list:
            n = max_g - r.opacities.shape[0]
            if n == 0:
                padded_opa.append(r.opacities)
            else:
                padded_opa.append(torch.cat([r.opacities, torch.zeros(n, dtype=r.opacities.dtype, device=r.opacities.device)]))

        return Gaussians(
            means=torch.stack([_pad(r.means, max_g) for r in refined_list]),
            covariances=torch.stack([_pad(r.covariances, max_g) for r in refined_list]),
            harmonics=torch.stack([_pad(r.harmonics, max_g) for r in refined_list]),
            opacities=torch.stack(padded_opa),
            feature=(
                torch.stack([_pad(r.feature, max_g) for r in refined_list])
                if refined_list[0].feature is not None
                else gaussians.feature
            ),
        )

    @torch.enable_grad()
    def _refine_single(
        self,
        means: Float[Tensor, "gaussian 3"],
        covariances: Float[Tensor, "gaussian 3 3"],
        harmonics: Float[Tensor, "gaussian 3 d_sh"],
        opacities: Float[Tensor, "gaussian"],
        feature: Optional[Float[Tensor, "gaussian f"]],
        extrinsics: Float[Tensor, "view 4 4"],   # W2C
        intrinsics: Float[Tensor, "view 3 3"],   # normalised K
        near: Float[Tensor, "view"],
        far: Float[Tensor, "view"],
        images: Float[Tensor, "view 3 h w"],
    ) -> Gaussians:
        device = means.device
        _, _, h, w = images.shape
        V = extrinsics.shape[0]
        d_sh = harmonics.shape[-1]
        sh_degree = int(d_sh ** 0.5) - 1

        # ── 1. Covariance → (log_scales, quaternions) ─────────────────────
        log_scales, quaternions = _cov_to_log_scales_quat(covariances)

        opc = opacities.clamp(1e-6, 1.0 - 1e-6)
        logit_opacities = torch.log(opc / (1.0 - opc))

        # ── 2. Wrap in nn.Parameter ───────────────────────────────────────
        p_means       = nn.Parameter(means.clone())
        p_log_scales  = nn.Parameter(log_scales.clone())
        p_quaternions = nn.Parameter(quaternions.clone())
        p_harmonics   = nn.Parameter(harmonics.clone())
        p_logit_opa   = nn.Parameter(logit_opacities.unsqueeze(-1).clone())  # [G, 1]

        optimizer = torch.optim.Adam(
            [
                {"params": [p_means],       "lr": self.cfg.lr_xyz,      "name": "xyz"},
                {"params": [p_log_scales],  "lr": self.cfg.lr_scale,    "name": "scaling"},
                {"params": [p_quaternions], "lr": self.cfg.lr_rotation, "name": "rotation"},
                {"params": [p_harmonics],   "lr": self.cfg.lr_color,    "name": "f_dc"},
                {"params": [p_logit_opa],   "lr": self.cfg.lr_opacity,  "name": "opacity"},
            ],
            lr=0.0,
            eps=1e-15,
        )

        # ── 3. Densification bookkeeping ─────────────────────────────────
        G0 = p_means.shape[0]
        xyz_grad_accum     = torch.zeros((G0, 1), device=device)
        xyz_grad_accum_abs = torch.zeros((G0, 1), device=device)
        denom              = torch.zeros((G0, 1), device=device)
        max_radii2D        = torch.zeros(G0, device=device)

        # scene extent: used by densify_and_prune to judge "large" Gaussians
        scene_extent = float(means.abs().max().item()) * 2.0 + 1e-6

        # ── 3D filter (MipSplatting anti-aliasing) ────────────────────────
        # Computed once initially, then recomputed after every densification
        # and every 100 steps during Phase B.  NOT an optimizable parameter.
        filter_3D = _compute_filter_3d(
            p_means.data, extrinsics, intrinsics, near, h, w,
            filter_boost=self.cfg.filter_boost, _print_depth=True,
        )   # [G, 1]

        # ── Diagnostic: print filter_3D vs scale stats at init ───────────
        with torch.no_grad():
            scales_raw = torch.exp(p_log_scales)
            ratio = filter_3D.squeeze(-1) / scales_raw.mean(dim=-1).clamp(min=1e-9)
            print(f"[DIAG] scene_extent={scene_extent:.4f}  "
                  f"p_means.abs().max={p_means.data.abs().max():.4f}  "
                  f"near={near.min():.4f}~{near.max():.4f}")
            print(f"[DIAG] filter_3D: min={filter_3D.min():.6f}  "
                  f"mean={filter_3D.mean():.6f}  max={filter_3D.max():.6f}")
            print(f"[DIAG] scales:    min={scales_raw.min():.6f}  "
                  f"mean={scales_raw.mean():.6f}  max={scales_raw.max():.6f}")
            print(f"[DIAG] filter/scale ratio: "
                  f"min={ratio.min():.4f}  mean={ratio.mean():.4f}  max={ratio.max():.4f}",
                  flush=True)

        def _get_params():
            return p_means, p_log_scales, p_quaternions, p_harmonics, p_logit_opa

        def _reset_bookkeeping():
            nonlocal xyz_grad_accum, xyz_grad_accum_abs, denom, max_radii2D
            G = p_means.shape[0]
            xyz_grad_accum     = torch.zeros((G, 1), device=device)
            xyz_grad_accum_abs = torch.zeros((G, 1), device=device)
            denom              = torch.zeros((G, 1), device=device)
            max_radii2D        = torch.zeros(G, device=device)

        def _recompute_filter():
            nonlocal filter_3D
            filter_3D = _compute_filter_3d(
                p_means.data, extrinsics, intrinsics, near, h, w,
                filter_boost=self.cfg.filter_boost,
            )

        def _prune(mask):
            """Remove Gaussians where mask=True, update optimizer states in-place."""
            nonlocal p_means, p_log_scales, p_quaternions, p_harmonics, p_logit_opa
            nonlocal xyz_grad_accum, xyz_grad_accum_abs, denom, max_radii2D
            nonlocal filter_3D
            keep = ~mask
            params_map = {
                "xyz": p_means, "scaling": p_log_scales, "rotation": p_quaternions,
                "f_dc": p_harmonics, "opacity": p_logit_opa,
            }
            new_params = {}
            for name, p in params_map.items():
                stored = optimizer.state.get(p, None)
                new_p = nn.Parameter(p.data[keep].requires_grad_(True))
                if stored is not None:
                    optimizer.state[new_p] = {
                        "exp_avg":    stored["exp_avg"][keep],
                        "exp_avg_sq": stored["exp_avg_sq"][keep],
                        "step":       stored["step"],
                    }
                    del optimizer.state[p]
                new_params[name] = new_p
            for pg in optimizer.param_groups:
                pg["params"][0] = new_params[pg["name"]]
            p_means, p_log_scales, p_quaternions, p_harmonics, p_logit_opa = (
                new_params["xyz"], new_params["scaling"], new_params["rotation"],
                new_params["f_dc"], new_params["opacity"],
            )
            xyz_grad_accum     = xyz_grad_accum[keep]
            xyz_grad_accum_abs = xyz_grad_accum_abs[keep]
            denom              = denom[keep]
            max_radii2D        = max_radii2D[keep]
            filter_3D          = filter_3D[keep]

        def _cat_new(new_dict):
            """Append new Gaussians to optimizer param groups."""
            nonlocal p_means, p_log_scales, p_quaternions, p_harmonics, p_logit_opa
            param_map = {
                "xyz": p_means, "scaling": p_log_scales, "rotation": p_quaternions,
                "f_dc": p_harmonics, "opacity": p_logit_opa,
            }
            new_params = {}
            for name, p in param_map.items():
                ext = new_dict[name]
                stored = optimizer.state.get(p, None)
                new_p = nn.Parameter(torch.cat([p.data, ext], dim=0).requires_grad_(True))
                if stored is not None:
                    optimizer.state[new_p] = {
                        "exp_avg":    torch.cat([stored["exp_avg"],    torch.zeros_like(ext)], dim=0),
                        "exp_avg_sq": torch.cat([stored["exp_avg_sq"], torch.zeros_like(ext)], dim=0),
                        "step":       stored["step"],
                    }
                    del optimizer.state[p]
                new_params[name] = new_p
            for pg in optimizer.param_groups:
                pg["params"][0] = new_params[pg["name"]]
            p_means, p_log_scales, p_quaternions, p_harmonics, p_logit_opa = (
                new_params["xyz"], new_params["scaling"], new_params["rotation"],
                new_params["f_dc"], new_params["opacity"],
            )
            _reset_bookkeeping()
            _recompute_filter()   # recompute filter_3D for expanded set

        def _densify_and_prune(iteration):
            nonlocal xyz_grad_accum, xyz_grad_accum_abs, denom
            # Compute per-point gradient norms (size = current G before this call)
            grads     = xyz_grad_accum / denom.clamp(min=1)
            grads[grads.isnan()] = 0.0
            grads_abs = xyz_grad_accum_abs / denom.clamp(min=1)
            grads_abs[grads_abs.isnan()] = 0.0

            ratio = (grads.norm(dim=-1) >= self.cfg.densify_grad_threshold).float().mean()
            Q = torch.quantile(grads_abs.reshape(-1), (1 - ratio).clamp(0, 1))

            G_orig = p_means.shape[0]
            scales_orig = torch.exp(p_log_scales.data)   # [G_orig, 3]

            if G_orig < self.cfg.max_gaussians:
                # ── Clone: small Gaussians with high gradient ─────────────────
                sel_clone = ((grads.norm(dim=-1) >= self.cfg.densify_grad_threshold) |
                             (grads_abs.norm(dim=-1) >= Q))
                sel_clone = sel_clone & (scales_orig.max(dim=1).values <= self.cfg.percent_dense * scene_extent)
                if sel_clone.any():
                    _cat_new({
                        "xyz":      p_means.data[sel_clone],
                        "scaling":  p_log_scales.data[sel_clone],
                        "rotation": p_quaternions.data[sel_clone],
                        "f_dc":     p_harmonics.data[sel_clone],
                        "opacity":  p_logit_opa.data[sel_clone],
                    })

                # ── Split: large Gaussians with high gradient ─────────────────
                # After clone, p_means has G_orig + N_cloned points.
                # Pad grads to current size (new cloned points get grad=0 → not split).
                G_now = p_means.shape[0]
                padded_grads     = torch.zeros(G_now, 1, device=device)
                padded_grads_abs = torch.zeros(G_now, 1, device=device)
                padded_grads[:G_orig]     = grads
                padded_grads_abs[:G_orig] = grads_abs

                scales_now = torch.exp(p_log_scales.data)
                sel_split = ((padded_grads.norm(dim=-1) >= self.cfg.densify_grad_threshold) |
                             (padded_grads_abs.norm(dim=-1) >= Q))
                sel_split = sel_split & (scales_now.max(dim=1).values > self.cfg.percent_dense * scene_extent)
                if sel_split.any():
                    N = 2
                    stds     = scales_now[sel_split].repeat(N, 1)
                    rots_mat = _quat_to_matrix(F.normalize(p_quaternions.data[sel_split], dim=-1))
                    samples  = torch.bmm(rots_mat.repeat(N, 1, 1),
                                         torch.normal(torch.zeros_like(stds), stds).unsqueeze(-1)).squeeze(-1)
                    new_xyz   = p_means.data[sel_split].repeat(N, 1) + samples
                    new_scale = torch.log(scales_now[sel_split].repeat(N, 1) / (0.8 * N))
                    _cat_new({
                        "xyz":      new_xyz,
                        "scaling":  new_scale,
                        "rotation": p_quaternions.data[sel_split].repeat(N, 1),
                        "f_dc":     p_harmonics.data[sel_split].repeat(N, 1, 1),
                        "opacity":  p_logit_opa.data[sel_split].repeat(N, 1),
                    })
                    # Remove the original split points (new points appended at end, keep them)
                    prune_split = torch.cat([
                        sel_split,
                        torch.zeros(N * sel_split.sum(), dtype=torch.bool, device=device)
                    ])
                    _prune(prune_split)

            # ── Prune: enforce max_gaussians hard cap ─────────────────────
            if p_means.shape[0] > self.cfg.max_gaussians:
                opa = torch.sigmoid(p_logit_opa.data).squeeze(-1)
                keep_k = self.cfg.max_gaussians
                _, keep_idx = torch.topk(opa, keep_k, largest=True)
                keep_mask = torch.zeros(opa.shape[0], dtype=torch.bool, device=opa.device)
                keep_mask[keep_idx] = True
                _prune(~keep_mask)

            # ── Prune: low opacity + large screen-size ─────────────────────
            opa_now = torch.sigmoid(p_logit_opa.data).squeeze(-1)
            prune_mask = opa_now < self.cfg.min_opacity
            size_threshold = self.cfg.max_screen_size if iteration > self.cfg.opacity_reset_interval else None
            if size_threshold is not None:
                big_vs = max_radii2D > size_threshold
                big_ws = torch.exp(p_log_scales.data).max(dim=1).values > 0.1 * scene_extent
                prune_mask = prune_mask | big_vs | big_ws
            _prune(prune_mask)

        # ── 4. Full TTO loop ──────────────────────────────────────────────
        target = (images + 1.0) * 0.5   # [-1,1] → [0,1]
        bg     = self.background_color.to(device)
        raster_settings_list = self._build_raster_settings(
            extrinsics, intrinsics, near, far, h, w, bg, sh_degree, device
        )


        import wandb as _wandb
        for iteration in range(1, self.cfg.num_opt_steps + 1):
            _log_this_step = (iteration % 500 == 0 or iteration == 1)
            v_idx = int(torch.randint(0, V, (1,)).item())

            scales   = torch.exp(p_log_scales)
            q_norm   = F.normalize(p_quaternions, p=2, dim=-1)
            curr_opa = torch.sigmoid(p_logit_opa)
            if curr_opa.dim() == 1:
                curr_opa = curr_opa.unsqueeze(-1)

            # ── Apply 3D filter (MipSplatting anti-aliasing) ─────────────
            # effective scale: sqrt(s² + filter_3D²) prevents needle structures
            scales_sq  = scales ** 2
            filter_sq  = filter_3D ** 2
            scales_eff = torch.sqrt(scales_sq + filter_sq)            # [G, 3]
            # opacity correction: attenuate by volume ratio
            det_s2  = scales_sq.prod(dim=-1)                          # [G]
            det_sf2 = (scales_sq + filter_sq).prod(dim=-1)            # [G]
            opa_coef = torch.sqrt((det_s2 / (det_sf2 + 1e-10)).clamp(0, 1)).unsqueeze(-1)  # [G,1]
            opa_eff  = curr_opa * opa_coef                            # [G, 1]

            means2D = torch.zeros(p_means.shape[0], 3, device=device, requires_grad=True)

            rasterizer = GaussianRasterizer(raster_settings=raster_settings_list[v_idx]["settings"])
            shs_arg = p_harmonics.permute(0, 2, 1).contiguous()

            color, radii = rasterizer(
                means3D=p_means, means2D=means2D, shs=shs_arg,
                colors_precomp=None, opacities=opa_eff,
                scales=scales_eff, rotations=q_norm, cov3D_precomp=None,
            )

            loss = F.l1_loss(color, target[v_idx])
            if _log_this_step:
                _g = p_means.shape[0]
                print(f"[TTO] iter {iteration}/{self.cfg.num_opt_steps}  G={_g}  loss={loss.item():.4f}", flush=True)
                try:
                    if _wandb.run is not None:
                        _wandb.log({
                            "tto/iteration": iteration,
                            "tto/n_gaussians": _g,
                            "tto/loss_l1": loss.item(),
                        })
                except Exception:
                    pass
            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                # ── Densification bookkeeping ────────────────────────────
                if iteration < self.cfg.densify_until_iter:
                    vis = radii > 0
                    if means2D.grad is not None:
                        g = means2D.grad
                        xyz_grad_accum[vis]     += g[vis, :2].norm(dim=-1, keepdim=True)
                        xyz_grad_accum_abs[vis] += g[vis, 2:].norm(dim=-1, keepdim=True)
                        denom[vis]              += 1
                    max_radii2D[vis] = torch.max(max_radii2D[vis], radii[vis].float())

                    # Densify + prune
                    if (iteration > self.cfg.densify_from_iter and
                            iteration % self.cfg.densification_interval == 0):
                        _densify_and_prune(iteration)
                        # Rebuild raster settings with current h/w (unchanged)
                        raster_settings_list = self._build_raster_settings(
                            extrinsics, intrinsics, near, far, h, w, bg, sh_degree, device
                        )

                    # Opacity reset
                    if iteration % self.cfg.opacity_reset_interval == 0:
                        reset_val = torch.log(torch.tensor(0.01 / 0.99, device=device))
                        p_logit_opa.data.fill_(reset_val.item())

                # Phase B: recompute 3D filter every 100 steps (positions have changed)
                elif iteration % 100 == 0 and iteration < self.cfg.num_opt_steps - 100:
                    _recompute_filter()

            optimizer.step()

            with torch.no_grad():
                p_harmonics.data.clamp_(min=-3.0, max=3.0)

        # ── 5. Convert back to C3G Gaussians ─────────────────────────────
        with torch.no_grad():
            scales_f = torch.exp(p_log_scales)
            q_f      = F.normalize(p_quaternions, p=2, dim=-1)
            cov_f    = _scales_quat_to_cov(scales_f, q_f)
            opa_f    = torch.sigmoid(p_logit_opa).squeeze(-1)

        return Gaussians(
            means=p_means.detach(),
            covariances=cov_f.detach(),
            harmonics=p_harmonics.detach(),
            opacities=opa_f.detach(),
            feature=feature.expand(p_means.shape[0], -1) if feature is not None else None,
        )

    def _build_raster_settings(
        self,
        extrinsics,    # [V, 4, 4]  C2W  (C3G convention)
        intrinsics,    # [V, 3, 3]  normalised K
        near,          # [V]
        far,           # [V]
        h, w,
        bg,
        sh_degree,
        device,
    ) -> list[dict]:
        """Pre-compute MipSplatting GaussianRasterizationSettings for each view."""
        V = extrinsics.shape[0]
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)  # [V]

        settings_list = []
        for i in range(V):
            near_i  = near[i]
            far_i   = far[i]

            # No scale-invariant in refiner: per-view scale breaks Adam
            # (each view has different near → different effective LR per gradient step).
            # MipSplatting rasterizer works fine in raw world coordinates.
            # C3G decoder applies its own scale-invariant on the returned means.
            view_mat = extrinsics[i].inverse().T.clone()   # W2C^T, world coords
            proj     = _get_projection_matrix(near_i, far_i, fov_x[i], fov_y[i])
            full_proj = view_mat @ proj.T

            # C2W[:3, 3] is the camera centre in world space
            campos = extrinsics[i, :3, 3]
            scale  = torch.ones(1, device=device)

            tanfovx = float((0.5 * fov_x[i]).tan())
            tanfovy = float((0.5 * fov_y[i]).tan())

            subpixel_offset = torch.zeros(h, w, 2, dtype=torch.float32, device=device)

            rs = GaussianRasterizationSettings(
                image_height=h,
                image_width=w,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                kernel_size=self.cfg.kernel_size,   # ← MipSplatting 3D smoothing
                subpixel_offset=subpixel_offset,
                bg=bg,
                scale_modifier=1.0,
                viewmatrix=view_mat.contiguous(),
                projmatrix=full_proj.contiguous(),
                sh_degree=sh_degree,
                campos=campos.contiguous(),
                prefiltered=False,
                debug=False,
            )
            settings_list.append({"settings": rs, "scale": scale})

        return settings_list

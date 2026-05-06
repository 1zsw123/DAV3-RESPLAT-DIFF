"""
DSLR-supervised loss with:
  1. Per-sample affine exposure compensation (render → DSLR color space)
  2. Valid-region mask (anon stickers + black border excluded)
  3. L1 + LPIPS on valid pixels only
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDslrCfg:
    weight: float
    lpips_weight: float = 0.05
    exp_comp_blur_sigma: float = 0.05   # fraction of image width for blur kernel
    exp_comp_min_valid: float = 0.1     # skip exp-comp if <10% valid pixels


@dataclass
class LossDslrCfgWrapper:
    dslr: LossDslrCfg


def _gaussian_blur2d(x: Tensor, sigma: float) -> Tensor:
    """Approximate Gaussian blur with two separable 1D convolutions."""
    H, W = x.shape[-2], x.shape[-1]
    k = max(3, int(sigma * min(H, W)) * 2 | 1)   # odd kernel size
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g = torch.exp(-0.5 * (coords / (sigma * min(H, W))) ** 2)
    g = g / g.sum()
    g1d = g.view(1, 1, -1)
    C = x.shape[1]
    # expand to per-channel
    gx = g1d.expand(C, 1, -1)
    gy = g1d.expand(C, 1, -1)
    pad = k // 2
    x = F.conv2d(x, gx.unsqueeze(2), padding=(0, pad), groups=C)
    x = F.conv2d(x, gy.unsqueeze(3), padding=(pad, 0), groups=C)
    return x


def _fit_exposure(render: Tensor, gt: Tensor, mask: Tensor,
                  sigma: float) -> Tensor:
    """
    Fit per-image 3×4 affine E s.t. render_blur @ E[:3] + E[3] ≈ gt_blur
    on valid (mask==1) pixels.

    Args:
        render: [3, H, W]   rendered image in iPhone color space
        gt:     [3, H, W]   DSLR GT (color-corrected)
        mask:   [1, H, W]   float, 1=valid

    Returns:
        E: [4, 3] affine matrix
    """
    with torch.no_grad():
        r_b = _gaussian_blur2d(render.unsqueeze(0), sigma).squeeze(0)  # [3,H,W]
        g_b = _gaussian_blur2d(gt.unsqueeze(0),     sigma).squeeze(0)

        m = mask.squeeze(0).bool()  # [H,W]
        if m.float().mean() < 0.1:
            return torch.eye(3, device=render.device, dtype=render.dtype).T.contiguous()

        src = r_b[:, m].T   # [N, 3]
        tgt = g_b[:, m].T   # [N, 3]
        ones = torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype)
        A = torch.cat([src, ones], dim=1)   # [N, 4]
        # least squares: A @ E ≈ tgt  →  E = (AᵀA)⁻¹Aᵀtgt
        try:
            E = torch.linalg.lstsq(A, tgt).solution   # [4, 3]
            if torch.isnan(E).any():
                raise ValueError("lstsq returned NaN")
        except Exception:
            E = torch.zeros(4, 3, device=render.device, dtype=render.dtype)
            E[:3] = torch.eye(3, device=render.device, dtype=render.dtype)
        return E


def _apply_exposure(render: Tensor, E: Tensor) -> Tensor:
    """Apply affine E [4,3] to render [3,H,W]. Returns [3,H,W] clamped [0,1]."""
    C, H, W = render.shape
    flat = render.permute(1, 2, 0).reshape(-1, C)   # [N, 3]
    ones = torch.ones(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)
    out  = torch.cat([flat, ones], dim=1) @ E        # [N, 3]
    return out.reshape(H, W, C).permute(2, 0, 1).clamp(0, 1)


class LossDslr(Loss[LossDslrCfg, LossDslrCfgWrapper]):

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image=None,
    ) -> Float[Tensor, ""]:
        pred   = prediction.color           # [B, V, 3, H, W]
        gt     = batch["target"]["image"]   # [B, V, 3, H, W]
        masks  = batch["target"].get("mask", None)  # [B, V, 1, H, W] or None

        B, V, C, H, W = pred.shape
        sigma = self.cfg.exp_comp_blur_sigma

        total_loss = pred.new_zeros(())

        for b in range(B):
            for v in range(V):
                r = pred[b, v]   # [3, H, W]
                g = gt[b, v]     # [3, H, W]
                m = masks[b, v] if masks is not None else torch.ones(1, H, W, device=r.device)

                # ── exposure compensation ─────────────────────────────────────
                E = _fit_exposure(r, g, m, sigma)
                r_comp = _apply_exposure(r, E)

                # ── masked L1 ────────────────────────────────────────────────
                diff  = (r_comp - g).abs()
                valid = m.expand_as(diff)
                n_valid = valid.sum().clamp(min=1)
                l1 = (diff * valid).sum() / n_valid

                total_loss = total_loss + l1

        return self.cfg.weight * total_loss / (B * V)

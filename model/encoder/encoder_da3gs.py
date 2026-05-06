"""
EncoderDA3GS: DA3-GIANT + GS head adapter for the C3G training pipeline.

DA3 Gaussians (specs.Gaussians): means, scales, rotations(quat_wxyz), harmonics, opacities
C3G types.Gaussians: means, covariances (R @ diag(s^2) @ R.T), harmonics, opacities, feature
"""

from __future__ import annotations
import sys, types as _types
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from ..types import Gaussians
from .encoder import Encoder

_DA3_SRC = Path(__file__).resolve().parents[4] / "Depth-Anything-3" / "src"
if str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))
for _k in ("moviepy", "moviepy.editor"):
    if _k not in sys.modules:
        sys.modules[_k] = _types.ModuleType(_k)

_DA3_CKPT_DEFAULT = "/gpfs/scratch1/shared/qzhang1/da3_pretrained/DA3-GIANT"
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


@dataclass
class EncoderDA3GSCfg:
    name: Literal["da3gs"]
    checkpoint: str = _DA3_CKPT_DEFAULT


def _quat_wxyz_to_matrix(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y),
        2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def da3_to_c3g(da3_gs) -> Gaussians:
    """Convert depth_anything_3.specs.Gaussians -> C3G types.Gaussians."""
    m, s, r = da3_gs.means, da3_gs.scales, da3_gs.rotations
    h, o    = da3_gs.harmonics, da3_gs.opacities
    if m.dim() == 2:
        m, s, r, h, o = (t.unsqueeze(0) for t in (m, s, r, h, o))
    B, G = m.shape[:2]
    if o.dim() == 4:   o = o[:, :, 0, 0]
    elif o.dim() == 3: o = o[:, :, 0]
    R   = _quat_wxyz_to_matrix(r)                               # [B,G,3,3]
    s2  = (s ** 2).unsqueeze(-1)                                # [B,G,3,1]
    cov = (R * s2.transpose(-1, -2)) @ R.transpose(-1, -2)     # [B,G,3,3]
    feat = torch.zeros(B, G, 0, device=m.device, dtype=m.dtype)
    return Gaussians(means=m, covariances=cov, harmonics=h, opacities=o, feature=feat)


class EncoderDA3GS(Encoder):
    """
    Frozen DA3-GIANT + GS head as encoder for C3G deblur pipeline.
    Only DiffusionHead is trained on top.

    Inputs batch["context"]:
        image      [B, V, 3, H, W]  in [0,1]
        extrinsics [B, V, 4, 4]     c2w
        intrinsics [B, V, 3, 3]     normalised (fx/W, fy/H, cx/W, cy/H)

    Output: C3G types.Gaussians in DA3-normalised world frame.
    """

    def __init__(self, cfg: EncoderDA3GSCfg) -> None:
        super().__init__(cfg)
        from depth_anything_3.api import DepthAnything3
        api = DepthAnything3.from_pretrained(cfg.checkpoint)
        api.eval()
        self.da3 = api.model
        if self.da3.gs_head is None:
            raise RuntimeError(f"DA3 checkpoint has no GS head: {cfg.checkpoint}")
        for p in self.da3.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _normalise_poses(self, w2c: torch.Tensor) -> torch.Tensor:
        """First-view-relative normalisation + median translation = 1 (DA3 convention)."""
        transform = torch.inverse(w2c[:, 0:1]).expand_as(w2c)
        w2c_n = w2c @ transform
        c2w_n = torch.inverse(w2c_n)
        dists = c2w_n[:, :, :3, 3].norm(dim=-1)
        median_dist = torch.median(dists, dim=1).values.clamp(min=0.1)
        w2c_n = w2c_n.clone()
        w2c_n[:, :, :3, 3] /= median_dist.view(-1, 1, 1)
        return w2c_n

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        context_feature=None,
    ) -> Gaussians:
        images    = context["image"]       # [B, V, 3, H, W]
        c2w       = context["extrinsics"]  # [B, V, 4, 4]
        intr_norm = context["intrinsics"]  # [B, V, 3, 3]

        B, V, _, H, W = images.shape

        # c2w -> w2c -> DA3-normalised
        w2c   = torch.inverse(c2w.view(B * V, 4, 4)).view(B, V, 4, 4)
        w2c_n = self._normalise_poses(w2c)

        # normalised intrinsics -> pixel-scale (DA3 expects pixel-space)
        K_px = intr_norm.clone()
        K_px[..., 0, 0] *= W;  K_px[..., 0, 2] *= W
        K_px[..., 1, 1] *= H;  K_px[..., 1, 2] *= H

        # ImageNet normalise
        mean = _IMAGENET_MEAN.to(images)
        std  = _IMAGENET_STD.to(images)
        imgs_in = (images - mean) / std

        with torch.no_grad():
            output = self.da3.forward(imgs_in, extrinsics=w2c_n, intrinsics=K_px,
                                      infer_gs=True, return_tokens=True)

        da3_gs = getattr(output, "gaussians", None)
        if da3_gs is None and isinstance(output, dict):
            da3_gs = output.get("gaussians")
        if da3_gs is None:
            raise RuntimeError("DA3 forward returned no Gaussians")

        gaussians = da3_to_c3g(da3_gs)

        # Only patch_tokens passed to DiffHead; cls_token omitted (semantic, not geometric;
        # camera geometry already handled by encoder input poses + ProPE in DiffHead).
        if visualization_dump is not None:
            pat = getattr(output, "patch_tokens", None)
            if pat is not None:
                visualization_dump["blurry_patch_tokens"] = pat    # [B, V, N, C]

        del output
        torch.cuda.empty_cache()
        return gaussians

    def get_data_shim(self):
        return []

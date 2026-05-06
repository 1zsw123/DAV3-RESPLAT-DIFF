"""
Conservative border mask for training loss to suppress mis-projection at frame
edges (DA3 pose / FOV mismatch between predicted-pose render and real GT image).

Activated via env var BORDER_MASK_ENABLE=1. Defaults: top 4%, bottom 10%
(bottom worse due to FOV bias), left/right 3%.
"""
import os
from functools import lru_cache

import torch
from torch import Tensor


def _enabled() -> bool:
    return os.environ.get("BORDER_MASK_ENABLE", "0") == "1"


def _pcts() -> tuple[float, float, float, float]:
    return (
        float(os.environ.get("BORDER_TOP_PCT", "0.04")),
        float(os.environ.get("BORDER_BOTTOM_PCT", "0.10")),
        float(os.environ.get("BORDER_LEFT_PCT", "0.03")),
        float(os.environ.get("BORDER_RIGHT_PCT", "0.03")),
    )


def crop_box(H: int, W: int) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) integer pixel indices for hard crop."""
    if not _enabled():
        return 0, H, 0, W
    t, b, l, r = _pcts()
    return int(t * H), H - int(b * H), int(l * W), W - int(r * W)


@lru_cache(maxsize=8)
def _soft_mask_cached(H: int, W: int, top: float, bottom: float, left: float, right: float, device_str: str, dtype_str: str) -> Tensor:
    device = torch.device(device_str)
    dtype = getattr(torch, dtype_str)
    y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, 1, H, 1)
    x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype).view(1, 1, 1, W)

    # Each side ramps from 0 at the edge to 1 at (top|bottom|left|right) percent
    # of the way in. clamp((coord - edge_pct) / edge_pct, 0, 1) gives a linear
    # ramp; mult of x and y produces 2D soft border mask.
    eps = 1e-6
    my = torch.clamp((y - top) / max(top, eps), 0.0, 1.0) * torch.clamp(((1.0 - bottom) - y) / max(bottom, eps), 0.0, 1.0)
    mx = torch.clamp((x - left) / max(left, eps), 0.0, 1.0) * torch.clamp(((1.0 - right) - x) / max(right, eps), 0.0, 1.0)
    return (my * mx).contiguous()


def soft_mask(H: int, W: int, device: torch.device, dtype: torch.dtype) -> Tensor | None:
    """Return [1, 1, H, W] soft mask, or None if disabled."""
    if not _enabled():
        return None
    t, b, l, r = _pcts()
    return _soft_mask_cached(H, W, t, b, l, r, str(device), str(dtype).split(".")[-1])

"""
sharpness_detector.py

Two sharpness detectors:

1. LaplacianSharpness  — zero-parameter, fast, always available.
   Laplacian variance: higher = sharper.

2. IQASharpness        — uses ARNIQA-CSIQ + nima-koniq from IQA-PyTorch.
   Validated as the best 2 of 32 blur metrics on multi-dataset benchmark.
   Both metrics: higher score = better quality (less blurry).
   Scores are min-max normalised across the candidate set and averaged.

Usage:
    det = IQASharpness(device)
    scores = det.score_frames(images)   # [V] float, higher = sharper
    weights = det.weights(images)       # [B, V] softmax-normalised × V
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 1. Laplacian baseline (no learnable params, no external deps)
# ─────────────────────────────────────────────────────────────────────────────

class LaplacianSharpness(nn.Module):
    _KERNEL = torch.tensor(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32
    )
    _GRAY = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """images [B,V,3,H,W] → weights [B,V] (mean=1.0)"""
        B, V, _, H, W = images.shape
        gray = (images * self._GRAY.to(images).view(1, 1, 3, 1, 1)).sum(2)
        k = self._KERNEL.to(images).view(1, 1, 3, 3)
        lap = F.conv2d(gray.view(B * V, 1, H, W), k, padding=1)
        scores = lap.view(B * V, -1).var(dim=1).view(B, V)
        return F.softmax(scores, dim=1) * V

    @torch.no_grad()
    def scores_only(self, images: Tensor) -> Tensor:
        """images [B,V,3,H,W] → raw scores [B,V]"""
        B, V, _, H, W = images.shape
        gray = (images * self._GRAY.to(images).view(1, 1, 3, 1, 1)).sum(2)
        k = self._KERNEL.to(images).view(1, 1, 3, 3)
        lap = F.conv2d(gray.view(B * V, 1, H, W), k, padding=1)
        return lap.view(B * V, -1).var(dim=1).view(B, V)


# keep old name as alias
SharpnessDetector = LaplacianSharpness


# ─────────────────────────────────────────────────────────────────────────────
# 2. IQA-based sharpness (ARNIQA-CSIQ + nima-koniq)
# ─────────────────────────────────────────────────────────────────────────────

class IQASharpness(nn.Module):
    """
    No-reference IQA sharpness scorer using:
      - arniqa-csiq  (AUC=0.877 on blur detection benchmark)
      - nima-koniq   (AUC=0.845)

    Both metrics: higher score = better perceptual quality (less blurry).
    Final score = mean of min-max normalised arniqa + nima scores.

    Parameters
    ----------
    device       : torch device
    iqa_root     : path to IQA-PyTorch checkout (default: auto-detected from C3G tree)
    crop_center  : if True, score on a centre-crop instead of full image
                   (more robust to background clutter in blur detection)
    crop_frac    : fraction of H/W to keep for centre crop (default 0.7)
    """

    def __init__(
        self,
        device: torch.device,
        iqa_root: str | None = None,
        crop_center: bool = True,
        crop_frac: float = 0.7,
    ):
        super().__init__()

        # Locate IQA-PyTorch
        if iqa_root is None:
            iqa_root = str(Path(__file__).resolve().parents[2] / "IQA-PyTorch")
        if iqa_root not in sys.path:
            sys.path.insert(0, iqa_root)

        import pyiqa
        self._arniqa = pyiqa.create_metric("arniqa-csiq", device=device, as_loss=False)
        self._nima   = pyiqa.create_metric("nima-koniq",  device=device, as_loss=False)
        self._arniqa.eval()
        self._nima.eval()

        self.crop_center = crop_center
        self.crop_frac   = crop_frac
        self.device      = device

    @torch.no_grad()
    def score_frames(self, images: Tensor) -> Tensor:
        """
        images : [V, 3, H, W] or [B, V, 3, H, W]  float32 in [0,1]
        returns: scores [V] or [B, V], higher = sharper
        """
        squeeze = images.dim() == 4
        if squeeze:
            images = images.unsqueeze(0)   # [1, V, 3, H, W]

        B, V, C, H, W = images.shape

        if self.crop_center:
            ch, cw = int(H * self.crop_frac), int(W * self.crop_frac)
            y0 = (H - ch) // 2
            x0 = (W - cw) // 2
            imgs = images[:, :, :, y0:y0+ch, x0:x0+cw]
        else:
            imgs = images

        # Process each frame individually (IQA models expect [1, 3, H, W])
        flat = imgs.view(B * V, C, imgs.shape[-2], imgs.shape[-1])
        s_arniqa = torch.stack([self._arniqa(flat[i:i+1]) for i in range(B * V)]).view(B, V)
        s_nima   = torch.stack([self._nima  (flat[i:i+1]) for i in range(B * V)]).view(B, V)

        # Min-max normalise each metric across views, then average
        def _norm(x: Tensor) -> Tensor:
            mn, mx = x.min(dim=1, keepdim=True).values, x.max(dim=1, keepdim=True).values
            denom = (mx - mn).clamp(min=1e-6)
            return (x - mn) / denom

        scores = (_norm(s_arniqa) + _norm(s_nima)) / 2.0   # [B, V] in [0, 1]

        return scores.squeeze(0) if squeeze else scores

    @torch.no_grad()
    def scores_only(self, images: Tensor) -> Tensor:
        """Alias matching LaplacianSharpness interface. [B,V,3,H,W] → [B,V]"""
        return self.score_frames(images)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """[B,V,3,H,W] → softmax weights [B,V] with mean=1.0"""
        scores = self.score_frames(images)   # [B, V] in [0,1]
        return F.softmax(scores, dim=-1) * scores.shape[-1]

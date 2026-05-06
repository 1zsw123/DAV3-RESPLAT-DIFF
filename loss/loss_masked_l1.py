"""
Masked L1 loss: L1 photometric loss weighted by the Laplacian sharp mask.

Sharp mask comes from batch["target"]["sharp_mask"] [B, V, 1, H, W] (0/1 float).
Pixels in the sharp region (mask=1) are weighted 1.0; blurry/artifact regions
(mask=0) are weighted by fallback_weight (default 0.1, not zero so gradient
still flows through all pixels at low weight).

Falls back to unweighted L1 when no mask is present.
"""

from dataclasses import dataclass, field

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMaskedL1Cfg:
    weight: float = 1.0
    fallback_weight: float = 0.1   # weight for mask=0 pixels


@dataclass
class LossMaskedL1CfgWrapper:
    masked_l1: LossMaskedL1Cfg


class LossMaskedL1(Loss[LossMaskedL1Cfg, LossMaskedL1CfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image: Float[Tensor, "batch view 3 height width"] | None = None,
    ) -> Float[Tensor, ""]:
        pred  = prediction.color   # [B, V, 3, H, W]

        # Graph-connected zero so DDP gradient allreduce stays in lockstep.
        if target_image is None:
            return (pred * 0).sum()

        delta = (pred - target_image).abs()  # [B, V, 3, H, W]

        mask = batch["target"].get("sharp_mask", None)  # type: ignore[attr-defined]

        if mask is None:
            return self.cfg.weight * delta.mean()

        # mask: [B, V, 1, H, W]  values 0 or 1
        mask = mask.to(pred.device, dtype=pred.dtype)

        # Build pixel weights: sharp=1.0, blurry=fallback_weight
        w = self.cfg.fallback_weight + (1.0 - self.cfg.fallback_weight) * mask  # [B,V,1,H,W]
        loss = (delta * w).sum() / (w.sum() * 3 + 1e-8)

        return self.cfg.weight * loss

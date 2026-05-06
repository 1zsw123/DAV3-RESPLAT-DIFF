from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .border_mask import soft_mask as _border_soft_mask
from .loss import Loss


@dataclass
class LossMseL1Cfg:
    weight: float
    empty_render_thresh: float = 0.02   # max(RGB) below this = empty GS region


@dataclass
class LossMseL1CfgWrapper:
    mse_l1: LossMseL1Cfg


class LossMseL1(Loss[LossMseL1Cfg, LossMseL1CfgWrapper]):
    """
    MSE-style photometric loss using L1 norm, with two validity masks:
      1. DSLR anon mask  : batch["target"]["mask"] [B,V,1,H,W], 1=valid
      2. Empty-render mask: pixels where GS rendered nothing (all-black background)
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image: Float[Tensor, "batch view 3 height width"] | None = None,
    ) -> Float[Tensor, ""]:
        pred  = prediction.color                         # [B, V, 3, H, W]
        # Graph-connected zero so DDP gradient allreduce stays in lockstep.
        if target_image is None:
            return (pred * 0).sum()
        # L2 form (squared) — keep the MseL1 class (mask logic) but switch the
        # per-pixel cost from L1 (.abs()) to L2 ((.)**2) per user request.
        delta = (pred - target_image) ** 2                 # [B, V, 3, H, W]

        # 1. DSLR anon mask (1 = valid pixel, 0 = anonymized/border)
        dslr_mask = batch["target"].get("mask", None)   # type: ignore[attr-defined]
        if dslr_mask is not None:
            valid = (dslr_mask.to(pred.device) > 0.5).expand_as(pred)
        else:
            valid = torch.ones_like(pred, dtype=torch.bool)

        # 2. Empty-render mask: GS has no coverage → all channels near background (0,0,0)
        render_hit = pred.max(dim=2, keepdim=True).values > self.cfg.empty_render_thresh
        valid = valid & render_hit.expand_as(pred)

        # Use weighted-mean form (not boolean-indexing + early-return) so the
        # graph stays connected to model params on EVERY rank. An empty-valid
        # rank that returned a graph-detached 0.0 used to skip DDP gradient
        # allreduce while other ranks did it — caused NCCL ALLREDUCE timeout
        # (3 ranks at NumelIn=1 vs 1 rank at NumelIn=trainable params).
        valid_f = valid.to(delta.dtype)

        # Optional soft border mask: down-weight edges where DA3 pose mismatch
        # forces model to fit content outside its frustum (env BORDER_MASK_ENABLE=1).
        H, W = delta.shape[-2:]
        bm = _border_soft_mask(H, W, delta.device, delta.dtype)
        if bm is not None:
            valid_f = valid_f * bm.unsqueeze(0)  # [1,1,1,H,W] broadcast over [B,V,3,H,W]

        denom = valid_f.sum().clamp_min(1.0)
        return self.cfg.weight * (delta * valid_f).sum() / denom

"""
iPhone LiDAR depth supervision at context (iPhone) views.

batch["context"]["depth"]  [B, V_ctx, H, W]  uint16 PNG → float32 metres (0=invalid)
output.depth[:, V_tgt:]    [B, V_ctx, H, W]  rendered GS depth at context poses

Requires share_weight > 0 so that context views are rendered by the decoder.

Scale-invariant L2: normalise each view's valid depths by their mean before
computing the loss, matching the approach in LossDepth for target views.
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
class LossContextDepthCfg:
    weight: float = 0.1


@dataclass
class LossContextDepthCfgWrapper:
    context_depth: LossContextDepthCfg


class LossContextDepth(Loss[LossContextDepthCfg, LossContextDepthCfgWrapper]):
    """
    Scale-invariant depth loss at context (iPhone) views using LiDAR GT.

    output.depth layout when share_weight > 0:
      [:, :V_tgt]  → target views  (used by LossDepth)
      [:, V_tgt:]  → context views (used here)
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image=None,
    ) -> Float[Tensor, ""]:
        # Graph-connected zero (use prediction.color which always has DiffusionHead
        # grad path) so DDP gradient allreduce stays in lockstep across ranks even
        # when depth supervision is unavailable.
        _zero = (prediction.color * 0).sum()

        depth_gt = batch["context"].get("depth", None)   # type: ignore[attr-defined]
        if depth_gt is None or prediction.depth is None:
            return _zero

        V_tgt = batch["target"]["image"].shape[1]
        depth_pred = prediction.depth[:, V_tgt:]   # context views [B, V_ctx, H, W]

        if depth_pred.shape[1] == 0:
            return _zero

        depth_gt = depth_gt.to(depth_pred.device, dtype=depth_pred.dtype)

        # Resize GT to match rendered depth resolution if needed
        if depth_gt.shape[-2:] != depth_pred.shape[-2:]:
            B, V = depth_gt.shape[:2]
            depth_gt = F.interpolate(
                depth_gt.view(B * V, 1, *depth_gt.shape[-2:]),
                size=depth_pred.shape[-2:], mode="nearest"
            ).view(B, V, *depth_pred.shape[-2:])

        valid = (depth_gt > 0) & (depth_pred > 0)
        if not valid.any():
            return _zero

        B, V, H, W = depth_gt.shape
        loss = torch.tensor(0.0, device=depth_pred.device)
        n = 0

        for b in range(B):
            for v in range(V):
                m = valid[b, v]
                if not m.any():
                    continue
                gt_v   = depth_gt[b, v][m]
                pred_v = depth_pred[b, v][m]
                gt_n   = gt_v   / (gt_v.mean()   + 1e-8)
                pred_n = pred_v / (pred_v.mean()  + 1e-8)
                loss  += ((pred_n - gt_n) ** 2).mean()
                n += 1

        if n == 0:
            return _zero

        return self.cfg.weight * loss / n

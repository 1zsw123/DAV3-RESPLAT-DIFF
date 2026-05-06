"""
GT depth supervision loss.

L_depth = || D_render - D_gt ||_2

D_gt comes from projecting the Replica mesh onto each frame using GT camera
poses (pre-computed offline by scripts/generate_depth_maps.py).
The dataset loader puts it in batch["target"]["depth"] as a [B, V, H, W]
float32 tensor (metres; 0 = invalid pixel).

If GT depth is not yet available (depth maps still being generated) the loss
silently returns 0 so training is not blocked.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDepthCfg:
    weight: float = 0.1


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    """
    L_depth = weight * || D_render - D_gt ||_2  (over valid GT pixels)

    Rendered depth: prediction.depth[:, :V_tgt]   [B, V, H, W]
    GT depth:       batch["target"]["depth"]        [B, V, H, W], 0=invalid
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

        depth_gt = batch["target"].get("depth", None)   # type: ignore[attr-defined]
        depth_pred = prediction.depth                    # [B, V_all, H, W] or None

        if depth_gt is None or depth_pred is None:
            return _zero

        V_tgt = batch["target"]["image"].shape[1]
        depth_pred = depth_pred[:, :V_tgt]              # target views only [B, V, H, W]

        # Move GT to same device/dtype as prediction
        depth_gt = depth_gt.to(depth_pred.device, dtype=depth_pred.dtype)

        # Valid mask: GT > 0 (invalid = 0) and pred > 0
        valid = (depth_gt > 0) & (depth_pred > 0)
        if not valid.any():
            return _zero

        # Scale-invariant: normalise each image by its GT mean depth so the
        # loss is not dominated by scenes with different absolute scales.
        B, V, H, W = depth_gt.shape
        loss = torch.tensor(0.0, device=depth_pred.device)
        n_valid_imgs = 0

        for b in range(B):
            for v in range(V):
                m = valid[b, v]
                if not m.any():
                    continue
                gt_v   = depth_gt[b, v][m]
                pred_v = depth_pred[b, v][m]
                # Normalise each side by its own mean so the loss measures
                # structural depth similarity regardless of absolute scale.
                # This handles the scale_invariant renderer (depth in 1/near
                # units) vs GT depth in metres.
                gt_norm   = gt_v   / (gt_v.mean()   + 1e-8)
                pred_norm = pred_v / (pred_v.mean()  + 1e-8)
                loss  += ((pred_norm - gt_norm) ** 2).mean()
                n_valid_imgs += 1

        if n_valid_imgs == 0:
            return _zero

        return self.cfg.weight * loss / n_valid_imgs

"""LPIPS loss with per-image exposure compensation and valid-region masking for DSLR targets."""
from dataclasses import dataclass

import torch
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from .loss_dslr import _fit_exposure, _apply_exposure


@dataclass
class LossLpipsDslrCfg:
    weight: float
    apply_after_step: int
    exp_comp_blur_sigma: float = 0.05


@dataclass
class LossLpipsDslrCfgWrapper:
    lpips_dslr: LossLpipsDslrCfg


class LossLpipsDslr(Loss[LossLpipsDslrCfg, LossLpipsDslrCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsDslrCfgWrapper) -> None:
        super().__init__(cfg)
        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image=None,
    ) -> Float[Tensor, ""]:
        pred = prediction.color   # [B, V, 3, H, W]
        # Graph-connected zero so DDP gradient allreduce stays in lockstep.
        if global_step < self.cfg.apply_after_step:
            return (pred * 0).sum()
        gt = target_image         # [B, V, 3, H, W]
        masks = batch["target"].get("mask", None)  # [B, V, 1, H, W] or None
        B, V, C, H, W = pred.shape

        pred_comp_list = []
        gt_list = []
        mask_list = []

        for b in range(B):
            for v in range(V):
                r = pred[b, v]   # [3, H, W]
                g = gt[b, v]     # [3, H, W]
                m = masks[b, v] if masks is not None else torch.ones(1, H, W, device=r.device)

                E = _fit_exposure(r, g, m, self.cfg.exp_comp_blur_sigma)
                r_comp = _apply_exposure(r, E)

                pred_comp_list.append(r_comp)
                gt_list.append(g)
                mask_list.append(m.expand(3, H, W))

        pred_comp = torch.stack(pred_comp_list)   # [B*V, 3, H, W]
        gt_stack = torch.stack(gt_list)           # [B*V, 3, H, W]
        mask_stack = torch.stack(mask_list)       # [B*V, 3, H, W]

        pred_masked = pred_comp * mask_stack
        gt_masked = gt_stack * mask_stack

        loss = self.lpips.forward(pred_masked, gt_masked, normalize=True)
        return self.cfg.weight * loss.mean()

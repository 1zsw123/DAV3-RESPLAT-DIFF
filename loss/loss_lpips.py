from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .border_mask import crop_box as _border_crop_box
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image = None,
    ) -> Float[Tensor, ""]:
        image = target_image
        pred = prediction.color           # [B, V, 3, H, W]

        # Before the specified step, don't apply the loss. Return graph-connected
        # zero so DDP gradient allreduce stays in lockstep across ranks.
        if global_step < self.cfg.apply_after_step:
            return (pred * 0).sum()

        # Optional dynamic-region mask: zero out dynamic pixels in BOTH pred and gt
        # so LPIPS contributions from those pixels become zero.
        mask = batch["target"].get("mask", None)  # type: ignore[attr-defined]
        if mask is not None:
            m = mask.to(pred.device, dtype=pred.dtype)  # [B, V, 1, H, W]
            pred = pred * m
            image = image * m

        pred_bv = rearrange(pred, "b v c h w -> (b v) c h w")
        image_bv = rearrange(image, "b v c h w -> (b v) c h w")

        # Optional hard center crop (BORDER_MASK_ENABLE=1). Hard crop because
        # LPIPS uses VGG conv features whose receptive fields make masked-zero
        # boundaries leak artificial activations into the loss.
        H, W = pred_bv.shape[-2:]
        t, b, l, r = _border_crop_box(H, W)
        if (t, b, l, r) != (0, H, 0, W):
            pred_bv = pred_bv[..., t:b, l:r]
            image_bv = image_bv[..., t:b, l:r]

        loss = self.lpips.forward(pred_bv, image_bv, normalize=True)
        return self.cfg.weight * loss.mean()

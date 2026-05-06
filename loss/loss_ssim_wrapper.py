from dataclasses import dataclass

from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .border_mask import crop_box as _border_crop_box
from .loss import Loss
from .loss_ssim import ssim


@dataclass
class LossSsimCfg:
    weight: float


@dataclass
class LossSsimCfgWrapper:
    ssim: LossSsimCfg


class LossSsim(Loss[LossSsimCfg, LossSsimCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image=None,
    ) -> Float[Tensor, ""]:
        pred_full = prediction.color    # [B, V, 3, H, W]
        gt_full = target_image          # [B, V, 3, H, W]

        # Optional dynamic-region mask: zero out dynamic pixels in BOTH pred and gt
        mask = batch["target"].get("mask", None)  # type: ignore[attr-defined]
        if mask is not None:
            m = mask.to(pred_full.device, dtype=pred_full.dtype)  # [B, V, 1, H, W]
            pred_full = pred_full * m
            gt_full = gt_full * m

        pred = rearrange(pred_full, "b v c h w -> (b v) c h w")
        gt   = rearrange(gt_full,   "b v c h w -> (b v) c h w")

        # Optional hard center crop to skip frustum-mismatch edges (BORDER_MASK_ENABLE=1).
        # Hard crop (not multiply) because SSIM uses windowed statistics and a
        # masked-zero region distorts the window means/variances at the boundary.
        H, W = pred.shape[-2:]
        t, b, l, r = _border_crop_box(H, W)
        if (t, b, l, r) != (0, H, 0, W):
            pred = pred[..., t:b, l:r]
            gt = gt[..., t:b, l:r]

        ssim_val, _, _, _ = ssim(pred, gt, data_range=1.0)
        return self.cfg.weight * (1.0 - ssim_val)

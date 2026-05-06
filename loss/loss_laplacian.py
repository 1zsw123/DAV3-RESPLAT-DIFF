from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

_LAP_KERNEL = torch.tensor(
    [[0.,  1., 0.],
     [1., -4., 1.],
     [0.,  1., 0.]]
).view(1, 1, 3, 3)


@dataclass
class LossLaplacianCfg:
    weight: float = 0.05


@dataclass
class LossLaplacianCfgWrapper:
    laplacian: LossLaplacianCfg


class LossLaplacian(Loss[LossLaplacianCfg, LossLaplacianCfgWrapper]):
    """
    Sharpness loss: penalises blurry predictions by rewarding large Laplacian magnitude.
    Loss = -mean(|Lap(pred)|), so blurry output (small |Lap|) → large loss.
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image: Float[Tensor, "batch 3 height width"] | None = None,
    ) -> Float[Tensor, ""]:
        pred = prediction.color  # [B, C, H, W] or [B, V, C, H, W]
        if pred.dim() == 5:
            B, V, C, H, W = pred.shape
            pred = pred.view(B * V, C, H, W)

        C = pred.shape[1]
        kernel = _LAP_KERNEL.to(pred.device, pred.dtype).expand(C, 1, 3, 3)
        lap = F.conv2d(pred, kernel, padding=1, groups=C)
        return -self.cfg.weight * lap.abs().mean()

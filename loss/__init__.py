from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_lpips_dslr import LossLpipsDslr, LossLpipsDslrCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_mae import LossMseL1, LossMseL1CfgWrapper
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_context_depth import LossContextDepth, LossContextDepthCfgWrapper
from .loss_ssim_wrapper import LossSsim, LossSsimCfgWrapper
from .loss_masked_l1 import LossMaskedL1, LossMaskedL1CfgWrapper
from .loss_dslr import LossDslr, LossDslrCfgWrapper
from .loss_laplacian import LossLaplacian, LossLaplacianCfg, LossLaplacianCfgWrapper

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossLpipsDslrCfgWrapper: LossLpipsDslr,
    LossMseCfgWrapper: LossMse,
    LossMseL1CfgWrapper: LossMseL1,
    LossDepthCfgWrapper: LossDepth,
    LossContextDepthCfgWrapper: LossContextDepth,
    LossSsimCfgWrapper: LossSsim,
    LossMaskedL1CfgWrapper: LossMaskedL1,
    LossDslrCfgWrapper: LossDslr,
    LossLaplacianCfgWrapper: LossLaplacian,
}

LossCfgWrapper = LossLpipsCfgWrapper | LossLpipsDslrCfgWrapper | LossMseCfgWrapper | LossMseL1CfgWrapper | LossDepthCfgWrapper | LossContextDepthCfgWrapper | LossSsimCfgWrapper | LossMaskedL1CfgWrapper | LossDslrCfgWrapper | LossLaplacianCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]

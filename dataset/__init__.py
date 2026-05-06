from dataclasses import fields

from torch.utils.data import Dataset

from .dataset_scannet_pose import DatasetScannetPose, DatasetScannetPoseCfgWrapper
from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg, DatasetRE10kCfgWrapper, DatasetDL3DVCfgWrapper, \
    DatasetScannetppCfgWrapper
from .dataset_scannet import ScannetCfg, DatasetScannet, DatasetScannetCfgWrapper
from .dataset_replica import DatasetReplica, ReplicaCfg, DatasetReplicaCfgWrapper
from .dataset_blur_replica import DatasetBlurReplica, BlurReplicaCfg, DatasetBlurReplicaCfgWrapper
from .dataset_colmap import DatasetColmap, ColmapCfg, DatasetColmapCfgWrapper
from .dataset_scannetpp_blur import DatasetScannetppBlur, ScannetppBlurCfg, DatasetScannetppBlurCfgWrapper
from .dataset_scannetpp_gs import DatasetScannetppGs, ScannetppGsCfg, DatasetScannetppGsCfgWrapper
from .dataset_scannetpp_dslr import DatasetScannetppDslr, ScannetppDslrCfg, DatasetScannetppDslrCfgWrapper
from .dataset_rsblur import DatasetRSBlur, RSBlurCfg, DatasetRSBlurCfgWrapper
from .dataset_gopro_defocus import DatasetGoProDefocus, GoProDefocusCfg, DatasetGoProDefocusCfgWrapper
from .dataset_tum import DatasetTum, TumCfg, DatasetTumCfgWrapper
from .dataset_scannet_i2slam import DatasetScannetI2Slam, ScannetI2SlamCfg, DatasetScannetI2SlamCfgWrapper
from .dataset_deblurnerf_blur import DatasetDeblurNeRFBlur, DeblurNeRFBlurCfg, DatasetDeblurNeRFBlurCfgWrapper
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "dl3dv": DatasetRE10k,
    "scannetpp": DatasetRE10k,
    "scannet_pose": DatasetScannetPose,
    "scannet": DatasetScannet,
    "replica": DatasetReplica,
    "blur_replica": DatasetBlurReplica,
    "colmap": DatasetColmap,
    "scannetpp_blur": DatasetScannetppBlur,
    "scannetpp_gs": DatasetScannetppGs,
    "scannetpp_dslr": DatasetScannetppDslr,
    "rsblur": DatasetRSBlur,
    "gopro_defocus": DatasetGoProDefocus,
    "tum": DatasetTum,
    "scannet_i2slam": DatasetScannetI2Slam,
    "deblurnerf_blur_finetune": DatasetDeblurNeRFBlur,
}


DatasetCfgWrapper = DatasetRE10kCfgWrapper | DatasetDL3DVCfgWrapper | DatasetScannetppCfgWrapper | DatasetScannetPoseCfgWrapper | DatasetScannetCfgWrapper | DatasetReplicaCfgWrapper | DatasetBlurReplicaCfgWrapper | DatasetColmapCfgWrapper | DatasetScannetppBlurCfgWrapper | DatasetScannetppGsCfgWrapper | DatasetScannetppDslrCfgWrapper | DatasetRSBlurCfgWrapper | DatasetGoProDefocusCfgWrapper | DatasetTumCfgWrapper | DatasetScannetI2SlamCfgWrapper | DatasetDeblurNeRFBlurCfgWrapper
DatasetCfg = DatasetRE10kCfg | ScannetCfg | ReplicaCfg | BlurReplicaCfg | ColmapCfg | ScannetppBlurCfg | ScannetppGsCfg | ScannetppDslrCfg | RSBlurCfg | GoProDefocusCfg | TumCfg | ScannetI2SlamCfg | DeblurNeRFBlurCfg


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
) -> list[Dataset]:
    datasets = []
    for cfg in cfgs:
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)

        view_sampler = get_view_sampler(
            cfg.view_sampler,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        datasets.append(dataset)

    return datasets

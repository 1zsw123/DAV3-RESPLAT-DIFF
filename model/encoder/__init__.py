from typing import Optional

from .encoder import Encoder
from .encoder_noposplat import EncoderNoPoSplatCfg, EncoderNoPoSplat
from .encoder_noposplat_multi import EncoderNoPoSplatMulti
from .encoder_vggt import EncoderVGGT, EncoderVGGTCfg
from .encoder_da3gs import EncoderDA3GS, EncoderDA3GSCfg
from .encoder_da3_resplat import EncoderDA3ReSplat, EncoderDA3ReSplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "noposplat": (EncoderNoPoSplat, None),
    "noposplat_multi": (EncoderNoPoSplatMulti, None),
    "vggt": (EncoderVGGT, None),
    "da3gs": (EncoderDA3GS, None),
    "da3_resplat": (EncoderDA3ReSplat, None),
}

EncoderCfg = EncoderNoPoSplatCfg | EncoderVGGTCfg | EncoderDA3GSCfg | EncoderDA3ReSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer

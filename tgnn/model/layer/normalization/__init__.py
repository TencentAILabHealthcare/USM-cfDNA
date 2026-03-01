# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch.nn as nn

from .layer_norm import LayerNorm
from .rms_norm import RMSNorm

NORM_LAYERS = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.LocalResponseNorm,
    LayerNorm,
    RMSNorm
]

try:
    from apex.normalization import FusedLayerNorm, MixedFusedRMSNorm, FusedRMSNorm, MixedFusedLayerNorm

    NORM_LAYERS.extend([
        FusedLayerNorm,
        MixedFusedRMSNorm,
        FusedRMSNorm,
        MixedFusedLayerNorm
    ])
except ImportError:
    pass

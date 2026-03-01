# Copyright (c) 2024, Tencent Inc. All rights reserved.
from typing import Union, Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgnn.utils.registry import Registry

ACTIVATION = Registry('activation')

if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SiLU(nn.Module):
    """Export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


@ACTIVATION.register()
class GELU(nn.Module):
    __constants__ = ['approximate']
    approximate: str

    def __init__(self, approximate: str = 'none') -> None:
        super().__init__()
        assert approximate in ['none', 'tanh', 'sigmoid', 'erf']
        self.approximate = approximate

    def forward(self, x):
        if self.approximate == 'sigmoid':
            return torch.sigmoid(1.702 * x) * x
        elif self.approximate == 'erf':
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        else:
            return F.gelu(x, approximate=self.approximate)

    def extra_repr(self) -> str:
        return 'approximate={}'.format(self.approximate)


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def get_activation(name: Union[str, nn.Module, Callable[..., nn.Module]] = "ReLU",
                   inplace=True, leak=0.1, **kwargs):
    if isinstance(name, nn.Module):
        return name

    elif isinstance(name, str):
        if name in ACTIVATION:
            module = ACTIVATION[name](**kwargs)
        elif name in ["SiLU", "ReLU", "SELU"]:
            module = getattr(nn, name)(inplace=inplace)
        elif name in ["LeakyReLU"]:
            module = getattr(nn, name)(leak, inplace=inplace)
        else:
            raise AttributeError("Unsupported act type: {}".format(name))
        return module

    return name(inplace=inplace)

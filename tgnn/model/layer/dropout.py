# Copyright (c) 2024, Tencent Inc. All rights reserved.
from typing import Union, List

import torch
import torch.nn as nn


def drop_path(x,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True,
              batch_dim: int = 0):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x

    if type(batch_dim) == int:
        batch_dim = [batch_dim]

    keep_prob = 1 - drop_prob
    shape = [1, ] * x.ndim
    for i in range(len(batch_dim)):  # work with diff dim tensors, not just 2D ConvNets
        shape[i] = x.size(i)

    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample
    """

    def __init__(self, p=0., scale_by_keep: bool = True, batch_dim: Union[int, List[int]] = 0):
        super(DropPath, self).__init__()
        if type(batch_dim) == int:
            batch_dim = [batch_dim]

        self.drop_prob = p
        self.batch_dim = batch_dim
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob,
                         training=self.training,
                         scale_by_keep=self.scale_by_keep,
                         batch_dim=self.batch_dim)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class DropoutAxiswise(nn.Dropout):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.
    """

    def __init__(self,
                 p: float,
                 inplace: bool = False,
                 batch_dim: Union[int, List[int]] = -2):
        super(DropoutAxiswise, self).__init__(p=p, inplace=inplace)
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = super().forward(x.new_ones(shape))
        return x * mask


class DropoutRowwise(DropoutAxiswise):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    def __init__(self, p: float, inplace: bool = False):
        super(DropoutRowwise, self).__init__(p=p, inplace=inplace, batch_dim=-3)


class DropoutColumnwise(DropoutAxiswise):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    def __init__(self, p: float, inplace: bool = False):
        super(DropoutColumnwise, self).__init__(p=p, inplace=inplace, batch_dim=-2)

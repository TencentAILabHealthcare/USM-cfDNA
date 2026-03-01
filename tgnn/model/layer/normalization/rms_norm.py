# Copyright (c) 2024, Tencent Inc. All rights reserved.
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from tgnn.utils import warn_rank_0

try:
    from apex.normalization.fused_layer_norm import mixed_dtype_fused_rms_norm_affine
    import importlib

    importlib.import_module("fused_layer_norm_cuda")
    warn_rank_0("using apex fused rms norm")
except ImportError:
    mixed_dtype_fused_rms_norm_affine = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    Ref：
        1. Root Mean Square Layer Normalization
    """

    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-6,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(*normalized_shape, **factory_kwargs))

    def _norm(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps)

    def _forward_impl(self, input):
        if hasattr(F, "rms_norm"):
            return F.rms_norm(input, self.normalized_shape, self.weight, eps=self.eps)

        if mixed_dtype_fused_rms_norm_affine is None or torch.jit.is_tracing() or torch.jit.is_scripting() or not input.is_cuda:
            input = self._norm(input).type_as(input)
            return self.weight * input
        else:
            return mixed_dtype_fused_rms_norm_affine(input, self.weight, self.normalized_shape, self.eps)

    def forward(self, input, mask=None, inplace=False):
        """
        Args:
            input: [*, dim]
            mask: [*,]

        Returns:
            [*, dim]
        """
        if mask is None:
            return self._forward_impl(input)

        output = input if inplace else input.clone()
        output[mask] = self._forward_impl(input[mask])
        return output

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}"

# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgnn.model.layer import Linear, GELU
from tgnn.utils import warn_rank_0

try:
    from xformers.ops import swiglu, unbind

    warn_rank_0(f"using xformer swiglu extention")
except:
    swiglu = None


class SwiGLU(nn.Module):
    """swiglu feed forward network

    Args:
        dim: input embedding dim
        hidden_dim: inner hidden dim, also named ffn_dim in other project
        multiple_of: emsure hidden dim are divided
        ffn_dim_multiplier: config param in llama2, default none for compact with llama
        bias: linear layer bias
        pack_weights: pack fc linear and than split, set true for faster training

    Note that MLP is also called swiglu operator in some papers, you call speed up by installing xformers
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: int = None,
            multiple_of: int = 256,
            ffn_dim_multiplier: Optional[float] = None,
            dropout=0.0,
            pack_weights=True,
            bias=False,
            xformer=True
    ):
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_dim or 4 * dim
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = int(2 * hidden_dim / 3)
        self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.pack_fc = pack_weights
        if self.pack_fc:
            self.c_fc = Linear(self.dim, 2 * self.hidden_dim, bias=bias)
        else:
            self.c_fc1 = Linear(self.dim, self.hidden_dim, bias=bias)
            self.c_fc2 = Linear(self.dim, self.hidden_dim, bias=bias)
        self.xformer = xformer
        self.c_proj = Linear(self.hidden_dim, self.dim, bias=bias, init="final")
        self.dropout = dropout

    def _native_impl(self, x):
        if self.pack_fc:
            x1, x2 = torch.chunk(self.c_fc(x), 2, dim=-1)
            x = F.silu(x1) * x2
        else:
            x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x

    def _swiglu_impl(self, x):
        if self.pack_fc:
            fcw = self.c_fc.weight
            fc1w, fc2w = unbind(
                fcw.view([2, fcw.shape[0] // 2, fcw.shape[1]]),
                dim=0
            )
            fcb = self.c_fc.bias
            if fcb is not None:
                fc1b, fc2b = unbind(fcb.view([2, fcb.shape[0] // 2]), dim=0)
            else:
                fc1b, fc2b = None, None
            x = swiglu(x,
                       fc1w, fc1b,
                       fc2w, fc2b,
                       self.c_proj.weight, self.c_proj.bias)
        else:
            x = swiglu(x,
                       self.c_fc1.weight, self.c_fc1.bias,
                       self.c_fc2.weight, self.c_fc2.bias,
                       self.c_proj.weight, self.c_proj.bias)
        return x

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, dim], hidden state
        """
        if self.xformer and swiglu is not None and x.numel() > 0:
            x = self._swiglu_impl(x)
        else:
            x = self._native_impl(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                inplace: bool = False) -> torch.Tensor:
        if mask is None:
            return self._forward_impl(x)

        assert mask.shape == x.shape[:-1], f"mask shape({mask.shape}) not match input({{x.shape}})"
        out = x if inplace else x.clone()
        out[mask] = self._forward_impl(x[mask])
        return out


class MLP(nn.Module):
    """feed forward network, invert bottleneck
    """

    def __init__(self,
                 dim: int,
                 hidden_dim: Optional[int] = None,
                 out_dim: Optional[int] = None,
                 multiple_of: int = 256,
                 ffn_dim_multiplier: Optional[float] = None,
                 bias: bool = False,
                 dropout: float = 0.0,
                 activation: str = "gelu",
                 approximate="tanh",
                 xformer=True):
        super().__init__()
        self.activation = activation
        assert self.activation in ["gelu", "swiglu", "squared_relu", "relu"]

        hidden_dim = hidden_dim or 4 * dim
        is_swiglu = self.activation == "swiglu"
        if is_swiglu:
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)

            hidden_dim = int(2 * hidden_dim / 3)
            self.hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        else:
            self.hidden_dim = hidden_dim

        self.dim = dim
        self.out_dim = out_dim or self.dim
        self.c_fc = Linear(self.dim,
                           self.hidden_dim * 2 if is_swiglu else self.hidden_dim,
                           bias=bias)
        self.xformer = xformer
        if self.activation == "gelu":
            self.approximate = approximate
            self.act_fn = GELU(self.approximate)
        elif self.activation == "squared_relu":
            self.act_fn = lambda x: nn.functional.relu(x) ** 2
        elif self.activation == "relu":
            self.act_fn = nn.ReLU()

        self.c_proj = Linear(self.hidden_dim, self.out_dim, bias=bias, init="final")
        self.dropout = dropout

    def _swiglu_impl(self, x):
        if self.xformer:
            x1, x2 = torch.chunk(self.c_fc(x), 2, dim=-1)
            x = F.silu(x1) * x2
            x = self.c_proj(x)
        else:
            fcw = self.c_fc.weight
            fc1w, fc2w = unbind(
                fcw.view([2, fcw.shape[0] // 2, fcw.shape[1]]),
                dim=0
            )
            fcb = self.c_fc.bias
            if fcb is not None:
                fc1b, fc2b = unbind(fcb.view([2, fcb.shape[0] // 2]), dim=0)
            else:
                fc1b, fc2b = None, None

            x = swiglu(x, fc1w, fc1b, fc2w, fc2b, self.c_proj.weight, self.c_proj.bias)

        return x

    def _common_impl(self, x):
        x = self.c_fc(x)
        x = self.act_fn(x)
        x = self.c_proj(x)
        return x

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "swiglu":
            x = self._swiglu_impl(x)
        else:
            x = self._common_impl(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                inplace: bool = False) -> torch.Tensor:
        if mask is None:
            return self._forward_impl(x)

        assert mask.shape == x.shape[:-1], f"mask shape({mask.shape}) not match input({{x.shape}})"
        out = x if inplace else x.clone()
        out[mask] = self._forward_impl(x[mask])

        return out
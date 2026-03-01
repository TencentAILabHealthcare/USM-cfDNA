# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sdpa_attention import sliding_window_mask


class CoreAttention(nn.Module):

    def __init__(self,
                 window_size=None,
                 attn_logit_softcapping: float = 0.0):
        super(CoreAttention, self).__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        if self.window_size is not None:
            assert len(self.window_size) == 2 and self.window_size[0] > 1 and self.window_size[
                1] > 1, f"unvalid window: {self.window_size}"
        self.attn_logit_softcapping = attn_logit_softcapping

    def forward(self,
                query, key, value,
                *,
                sink: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None,
                scale: Optional[float] = None,
                dropout_p: float = 0.0,
                is_causal: bool = False,
                return_attn_probs: bool = False):
        if is_causal:
            assert attn_mask is None and attn_bias is None, (f"causal attention don't support custom attention "
                                                             f"mask and bias")

        if attn_mask is not None:
            attn_mask = attn_mask.bool()

        if attn_bias is not None:
            if isinstance(attn_bias, (list, tuple)):
                attn_bias = sum(attn_bias)

            if attn_mask is not None:
                attn_bias.masked_fill_(~attn_mask, torch.finfo(attn_bias.dtype).min)
            attn_mask = attn_bias.to(query.dtype)

        if scale is None:
            head_dim = query.size(-1)
            scale = math.sqrt(head_dim)

        query = query.transpose(-2, -3)  # [*, num_heads, q_len, head_dim]
        key = key.transpose(-2, -3)  # [*, num_heads, k_len, head_dim]
        value = value.transpose(-2, -3)  # [*, num_heads, k_len, head_dim]
        attn_weights = (query @ key.transpose(-2, -1)) / scale  # [*, num_heads, q_len, k_len]

        if self.attn_logit_softcapping > 0:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        seq_len = query.size(-2)
        if is_causal:
            attn_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device))
            attn_mask = attn_mask.expand_as(attn_weights)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(~attn_mask, torch.finfo(attn_weights.dtype).min)
            else:
                attn_weights += attn_mask

        if self.window_size is not None and seq_len > min(self.window_size):
            sw_mask = sliding_window_mask(seq_len, self.window_size, device=query.device)
            sw_mask = sw_mask.view([1, ] * (len(attn_mask.shape) - 2) + [seq_len, seq_len])
            attn_weights.masked_fill_(~sw_mask, torch.finfo(attn_weights.dtype).min)

        if sink is None:
            attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(query)  # [*, num_heads, q_len, k_len]
        else:
            attn_weights = torch.cat([attn_weights, sink], dim=-1)
            attn_weights = F.softmax(attn_weights, dim=-1).type_as(query)

        scores = F.dropout(attn_weights, p=dropout_p, training=self.training)
        y = scores @ value  # [*, num_heads, seq_len, head_dim]
        y = y.transpose(-2, -3)

        if not return_attn_probs:
            attn_weights = None

        return y, attn_weights

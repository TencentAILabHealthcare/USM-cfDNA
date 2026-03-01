# Copyright (c) 2024, Tencent Inc. All rights reserved.
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgnn.utils.env import get_torch_version


@lru_cache(maxsize=2)
def sliding_window_mask(seq_len, window, device=None):
    band = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    band = torch.triu(band, diagonal=-window[0])
    band = band * torch.tril(band, diagonal=window[1])
    return band


class SdpaAttention(nn.Module):
    """native scaled dot product attention"""

    def __init__(self, window_size=None):
        super(SdpaAttention, self).__init__()
        if window_size is not None:
            if isinstance(window_size, int):
                window_size = (window_size, window_size)
            else:
                window_size = tuple(window_size)
        self.window_size = window_size
        if self.window_size is not None:
            assert len(self.window_size) == 2 and self.window_size[0] > 1 and self.window_size[
                1] > 1, f"unvalid window: {self.window_size}"

    def _scaled_dot_product_attention(self,
                                      query, key, value,
                                      attn_mask: Optional[torch.Tensor] = None,
                                      scale: Optional[float] = None,
                                      is_causal: bool = False,
                                      dropout_p: Optional[float] = 0.0):
        """
        Args:
            q, k, v: tensor[*, q_len or kv_len, num_heads, head_dim]
            attn_mask: [*, num_heads, q_len, kv_len], attention bool mask or float bias

        Returns:
            out: [*, seq_len, num_heads, head_dim]
        """

        query = query.transpose(-2, -3)  # [*, num_heads, seq_len, dim]
        key = key.transpose(-2, -3)
        value = value.transpose(-2, -3)

        kwargs = {
            "is_causal": is_causal,
            "dropout_p": dropout_p if self.training else 0.0
        }

        if get_torch_version() >= (2, 1):
            kwargs["scale"] = scale
        else:
            assert scale is None, (f"torch version: {get_torch_version()} not supported scale kwargs input"
                                   f"you must set scale=None")

        y = F.scaled_dot_product_attention(query, key, value,
                                           attn_mask=attn_mask,
                                           **kwargs)  # [*, num_heads, seq_len, head_dim]
        y = y.transpose(-2, -3)

        return y

    def forward(self,
                query, key, value,
                attn_mask: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None,
                scale: Optional[float] = None,
                dropout_p: float = 0.0,
                is_causal: bool = False,
                return_attn_probs: bool = False):
        """
        Args:
            query, key, value: [*, q_len or kv_len, num_heads, head_dim]
            attn_mask: [*, num_heads, q_len, kv_len], attention bool mask
            attn_bias: [*, num_heads, q_len, kv_len], attention float bias

        Returns:
            out: [*, seq_len, num_heads, head_dim]
            attn_probs: [*, seq_len, num_heads, head_dim], current not supported
        """
        if is_causal:
            assert attn_mask is None and attn_bias is None, (f"causal attention don't support custom attention "
                                                             f"mask and bias")

        if attn_mask is not None:
            attn_mask = attn_mask.bool()

        assert not return_attn_probs, f"native sdpa not supported return attention probs"
        if attn_bias is not None:
            if isinstance(attn_bias, (list, tuple)):
                attn_bias = sum(attn_bias)

            if attn_mask is not None:
                attn_bias.masked_fill_(~attn_mask, torch.finfo(attn_bias.dtype).min)
            attn_mask = attn_bias.to(query.dtype)

        seq_len = query.shape[-3]
        if self.window_size is not None and seq_len > min(self.window_size):
            sw_mask = sliding_window_mask(seq_len, self.window_size, device=query.device)
            if attn_mask is not None:
                sw_mask = sw_mask.view([1, ] * (len(attn_mask.shape) - 2) + [seq_len, seq_len])
                if attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask * sw_mask
                else:
                    attn_mask.masked_fill_(~sw_mask, torch.finfo(attn_mask.dtype).min)
            else:
                attn_mask = sw_mask

        y = self._scaled_dot_product_attention(query, key, value,
                                               attn_mask=attn_mask,
                                               dropout_p=dropout_p,
                                               is_causal=is_causal,
                                               scale=scale)

        return y, None

# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
from functools import partialmethod, partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tgnn.config import get_config
from tgnn.distributed import comm
from tgnn.model.layer import Linear
from tgnn.model.layer.embedding import precompute_freqs_cis, apply_rotary_emb
from tgnn.model.utils import chunk_layer
from tgnn.utils.tensor import flatten_final_dims
from .core_attention import CoreAttention
from .distributed_attention import DistributedAttention
from .flash_attention import FlashSelfAttention
from .sdpa_attention import SdpaAttention

KVCache = Tuple[torch.Tensor, torch.Tensor]


def repeat_kv(x: torch.Tensor, repeats: int, dim=-2) -> torch.Tensor:
    if repeats == 1:
        return x

    return torch.repeat_interleave(x, dim=dim, repeats=repeats)


class MultiHeadAttention(nn.Module):
    """sequence or msa mutlihead attention

    Args:
        query_pre_attn_scalar: inverse square root of this value instead of head_dim
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 *,
                 num_kv_heads: Optional[int] = None,
                 bias: bool = False,
                 dropout: float = 0.0,
                 pack_qkv: bool = True,
                 query_pre_attn_scalar: Optional[int] = None,
                 attn_logit_softcapping: float = 0.0,
                 sliding_window_size: Optional[Union[int, Tuple[int, int]]] = None,
                 gating=False,
                 attention_mode=None):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, f"number of heads must devide dim"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.bias = bias
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.repeats = self.dim // self.kv_dim
        self.attn_dropout = dropout
        self.resid_dropout = dropout
        self.pack_qkv = pack_qkv
        if self.num_heads * self.head_dim != self.dim:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.dim}"
                f" and `num_heads`: {num_heads})."
            )
        if self.pack_qkv:
            self.c_attn = Linear(self.dim, self.dim + 2 * self.kv_dim, bias=self.bias)
        else:
            self.q_proj = Linear(self.dim, self.dim, bias=self.bias)
            self.k_proj = Linear(self.dim, self.kv_dim, bias=self.bias)
            self.v_proj = Linear(self.dim, self.kv_dim, bias=self.bias)

        self.c_proj = Linear(self.dim, self.dim, bias=self.bias, init="final")
        self.gating = gating
        if self.gating:
            self.g_proj = Linear(self.dim, self.dim, bias=self.bias, init="gating")

        self.scaling = query_pre_attn_scalar ** -0.5 if query_pre_attn_scalar is not None else None
        self.sliding_window_size = sliding_window_size
        self.attention_mode = get_config().model.attention_mode.lower() if attention_mode is None else attention_mode
        assert self.attention_mode in ("none", "eager",
                                       "native", "sdpa",
                                       "v1", "v2", "flash_attn", "flash",
                                       "ds_evo", "evo", "evo_attn")
        if self.attention_mode in ("v1", "v2", "flash_attn", "flash"):
            self.attn_fn = FlashSelfAttention(window_size=self.sliding_window_size,
                                              attn_logit_softcapping=attn_logit_softcapping)
        elif self.attention_mode in ("native", "sdpa"):
            self.attn_fn = SdpaAttention(window_size=self.sliding_window_size)
        else:
            self.attn_fn = CoreAttention(window_size=self.sliding_window_size,
                                         attn_logit_softcapping=attn_logit_softcapping)
        self.enable_ds_sequence_parallel = comm.get_sequence_parallel_world_size() > 1
        if self.enable_ds_sequence_parallel:
            self.dist_attn = DistributedAttention(comm.get_sequence_parallel_group())

    def _project_qkv(self, x: torch.Tensor, freqs_cis: torch.Tensor = None):
        """
        Args:
            x: [*, seq_len, self.dim + 2 * self.kv_dim], qkv packed tensor
            freqs_cis: [seq_len, head_dim // 2, 2] or [seq_len, head_dim // 2, 4]

        Returns:
            q, k, v: tensor[*, seq_len, num_heads, head_dim]
        """
        if self.pack_qkv:
            q, k, v = self.c_attn(x).split([self.dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = q.view(q.shape[:-1] + (-1, self.head_dim))  # [*, seq_len, num_kv_heads, head_dim]
        k = k.view(k.shape[:-1] + (-1, self.head_dim))  # [*, seq_len, num_q_head, head_dim]
        v = v.view(v.shape[:-1] + (-1, self.head_dim))  # [*, seq_len, num_kv_heads, head_dim]
        if freqs_cis is not None:
            if freqs_cis.size(-1) == 2:
                freqs_cis_q, freqs_cis_k = freqs_cis, freqs_cis
            else:
                assert freqs_cis.size(-1) == 4, f"expect size 4 in last dimension, get {freqs_cis.size(-1)}"
                freqs_cis_q, freqs_cis_k = freqs_cis[..., :2], freqs_cis[..., 2:]

            q = apply_rotary_emb(q, freqs_cis_q)  # [*, seq_len, num_heads, head_dim]
            k = apply_rotary_emb(k, freqs_cis_k)  # [*, seq_len, num_kv_heads, head_dim]

        k = repeat_kv(k, self.repeats, dim=-2)  # [*, seq_len, num_heads, head_dim]
        v = repeat_kv(v, self.repeats, dim=-2)

        return q, k, v

    def _scaled_dot_product_attention(self,
                                      q, k, v,
                                      attn_mask=None,
                                      attn_bias=None,
                                      is_causal=False,
                                      dropout_p=0.0,
                                      scaling=None,
                                      return_attn_weight=False):
        """
        Args:
            q, k, v: tensor[*, q_len or kv_len, num_heads, head_dim]
            attn_mask: [*, num_heads, q_len, kv_len]
            attn_bias: [*, num_heads, q_len, kv_len] or list of bias

        Returns:
            out: [*, seq_len, num_heads, head_dim]
            attn_weights: [*, num_heads, seq_len, seq_len]
        """
        y, attn_weights = self.attn_fn(q, k, v,
                                       attn_mask=attn_mask,
                                       attn_bias=attn_bias,
                                       dropout_p=dropout_p,
                                       is_causal=is_causal,
                                       scale=scaling,
                                       return_attn_probs=return_attn_weight)
        return y, attn_weights

    def forward(self,
                x: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None,
                is_causal: bool = False,
                return_attn_weight: bool = False):
        """
        Args:
            x: [*, seq_len, dim]
            freqs_cis: [seq_len, head_dim // 2, 2]
            attn_mask: [*, 1, seq_len, seq_len]
            attn_bias: [*, 1, seq_len, seq_len], tensor of list of attention biases
            is_causal: [seq_len, seq_len]
            return_attn_weight: whether to return the attention weights

        Returns:
            y: [*, seq_len, dim], ouptut hiddens
            attn_weight: [*, num_heads, seq_len, seq_len]
        """
        q, k, v = self._project_qkv(x, freqs_cis=freqs_cis)

        if self.enable_ds_sequence_parallel:
            y, attn_weight = self.dist_attn(self._scaled_dot_product_attention, q, k, v,
                                            is_causal=is_causal,
                                            attn_mask=attn_mask,
                                            attn_bias=attn_bias,
                                            scaling=self.scaling,
                                            return_attn_weight=return_attn_weight,
                                            dropout_p=self.attn_dropout if self.training else 0.0)
        else:
            y, attn_weight = self._scaled_dot_product_attention(q, k, v,
                                                                is_causal=is_causal,
                                                                attn_mask=attn_mask,
                                                                attn_bias=attn_bias,
                                                                scaling=self.scaling,
                                                                return_attn_weight=return_attn_weight,
                                                                dropout_p=self.attn_dropout if self.training else 0.0)

        # [*, seq_len, dim]
        y = flatten_final_dims(y, 2)
        if self.gating:
            y = y * self.g_proj(x).sigmoid()

        y = F.dropout(self.c_proj(y), p=self.resid_dropout, training=self.training)

        return y, attn_weight


class GatedMultiHeadAttention(MultiHeadAttention):
    __init__ = partialmethod(MultiHeadAttention.__init__, gating=True)


class MSARowAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 num_kv_heads=None,
                 sliding_window_size=None,
                 bias=False,
                 dropout=0.0,
                 pack_qkv=True,
                 gating=False,
                 attention_mode=None,
                 is_causal: bool = False):
        super(MSARowAttention, self).__init__()
        self.mha = MultiHeadAttention(dim,
                                      num_heads,
                                      num_kv_heads=num_kv_heads,
                                      sliding_window_size=sliding_window_size,
                                      bias=bias,
                                      dropout=dropout,
                                      pack_qkv=pack_qkv,
                                      gating=gating,
                                      attention_mode=attention_mode)
        self.num_heads = num_heads
        self.head_dim = self.mha.head_dim
        self.attention_mode = get_config().model.attention_mode.lower() if attention_mode is None else attention_mode
        self.is_flash_attn = self.attention_mode in ("flash", "flash_attn")
        self.is_causal = is_causal
        self.freqs_cis = None

    def update_freqs_cis(self, seq_len, dtype=None, device=None):
        if self.freqs_cis is None or seq_len > self.freqs_cis.shape[0]:
            self.freqs_cis = precompute_freqs_cis(seq_len,
                                                  rotary_dim=self.head_dim,
                                                  dtype=dtype,
                                                  device=device)

    @torch.jit.ignore
    def _chunk(self,
               m: torch.Tensor,
               attn_mask: Optional[torch.Tensor],
               is_causal: bool,
               return_attn_weight: bool,
               chunk_size: int,
               ) -> torch.Tensor:

        def fn(m, biases):
            return super().forward(m, biases=biases)

        inputs = {"m": m}
        if attn_mask is not None:
            inputs["attn_mask"] = attn_mask
            fn = partial(fn,
                         return_attn_weight=return_attn_weight,
                         is_causal=is_causal,
                         freqs_cis=self.freqs_cis)

        return chunk_layer(
            fn,
            inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(m.shape[:-2])
        )

    def _forward_imp(self,
                     m: torch.Tensor,
                     attn_mask: Optional[torch.Tensor] = None,
                     is_causal: bool = False,
                     return_attn_weight: bool = False,
                     chunk_size: Optional[int] = None):
        """
        Args:
            m: [bs, seq_len, dim]
            attn_mask: [bs, num_heads, seq_len, seq_len], or [bs, seq_len]

        Returns:
            out: [bs, seq_len, dim]
            attn_weight: [bs, num_heads, seq_len, seq_len]
        """
        self.update_freqs_cis(seq_len=m.shape[-2], dtype=m.dtype, device=m.device)
        if chunk_size is not None:
            return self._chunk(
                m,
                attn_mask,
                return_attn_weight,
                chunk_size
            )

        return self.mha(m,
                        freqs_cis=self.freqs_cis,
                        attn_mask=attn_mask,
                        is_causal=is_causal,
                        return_attn_weight=return_attn_weight)

    def forward(self,
                m: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attn_weight: bool = False,
                chunk_size: Optional[int] = None,
                inplace=False):
        """
        Args:
            m: [bs, row, col, dim]
            mask: [bs, row, col]

        Returns:
            out: [bs, row, col, dim]
            attn_weight: [bs, row, num_heads, col, col]
        """
        if mask is None:
            mx = m.reshape(-1, *m.shape[-2:])
            row_mask = None
            attn_mask = None
        else:
            mask = mask.bool()
            row_mask = mask.sum(dim=-1) > 0
            mx = m[row_mask]
            attn_mask = mask[row_mask]
            if not self.is_flash_attn:
                attn_mask = attn_mask[..., None, None, :]

        y, attn_weights = self._forward_imp(mx,
                                            attn_mask=attn_mask,
                                            return_attn_weight=return_attn_weight,
                                            is_causal=self.is_causal,
                                            chunk_size=chunk_size)
        aw = None
        bs, row = m.shape[:-2]
        if mask is not None:
            out = m if inplace else m.clone()
            out[row_mask] = y
            if return_attn_weight:
                aw = attn_weights.new_zeros(bs, row, *attn_weights.shape[-3:])
                aw[row_mask] = attn_weights
        else:
            out = y.reshape(*m.shape)
            if return_attn_weight:
                aw = attn_weights.reshape(bs, row, *attn_weights.shape[-3:])

        return out, aw


class MSAColumnAttention(MSARowAttention):

    def forward(self,
                m: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attn_weight: bool = False,
                inplace: bool = False,
                chunk_size: Optional[int] = None):
        """
        Args:
            m: [*, row, col, dim]
            mask: [*, row, col]

        Returns:
            out: [*, row, col, dim]
            attn_weights: [*, col, num_heads, row, row]
        """
        m = m.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        m, attn_weights = super().forward(m,
                                          mask=mask,
                                          return_attn_weight=return_attn_weight,
                                          inplace=inplace,
                                          chunk_size=chunk_size)
        m = m.transpose(-2, -3)

        return m, attn_weights


class CausalMultiheadAttention(MultiHeadAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 max_len=1024,
                 num_kv_heads=None,
                 bias=False,
                 dropout=0.0,
                 **kwargs):
        super().__init__(dim, num_heads,
                         num_kv_heads=num_kv_heads,
                         bias=bias,
                         dropout=dropout,
                         **kwargs)

        self.kv_cache = None
        self.max_len = max_len

    def update_kv_cache(self, k, v, start_pos, dim=1):
        # [*, seq_len, num_heads, head_dim]
        if self.kv_cache is None:
            bs = k.shape[0]
            dtype = k.dtype
            device = k.device
            cache_shape = (bs, self.max_len, self.num_heads, self.head_dim)
            self.kv_cache = (
                torch.zeros(cache_shape, device=device, dtype=dtype),
                torch.zeros(cache_shape, device=device, dtype=dtype)
            )
        cache_k, cache_v = self.kv_cache
        if start_pos[-1] >= self.max_len:
            start_pos = torch.tensor(self.max_len - 1, device=start_pos.device)
            # shift 1 position to the left
            cache_k = torch.roll(cache_k, shifts=-1, dims=dim)
            cache_v = torch.roll(cache_v, shifts=-1, dims=dim)

        k = cache_k.index_copy(dim, start_pos, k)
        v = cache_v.index_copy(dim, start_pos, v)
        self.kv_cache = (k, v)

        return k, v

    def forward(self,
                x: torch.Tensor,
                freqs_cis: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                start_pos: Optional[torch.Tensor] = None,
                return_attn_weight: bool = False):
        q, k, v = self._project_qkv(x, freqs_cis=freqs_cis)
        if start_pos is not None:
            k, v = self.update_kv_cache(k, v, start_pos)

        y, attn_weight = self._scaled_dot_product_attention(q, k, v,
                                                            attn_mask=attn_mask,
                                                            is_causal=attn_mask is None,
                                                            return_attn_weight=return_attn_weight,
                                                            dropout_p=self.attn_dropout)
        # [*, seq_len, dim]
        y = flatten_final_dims(y, 2)
        y = F.dropout(self.c_proj(y), p=self.resid_dropout, training=self.training)

        return y, attn_weight

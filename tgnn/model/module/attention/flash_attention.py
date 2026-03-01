# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import importlib
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from packaging import version

from tgnn.utils import warn_rank_0

try:
    import flash_attn
    from flash_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func,
        flash_attn_with_kvcache
    )
    from flash_attn.bert_padding import index_first_axis, unpad_input, pad_input

    warn_rank_0(f"using flash attention: {flash_attn.__version__}")
except ImportError:
    flash_attn = None
    flash_attn_func = None
    flash_attn_qkvpacked_func = None
    flash_attn_kvpacked_func = None
    flash_attn_with_kvcache = None

try:
    from flash_attn.flash_attn_triton import (
        flash_attn_func as flash_attn_triton_func,
        flash_attn_qkvpacked_func as flash_attn_triton_qkvpacked_func,
        flash_attn_kvpacked_func as flash_attn_triton_kv_packed_func
    )
except ImportError:
    flash_attn_triton_func = None
    flash_attn_triton_qkvpacked_func = None
    flash_attn_triton_kv_packed_func = None


def is_flash_attn_greater_or_equal(library_version: str):
    if flash_attn is None:
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(library_version)


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    """

    def __init__(self,
                 window_size=None,
                 attn_logit_softcapping: float = 0.0):
        super(FlashSelfAttention, self).__init__()
        assert flash_attn_func is not None, f"please install flash attention first"
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        self.window_size = window_size if window_size is not None else (-1, -1)
        self.attn_logit_softcapping = attn_logit_softcapping
        self.is_attn_logit_softcapping_enabled = is_flash_attn_greater_or_equal("2.6.2")
        if not self.is_attn_logit_softcapping_enabled:
            assert self.attn_logit_softcapping == 0.0, f"current version {flash_attn.__version__} not support"

    def inference(self, q, k_cache, v_cache, k=None, v=None):
        return flash_attn_with_kvcache(q, k_cache, v_cache, k, v)

    def forward(self,
                q,
                k=None,
                v=None,
                scale=None,
                attn_mask: torch.Tensor = None,
                attn_bias: torch.Tensor = None,
                is_causal: bool = False,
                dropout_p: float = 0.0,
                return_attn_probs: bool = False):
        """Implements the multi-head softmax attention. support mult-query and grouped-query attention

        Args:
            q, k, v: [*, q/k_len, num_heads, dim], containing the query, key, and value.
            attn_mask: [*, num_heads, q_len, k_len] or [*, seq_len]
            attn_bias: [*, num_heads, q_len, k_len], one or list of attention bias

        Returns:
            out: [*, seq_len, num_heads, dim]
            attn_probs: [*, num_heads, q_len, k_len]
        """
        if attn_mask is not None or attn_bias is not None:
            assert (flash_attn_triton_func is not None and
                    flash_attn_triton_qkvpacked_func is not None and
                    flash_attn_triton_kv_packed_func is not None)
            assert not return_attn_probs, f"flash attention with bias not support return probabilities"

        if attn_mask is not None:
            attn_mask = attn_mask.bool()

        if attn_bias is not None:
            if isinstance(attn_bias, (list, tuple)):
                attn_bias = sum(attn_bias)

            if attn_mask is not None:
                assert attn_mask.size() == attn_bias.size(), f"attn mask shape {attn_mask.shape} != attn bias shape {attn_bias.size()}"
                attn_bias.masked_fill_(~attn_mask, torch.finfo(attn_bias.dtype).min)
            attn_mask = attn_bias.to(q.dtype)
            assert self.window_size == (-1, -1), f"flash attention with bias not support sliding window"

        batch_dims = q.shape[:-3]
        if len(q.shape) > 4:
            if q is not None:
                q = q.reshape(-1, *q.shape[-3:])

            if k is not None:
                k = k.reshape(-1, *k.shape[-3:])

            if v is not None:
                v = v.reshape(-1, *v.shape[-3:])

            if attn_mask is not None:
                if q.ndim - attn_mask.ndim == 2:
                    # attn_mask is seq_mask for varlen forward
                    attn_mask = attn_mask.reshape(-1, attn_mask.shape[-1])
                else:
                    attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                out = self.fa_varlen_forward(q, k, v,
                                             attn_mask=attn_mask,
                                             scale=scale,
                                             is_causal=is_causal,
                                             dropout_p=dropout_p,
                                             return_attn_probs=return_attn_probs)
            else:
                out = self.fa_with_bias_forward(q, k, v, attn_mask,
                                                scale=scale,
                                                is_causal=is_causal,
                                                dropout_p=dropout_p)
        else:
            out = self.fa_forward(q, k, v,
                                  scale=scale,
                                  is_causal=is_causal,
                                  dropout_p=dropout_p,
                                  return_attn_probs=return_attn_probs)

        attn_probs = None
        if return_attn_probs:
            out, _, attn_probs = out

        if len(batch_dims) > 1:
            out = out.reshape(*batch_dims, *out.shape[-3:])
            if attn_bias is not None:
                attn_probs = attn_probs.reshape(*batch_dims, *attn_probs.shape[-3:])

        return out, attn_probs

    def fa_forward(self,
                   q, k, v,
                   scale=None,
                   is_causal=False,
                   dropout_p=0.0,
                   return_attn_probs=False):

        factory_kwargs = {
            "causal": is_causal,
            "softmax_scale": scale,
            "window_size": self.window_size,
            "return_attn_probs": return_attn_probs,
            "dropout_p": dropout_p if self.training else 0.0
        }

        if self.is_attn_logit_softcapping_enabled:
            factory_kwargs["softcap"] = self.attn_logit_softcapping

        qkv_packed = k is None and v is None
        kv_packed = k is not None and v is None
        if qkv_packed:
            out = flash_attn_qkvpacked_func(q, **factory_kwargs)
        elif kv_packed:
            out = flash_attn_kvpacked_func(q, k, **factory_kwargs)
        else:
            q_head_dim = q.shape[-1]
            v_head_dim = v.shape[-1]
            # in MLA, q head_dim maybe not equal to v head dim
            if q_head_dim != v_head_dim:
                v = F.pad(v, [0, q_head_dim - v_head_dim])

            out = flash_attn_func(q, k, v, **factory_kwargs)
            if q_head_dim != v_head_dim:
                if return_attn_probs:
                    y, *other = out
                    y = y[..., :v_head_dim]
                    out = (y, *other)
                else:
                    out = out[..., :v_head_dim]
        return out

    def fa_with_bias_forward(self,
                             q, k, v,
                             attn_bias,
                             scale=None,
                             is_causal=False,
                             dropout_p=0.0,
                             return_attn_probs=False):
        qkv_packed = k is None and v is None
        kv_packed = k is not None and v is None
        assert dropout_p == 0, f"flash attention with bias not support dropout({dropout_p}) now."
        factory_kwargs = {
            "causal": is_causal,
            "softmax_scale": scale
        }
        if qkv_packed:
            out = flash_attn_triton_qkvpacked_func(q, attn_bias, **factory_kwargs)
        elif kv_packed:
            out = flash_attn_triton_kv_packed_func(q, attn_bias, **factory_kwargs)
        else:
            q_head_dim = q.shape[-1]
            v_head_dim = v.shape[-1]
            # in MLA, q head_dim maybe not equal to v head dim
            if q_head_dim != v_head_dim:
                v = F.pad(v, [0, q_head_dim - v_head_dim])
            out = flash_attn_triton_func(q, k, v, attn_bias, **factory_kwargs)
            if q_head_dim != v_head_dim:
                if return_attn_probs:
                    y, *other = out
                    y = y[..., :v_head_dim]
                    out = (y, *other)
                else:
                    out = out[..., :v_head_dim]

        return out

    def fa_varlen_forward(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          attn_mask: torch.Tensor,
                          scale: Optional[float] = None,
                          is_causal: bool = False,
                          dropout_p: float = 0.0,
                          return_attn_probs: bool = False):
        batch_size, query_length = q.shape[:2]
        q, k, v, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
            q, k, v, attn_mask, query_length
        )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        out = self._varlen_forward_imp(q, k, v,
                                       cu_seqlens_q=cu_seqlens_q,
                                       cu_seqlens_k=cu_seqlens_k,
                                       max_seqlen_q=max_seqlen_in_batch_q,
                                       max_seqlen_k=max_seqlen_in_batch_k,
                                       dropout_p=dropout_p,
                                       scale=scale,
                                       is_causal=is_causal,
                                       return_attn_probs=return_attn_probs
                                       )

        if return_attn_probs:
            attn_output_unpad, *other = out
            attn_out = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
            out = (attn_out, *other)
        else:
            out = pad_input(out, indices_q, batch_size, query_length)

        return out

    def _varlen_forward_imp(self,
                            q: torch.Tensor,
                            k: torch.Tensor,
                            v: torch.Tensor,
                            cu_seqlens_q: torch.Tensor,
                            max_seqlen_q: int,
                            cu_seqlens_k: torch.Tensor = None,
                            max_seqlen_k: int = None,
                            scale=None,
                            is_causal=False,
                            dropout_p=0.0,
                            return_attn_probs=False):
        """
        Args:
            qkv: [total_nnz, 3, num_heads, head_dim]
            cu_seqlens: [bs + 1, ], the cumulative sequence lengths of the sequences in the batch
            max_seqlen: maximum sequence length in the batch

        Returns:
            out: [total, num_heads, head_dim]
        """
        factory_kwargs = {
            "window_size": self.window_size,
            "softmax_scale": scale,
            "causal": is_causal,
            "dropout_p": dropout_p,
            "return_attn_probs": return_attn_probs
        }
        if self.is_attn_logit_softcapping_enabled:
            factory_kwargs["softcap"] = self.attn_logit_softcapping

        if k is None and v is None:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(q,
                                                              cu_seqlens_q,
                                                              cu_seqlens_k,
                                                              **factory_kwargs)
        elif k is not None and v is None:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q,
                                                             k,
                                                             cu_seqlens_q=cu_seqlens_q,
                                                             cu_seqlens_k=cu_seqlens_k,
                                                             max_seqlen_q=max_seqlen_q,
                                                             max_seqlen_k=max_seqlen_k,
                                                             **factory_kwargs)
        else:
            out = flash_attn.flash_attn_varlen_func(q, k, v,
                                                    cu_seqlens_q=cu_seqlens_q,
                                                    cu_seqlens_k=cu_seqlens_k,
                                                    max_seqlen_q=max_seqlen_q,
                                                    max_seqlen_k=max_seqlen_k,
                                                    **factory_kwargs)

        return out


def _get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask: [bs, seq_len], Boolean or int tensor,  1 means valid and 0 means not valid.

    Return:
        indices: The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch: Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.

    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer: Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer: Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer: Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask: Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length: Target length.

    Return:
        query_layer: Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer: Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer: Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q: The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )

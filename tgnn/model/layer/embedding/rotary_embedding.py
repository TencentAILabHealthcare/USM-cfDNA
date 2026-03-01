# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
from typing import Optional, Tuple

import torch


class ApplyRotaryEmb(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        """
        Args:
            x: [batch_size, seqlen, nheads, headdim]
            cos, sin: [seqlen, rotary_dim / 2]
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        import rotary_emb
        batch, seqlen, nheads, head_dim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= head_dim, f"rotary_dim {rotary_dim} hea_dim {head_dim}"
        assert seqlen <= rotary_seqlen, f"input seq_len {seqlen} rotary seq_len {rotary_seqlen}"
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x_ro = x[..., :rotary_dim]
        x1, x2 = x_ro.chunk(2, dim=-1) if not interleaved else (x_ro[..., ::2], x_ro[..., 1::2])
        out = torch.empty_like(x) if not inplace else x
        out_ro = out[..., :rotary_dim]
        if inplace:
            o1, o2 = x1, x2
        else:
            o1, o2 = (
                out_ro.chunk(2, dim=-1)
                if not interleaved
                else (out_ro[..., ::2], out_ro[..., 1::2])
            )
        rotary_emb.apply_rotary(
            x1,
            x2,
            cos[:seqlen, None],
            sin[:seqlen, None],
            o1,
            o2,
            False,
        )
        if not inplace and rotary_dim < head_dim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])

        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        import rotary_emb
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        do1, do2 = (
            do_ro.chunk(2, dim=-1) if not ctx.interleaved else (do_ro[..., ::2], do_ro[..., 1::2])
        )
        dx = torch.empty_like(do) if not inplace else do
        if inplace:
            dx1, dx2 = do1, do2
        else:
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (
                dx_ro.chunk(2, dim=-1)
                if not ctx.interleaved
                else (dx_ro[..., ::2], dx_ro[..., 1::2])
            )
        rotary_emb.apply_rotary(
            do1,
            do2,
            cos[:seqlen, None],
            sin[:seqlen, None],
            dx1,
            dx2,
            True,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None


def precompute_freqs_cis(seq_len: int,
                         rotary_dim: int,
                         theta: float = 10000.0,
                         interpolation_factor: Optional[float] = None,
                         dtype: Optional[torch.dtype] = torch.float32,
                         scale_base: Optional[int] = None,
                         device: Optional[torch.device] = None,
                         complex=False) -> torch.Tensor:
    """The rotary position embeddings from RoFormer_ (Su et. al).

    Ref:
        1. https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py

    Args:
        seq_len: max sequence length
        rotary_dim: head dim, is also rotary dim
        theta: Scaling factor for frequency computation. Defaults to 10000
        interpolation_factor: rotary embedding extended with linear scaling
        scale_base: if scale_base is not None, a recommended value for scale_base is 512

    Returns:
        freqs_cis: [seq_len, head_dim // 2, 2], or [seq_len, head_dim // 2, 4] when scale_base is None
    """
    # compute inv freq
    freqs = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=torch.float32, device=device)
    if interpolation_factor is not None:
        seq_idx *= 1 / interpolation_factor

    freqs = torch.outer(seq_idx, freqs)
    # TODO: if pytorch2.x compile support complex64, delete it
    if complex:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    else:
        freqs_cis = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    if scale_base is not None:
        assert not complex, f"scale_base is not supported complex dtype"
        scale = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
        scale = (scale + 0.4 * rotary_dim) / (1.4 * rotary_dim)
        power = (torch.arange(seq_len, dtype=torch.float32, device=device) - seq_len // 2) / scale_base
        scale = scale ** power[:, None, None]
        freqs_cis = freqs_cis * scale
        freqs_cis_k = freqs_cis / scale
        return torch.cat([freqs_cis, freqs_cis_k], dim=-1)

    if complex:
        low_precison_dtypes = (torch.float16, torch.bfloat16, torch.int8)
        dtype = (
            torch.complex32 if dtype in low_precison_dtypes else torch.complex64
        )
    return freqs_cis.to(dtype)


def apply_rotary_emb(
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        interleaved: bool = True,
        inplace: bool = False,
        fused_rotary_embedding: bool = True
) -> torch.Tensor:
    """
    Refs:
        1. https://github.com/huggingface/transformers/issues/25199

    Args:
        x: [*, seq_len, num_heads, head_dim], xq or xk
        freqs_cis: [seq_len, head_dim // 2, 2], last 2 is cos and sin value

    Returns:
        rotary embedding: [*, seq_len, num_heads, head_dim]
    """
    seq_len = x.shape[-3]
    freqs_cis = freqs_cis[:seq_len]
    if freqs_cis.dtype in (torch.complex32, torch.complex64):
        # cast because `view_as_complex` does not support bfloat16 tensors
        # force convert x to complex64
        if interleaved:
            xc = x.reshape(*x.shape[:-1], -1, 2)
        else:
            x1, x2 = x.chunk(2, dim=-1)
            xc = torch.stack([x1, x2], dim=-1)

        xc = torch.view_as_complex(xc).to(freqs_cis.dtype)
        freqs_cis = freqs_cis.view(xc.size(1), 1, xc.size(3))
        out = torch.view_as_real(xc * freqs_cis).flatten(start_dim=-2)
        out = out.type_as(x)
        return out

    try:
        import rotary_emb
    except:
        rotary_emb = None

    freqs_cis = freqs_cis.to(x.dtype)
    if fused_rotary_embedding and rotary_emb is not None and freqs_cis.is_cuda:
        xc = x  # [bs, seq_len, num_heads, head_dim // 2, 2]
        batch_dims = xc.shape[:-3]
        if len(x.shape) > 4:
            xc = xc.reshape(-1, *xc.shape[-3:])

        cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]
        # llama rotary like GPT-J, inplace must be False
        out = ApplyRotaryEmb.apply(xc, cos, sin, interleaved, inplace)
        if len(x.shape) > 4:
            out = out.reshape(*batch_dims, *out.shape[-3:])
        return out
    else:
        if interleaved:
            xc = x.reshape(*x.shape[:-1], -1, 2)  # [*, seq_len, num_heads, head_dim // 2, 2]
            x0, x1 = xc[..., 0], xc[..., 1]
        else:
            x0, x1 = x.chunk(2, dim=-1)

        freqs_cis = freqs_cis.view(seq_len, 1, x0.size(-1), 2)  # [seq_len, 1, head_dim // 2, 2]
        cos, sin = freqs_cis[..., 0], freqs_cis[..., 1]
        out = torch.stack([
            x0 * cos - x1 * sin,
            x1 * cos + x0 * sin
        ], dim=-1).flatten(start_dim=-2)

        return out


class RotaryEmbedding(torch.nn.Module):
    """partial rotary embeddings, which is better than full rotary

    Ref:
        1. Wang and Komatsuzaki et al https://github.com/kingoflolz/mesh-transformer-jax/

    Args:
        dim: rotary dim, in transformer is head dim
        interpolation_factor: sequence length interpolation factor
    """

    def __init__(self,
                 dim: int,
                 theta: float = 10000.0,
                 interpolation_factor: Optional[float] = None,
                 interleaved: bool = True,
                 max_len: Optional[int] = None,
                 scale_base: Optional[float] = None,
                 inplace: bool = False,
                 dtype=None,
                 device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.interpolation_factor = interpolation_factor
        self.dim = dim
        self.max_len = max_len
        self.theta = theta
        self.interleaved = interleaved
        self.inplace = inplace
        self.scale_base = scale_base
        # lazy register buffer
        if self.max_len is not None:
            freqs_cis = precompute_freqs_cis(self.max_len,
                                             self.dim,
                                             theta=self.theta,
                                             interpolation_factor=self.interpolation_factor,
                                             scale_base=self.scale_base,
                                             **factory_kwargs)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _update_cos_sin_tables(self, seq_len, dtype=None, device=None):
        if self.max_len is None or seq_len > self.max_len:
            self.max_len = seq_len
            freqs_cis = precompute_freqs_cis(self.max_len,
                                             self.dim,
                                             theta=self.theta,
                                             interpolation_factor=self.interpolation_factor,
                                             dtype=dtype,
                                             device=device)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k: [*, seq_len, num_heads, head_dim]
        """
        seq_len = q.shape[-3]
        self._update_cos_sin_tables(seq_len, dtype=q.dtype, device=q.device)
        freqs_cis = self.freqs_cis[:seq_len]
        if self.scale_base is None:
            return (apply_rotary_emb(q, freqs_cis, interleaved=self.interleaved, inplace=self.inplace),
                    apply_rotary_emb(k, freqs_cis, interleaved=self.interleaved, inplace=self.inplace))
        else:
            return (apply_rotary_emb(q, freqs_cis[..., :2], interleaved=self.interleaved, inplace=self.inplace),
                    apply_rotary_emb(k, freqs_cis[..., 2:], interleaved=self.interleaved, inplace=self.inplace))

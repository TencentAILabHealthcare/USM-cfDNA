# Copyright (c) 2024, Tencent Inc. All rights reserved.
from typing import Any, Tuple

import torch
import torch.distributed as dist


def single_all_to_all(input, scatter_idx=-2, gather_idx=-3, group=None):
    """all-to-all for qkv

    Args:
        input: a tensor sharded along dim scatter dim, for example qkv [*, seq_len/P, num_heads, head_dim]
        scatter_idx (int): default -2
        gather_idx (int): default -3
        group : sequence parallel group

    Returns:
        [*, seq_len, num_heads/p, head_dim]
    """
    seq_world_size = dist.get_world_size(group)
    ndim = input.dim()
    if scatter_idx < 0:
        scatter_idx = ndim + scatter_idx

    if gather_idx < 0:
        gather_idx = ndim + gather_idx

    # [*, seq_len/p, num_heads/p, head_dim] -> [p, *, seq_len/p, num_heads/p, head_dim]
    input_t = torch.stack(input.chunk(seq_world_size, dim=scatter_idx), dim=0)
    output = torch.empty_like(input_t)  # [p, *, seq_len/p, num_heads/p, head_dim]

    if not input_t.is_contiguous():
        input_t = input_t.contiguous()

    if not output.is_contiguous():
        output = output.contiguous()

    dist.all_to_all_single(output, input_t, group=group)  # [p, *, seq_len/p, num_heads/p, head_dim]
    # [*, seq_len, num_heads/p, head_dim]
    output = torch.concat(output.unbind(dim=0), dim=gather_idx)
    return output


class SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: torch.Tensor, scatter_idx: int,
                gather_idx: int) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return single_all_to_all(input, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        return (None, SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class DistributedAttention(torch.nn.Module):
    """Deepspeed Ulysess distributed attention.

    Refs:
        1. from deepspeed.sequence.layer import DistributedAttention
        2. https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md

    Args:
        sequence_process_group: sequence parallel process group
        scatter_idx: scatter_idx for all2all comm, scatter along num_head axis
        gather_idx: gather_idx for all2all comm, gatther along seq_len axis
    """

    def __init__(
            self,
            sequence_process_group: dist.ProcessGroup,
            scatter_idx: int = -2,
            gather_idx: int = -3
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.spg = sequence_process_group
        self.world_size = dist.get_world_size(self.spg)
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self,
                local_attn,
                query: torch.Tensor,
                key: torch.Tensor = None,
                value: torch.Tensor = None,
                *args: Any,
                **kwargs):
        """
        Args:
            local_attn: local attention with q,k,v
            query: default shape is [*, seq_len/p, num_heads, head_dim], query input to the layer
            key: default shape is [*, seq_len/p, num_heads, head_dim], key input to the layer
            value: default shape is [*, seq_len/p, num_heads, head_dim], value input to the layer

        Returns:
            context_layer:  [*, seq_len/p, num_heads, head_dim]
            attn_weights: [*, num_heads/p, seq_len, seq_len]
        """
        num_heads = query.size(-2)
        assert num_heads % self.world_size == 0, f"heads must be divisible by {self.world_size}"

        # [*, seq_len/p, num_heads, head_dim] -> [*, seq_len, num_heads/p, head_dim]
        query = SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        if key is not None:
            key = SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)

        if value is not None:
            value = SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)

        context, attn_weight = local_attn(query, key, value, *args, **kwargs)
        assert attn_weight is None, "current not support attn weights"
        # [*, seq_len/p, num_heads, head_dim]
        context = SeqAllToAll.apply(self.spg, context, self.gather_idx, self.scatter_idx)

        return context, attn_weight


class UnifiedDistributedAttention(torch.nn.Module):
    """
    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
            self,
            ring_pg: dist.ProcessGroup,
            ulysses_pg: dist.ProcessGroup,
            scatter_idx: int = 2,
            gather_idx: int = 1,
            ring_impl_type: str = "basic",
    ) -> None:
        super(UnifiedDistributedAttention, self).__init__()
        self.ring_pg = ring_pg
        self.ulysses_pg = ulysses_pg
        assert (
                self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.ring_attn_fn = RING_IMPL_DICT[ring_impl_type]

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            *args,
            **kwargs
    ):
        query_layer = SeqAllToAll.apply(
            self.ulysses_pg, query, self.scatter_idx, self.gather_idx
        )
        key_layer = SeqAllToAll.apply(
            self.ulysses_pg, key, self.scatter_idx, self.gather_idx
        )
        value_layer = SeqAllToAll.apply(
            self.ulysses_pg, value, self.scatter_idx, self.gather_idx
        )
        context, attn_weights = self.ring_attn_fn(
            query_layer,
            key_layer,
            value_layer,
            *args,
            **kwargs,
            group=self.ring_pg
        )

        context = SeqAllToAll.apply(
            self.ulysses_pg, context, self.gather_idx, self.scatter_idx
        )
        return context

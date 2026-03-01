# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

from torch.utils.data.sampler import Sampler

from tgnn.config import configurable
from tgnn.distributed import comm
from .build import SAMPLER_REGISTRY


@SAMPLER_REGISTRY.register()
class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """
    @classmethod
    def from_config(cls, cfg, dataset):
        return {
            "size": len(dataset)
        }
        
    @configurable
    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_data_parallel_rank()
        self._world_size = comm.get_data_parallel_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

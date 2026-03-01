# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.

from typing import TypeVar, Optional

from torch.utils.data.distributed import DistributedSampler as NativeDistributedSampler

from tgnn.config import configurable
from tgnn.distributed import comm
from .build import SAMPLER_REGISTRY

T_co = TypeVar('T_co', covariant=True)


@SAMPLER_REGISTRY.register()
class DistributedSampler(NativeDistributedSampler):

    @classmethod
    def from_config(cls, cfg, dataset):
        return {
            "dataset": dataset,
            "num_replicas": comm.get_data_parallel_world_size(),
            "rank": comm.get_data_parallel_rank(),
            "drop_last": cfg.dataloader.train.drop_last,
            "shuffle": cfg.dataloader.train.shuffle
        }

    @configurable
    def __init__(self,
                 dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 drop_last: bool = False,
                 shuffle: bool = True):
        super().__init__(dataset,
                         num_replicas=num_replicas,
                         rank=rank,
                         drop_last=drop_last,
                         shuffle=shuffle)

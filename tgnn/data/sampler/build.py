# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

import torch
from torch.utils.data.distributed import DistributedSampler

from tgnn.utils.registry import Registry

SAMPLER_REGISTRY = Registry("sampler")


def build_sampler(cfg, dataset, train=True):
    if train:
        name = cfg.dataloader.train.sampler
    else:
        name = cfg.dataloader.eval.sampler

    if name not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown training sampler: {name}")

    sampler = SAMPLER_REGISTRY[name](cfg, dataset)
    assert isinstance(sampler, torch.utils.data.Sampler), f"Expect a Sampler but got {type(sampler)}"

    return sampler

# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

from tgnn.utils import print_rank_0
from tgnn.utils.registry import Registry

from .blendable_dataset import BlendableDataset

DATASET = Registry()


def blend_datasets(cfg, datasets):
    weights = cfg.dataset.weights
    ds = {}
    for name in datasets:
        if isinstance(datasets[name], (list, tuple)):
            if len(datasets[name]) == 1:
                ds[name] = datasets[name][0]
            else:
                print_rank_0(f"blending datasets: {len(datasets[name])}")
                ds[name] = BlendableDataset(datasets[name], weights=weights)
        else:
            ds[name] = datasets[name]

    return ds


def build_datasets(cfg, DATASET_REGISTRY=None, splits=None):
    print_rank_0(f"building dataset: {cfg.dataset.name}")
    if DATASET_REGISTRY is None:
        DATASET_REGISTRY = DATASET
    try:
        build_fn = DATASET_REGISTRY[cfg.dataset.name]
    except KeyError:
        raise KeyError(f"dataset {cfg.dataset.name} not found. Available datasets: {list(DATASET_REGISTRY.keys())}")

    if hasattr(build_fn, "build_dataset"):
        build_fn = build_fn.build_dataset

    datasets = build_fn(cfg, splits=splits)
    return blend_datasets(cfg, datasets)

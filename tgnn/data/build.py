# Copyright (c) 2024, Tencent Inc. All rights reserved.

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from tgnn.utils import print_rank_0
from .dataset import build_datasets
from .sampler import build_sampler
from ..utils.env import seed_all_rng


def build_data_loader(cfg, datasets=None, splits=None, collate_fn=None):
    """build dataset -> sampler -> dataloader"""
    num_workers = cfg.dataloader.num_workers
    persistent_workers = cfg.dataloader.persistent_workers
    if num_workers <= 0:
        persistent_workers = False

    datasets = datasets or build_datasets(cfg, splits=splits)
    batch_size = cfg.dataloader.batch_size
    eval_batch_size = cfg.dataloader.eval_batch_size
    if cfg.dataloader.eval_batch_size is None:
        eval_batch_size = batch_size

    prefetch_factor = cfg.dataloader.prefetch_factor
    loaders = {}
    for name, ds in datasets.items():
        is_train = name == 'train'
        if is_train:
            shuffle = cfg.dataloader.train.shuffle
        else:
            shuffle = False

        drop_last = is_train
        bs = batch_size if is_train else eval_batch_size
        if isinstance(ds, torch.utils.data.IterableDataset):
            print_rank_0(f"WARNING: {name} IterableDataset has no sampler, set sampler to None")
            sampler = None
        else:
            sampler = build_sampler(cfg, ds, train=is_train)

        if hasattr(ds, "collate_fn") and collate_fn is None:
            collate_fn = ds.collate_fn

        worker_init_fn = worker_init_reset_seed if is_train else None
        loader = DataLoader(ds,
                            batch_size=bs,
                            num_workers=num_workers,
                            pin_memory=cfg.dataloader.pin_memory,
                            drop_last=drop_last,
                            shuffle=shuffle if sampler is None else None,
                            sampler=sampler,  # If sampler specified, shuffle must be None
                            collate_fn=collate_fn,
                            worker_init_fn=worker_init_fn,
                            prefetch_factor=prefetch_factor,
                            persistent_workers=persistent_workers)
        loaders[name] = loader
        print_rank_0(f"batch size per gpu: {bs}")
        if not isinstance(ds, torch.utils.data.IterableDataset):
            print_rank_0(f"number of samples in {name} dataset: {len(ds)}")
            print_rank_0(f"number of batches in {name} dataloader: {len(loader)}")

    return loaders


def build_train_loader(cfg, collate_fn=None):
    return build_data_loader(cfg, splits=("train",), collate_fn=collate_fn)["train"]


def build_test_loader(cfg, split="eval", collate_fn=None):
    return build_data_loader(cfg, splits=(split,), collate_fn=collate_fn)[split]


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)
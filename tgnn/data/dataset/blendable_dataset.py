# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

import bisect
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset

from tgnn.utils import print_rank_0


class BlendableDataset(Dataset):
    """concat dataset with weight sampling"""

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self,
                 datasets: Iterable[Dataset],
                 weights: Sequence = None,
                 num_samples: int = None,
                 seed=42):
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            print_rank_0(f"blending subset: {len(d)}")
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.num_samples = self.cumulative_sizes[-1] if num_samples is None else num_samples
        print_rank_0(f"#blending dataset: {self.num_samples}")
        self.indices = None
        self.weights = None
        if weights is not None:
            sizes = [len(ds) for ds in self.datasets]
            assert len(self.datasets) == len(weights), f"number of datasets do not match weights"
            self.weights = torch.ones(sum(sizes), dtype=torch.float32)
            assert len(weights) == len(self.datasets)
            start = 0
            for s, w in zip(sizes, weights):
                self.weights[start: start + s] *= w
                start += s
            self.rng = torch.Generator().manual_seed(seed)
            self.indices = torch.multinomial(self.weights,
                                             self.num_samples,
                                             replacement=True,
                                             generator=self.rng)

    def _infinite_indices(self):
        while True:
            indices = torch.multinomial(self.weights, self.num_samples, True, generator=self.rng)
            yield from indices

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        if self.indices is not None:
            idx = self.indices[idx]

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.num_samples

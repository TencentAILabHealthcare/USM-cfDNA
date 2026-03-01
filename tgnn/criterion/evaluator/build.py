# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

import sys

import torch
import torch.nn as nn

from tgnn.distributed import comm
from tgnn.utils.registry import Registry
from .base_evaluator import DatasetEvaluator

EVALUATOR_REGISTRY = Registry()


class BuildInEvaluator(DatasetEvaluator):

    def __init__(self, name, monitor="mean", best_max=True):
        super().__init__(monitor, best_max)
        self.reset()
        self.metric = getattr(nn, name)()

    def reset(self):
        self._predictions = []

    def forward(self, output, target):
        meters = {}
        meters["metric"] = self.metric(output, target)

        return meters

    def evaluate(self):
        if comm.is_parallel_world():
            comm.synchronize()
            self._predictions = comm.all_gather(self._predictions, comm.get_data_parallel_group())

        predictions = torch.Tensor(self._predictions)
        self.meters = {
            "mean": torch.mean(predictions).item(),
            "std": torch.std(predictions).item(),
            "max": torch.max(predictions).item(),
            "min": torch.min(predictions).item()
        }
        return self.meters


def build_evaluator(cfg):
    name = cfg.criterion.evaluator.name

    if not name:
        print(f"not set evaluator name for building pipline", file=sys.stderr)
        return None

    if hasattr(nn, name):
        evaluator = BuildInEvaluator(name,
                                     cfg.criterion.evaluator.monitor,
                                     cfg.criterion.evaluator.best_max)
    else:
        assert name in EVALUATOR_REGISTRY, f"not registered evaluator: {name}"
        evaluator = EVALUATOR_REGISTRY.get(name)(cfg)

    evaluator.to(torch.device(cfg.solver.device))

    return evaluator.eval()

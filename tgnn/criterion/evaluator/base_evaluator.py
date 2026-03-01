# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

from collections import defaultdict

import torch.nn as nn

from tgnn.distributed import comm


class DatasetEvaluator(nn.Module):
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def __init__(self, monitor, best_max=True):
        super().__init__()
        self.monitor = monitor
        self.best_max = best_max
        self.meters = {}

    def get_monitor(self):
        assert self.monitor in self.meters, f"{self.monitor} not in meters, please evaluete first or check key"
        value = self.meters[self.monitor]
        return value if self.best_max else -value

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self.meters = {}

    def forward(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class LamadaEvaluator(DatasetEvaluator):

    def __init__(self, metric_fn, monitor, best_max=True):
        super().__init__(monitor, best_max)
        self.metric_fn = metric_fn
        self.step = 0
        self.meters = defaultdict(float)

    def forward(self, inputs, targets):
        meters = self.metric_fn(inputs, targets)
        if not isinstance(meters, dict):
            meters = dict(metric=meters)
        for name in meters:
            self.meters[name] += meters[name]
        self.step += 1

    def evaluate(self):
        if comm.is_parallel_world():
            comm.synchronize()
            comm.all_reduce_dict(self.meters)
            comm.all_reduce(self.step)

        for name in self.meters:
            self.meters[name] /= self.step

        return self.meters

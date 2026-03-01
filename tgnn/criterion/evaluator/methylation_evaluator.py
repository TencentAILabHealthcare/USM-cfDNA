# Copyright (c) 2024, Tencent Inc. All rights reserved.

import torch

from tgnn.config import configurable
from tgnn.distributed import comm
from .build import EVALUATOR_REGISTRY
from .confusion_matrix import MultiClassConfusionMatrixEvaluator, multiclass_confusion_matrix
from ..metric.roc import compute_binary_auc


@EVALUATOR_REGISTRY.register()
class MethylationEvaluator(MultiClassConfusionMatrixEvaluator):

    @classmethod
    def from_config(cls, cfg):
        return {}

    @configurable
    def __init__(self):
        super().__init__(num_classes=2)
        self.preds = []
        self.targets = []

    def reset(self):
        super().reset()
        self.preds = []
        self.targets = []

    def forward(self, outputs, targets):
        cm = multiclass_confusion_matrix(outputs["logits"], targets["target"], self.num_classes)
        self.mat += cm
        tp = torch.diag(self.mat)
        acc_global = tp.sum() / self.mat.sum()
        self.preds.append(outputs["logits"].softmax(dim=-1)[..., 1])
        self.targets.append(targets["target"])
        return acc_global

    @torch.no_grad()
    def evaluate(self):
        meters = super().evaluate()
        state = torch.stack([torch.cat(self.preds, dim=0), torch.cat(self.targets, dim=0)], dim=0)
        group = comm.get_data_parallel_group()
        state = comm.all_gather(state.to("cpu"), group=group)
        state = torch.cat(state, dim=-1)
        y_true = state[1].long()
        y_pred = state[0].float()
        auc = compute_binary_auc(y_pred, y_true)
        meters["auc"] = auc
        return meters

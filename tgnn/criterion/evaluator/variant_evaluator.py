# Copyright (c) 2024, Tencent Inc. All rights reserved.

import torch

from tgnn.config import configurable
from .build import EVALUATOR_REGISTRY
from .confusion_matrix import MultiClassConfusionMatrixEvaluator, multiclass_confusion_matrix


@EVALUATOR_REGISTRY.register()
class VariantEvaluator(MultiClassConfusionMatrixEvaluator):

    @classmethod
    def from_config(cls, cfg):
        return {}

    @configurable
    def __init__(self):
        super().__init__(num_classes=24)

    def forward(self, outputs, targets):
        self.mat += multiclass_confusion_matrix(outputs["at"], targets["at"], self.num_classes)
        h = self.mat.float()
        tp = torch.diag(h)
        acc_global = tp.sum() / h.sum()
        return {"metric": acc_global}
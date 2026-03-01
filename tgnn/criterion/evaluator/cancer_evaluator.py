# Copyright (c) 2025, Tencent Inc. All rights reserved.

import numpy as np
from tgnn.config import configurable
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from .base_evaluator import DatasetEvaluator
from .build import EVALUATOR_REGISTRY
from ..metric.classification import sensitivity_at_specificity98


@EVALUATOR_REGISTRY.register()
class MCEDEvaluator(DatasetEvaluator):

    @classmethod
    def from_config(cls, cfg):
        return {}

    @configurable
    def __init__(self):
        super().__init__("acc", best_max=True)
        self.preds = []
        self.targets = []

    def reset(self):
        self.preds = []
        self.targets = []

    def forward(self, y_prob, y_true):
        self.preds.append(y_prob)
        self.targets.append(y_true)
        acc = accuracy_score(y_true, y_prob.argmax(-1))
        return {"acc": acc}

    def metrics_from_arrays(self, preds, targets):

        if preds.shape[1] in [4, 12]:
            probs = 1.0 - (preds[:, 0] + preds[:, 1] + preds[:, 2])
            targets = (targets > 2).astype(np.int32)

        elif preds.shape[1] in [3, 11]:
            probs = 1.0 - (preds[:, 0] + preds[:, 1])
            targets = (targets > 1).astype(np.int32)


        elif preds.shape[1] == 10:
            probs = 1.0 - (preds[:, 0])
            targets = (targets > 0).astype(np.int32)

        elif preds.shape[1] == 2:
            probs = preds[:, 1]

        preds = probs > 0.5

        acc = accuracy_score(targets, preds)
        auc = roc_auc_score(targets, probs)
        f1 = f1_score(targets, preds)
        sens_98, spec_98, _ = sensitivity_at_specificity98(targets, probs)

        return {
            "acc": acc,
            "f1": f1,
            "auc": auc,
            "sens_98": sens_98,
            "spec_98": spec_98,
        }

    def compute_metrics(self):
        y_prob = np.concatenate(self.preds, axis=0)
        y_true = np.concatenate(self.targets, axis=0)
        return self.metrics_from_arrays(y_prob, y_true)


    def evaluate(self):
        self.meters = self.compute_metrics()
        return self.meters


    def __str__(self):
        result = self.evaluate()
        items = []
        for name, value in result.items():
            if value.dim() == 0:
                item = f"{value.item():.2f}"
            else:
                item = ['{:.2f}'.format(i) for i in value.tolist()]
            items.append(f"{name}={item}")

        return "\n".join(items)
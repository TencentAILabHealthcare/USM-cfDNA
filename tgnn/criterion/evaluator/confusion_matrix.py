# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

from typing import Optional

import torch

from tgnn.config import configurable
from tgnn.distributed import comm
from .base_evaluator import DatasetEvaluator
from .build import EVALUATOR_REGISTRY

def multiclass_confusion_matrix(y_pred: torch.Tensor,
                                y_true: torch.Tensor,
                                num_classes: Optional[int] = None,
                                ignore_index: Optional[int] = None,
                                mask: Optional[torch.Tensor] = None):
    """
    Args:
        y_pred: [bs,] or [bs, num_classes], index or probility of classes
        y_true: [bs,], index of classes

    Returns:
        matrix: tensor[nc, nc]
    """
    # Apply argmax if we have one more dimension
    if y_pred.ndim == y_true.ndim + 1:
        y_pred = y_pred.argmax(dim=-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_pred and y should have same shapes.")

    y_pred = y_pred.view(-1).long()
    y_true = y_true.view(-1).long()

    if mask is not None:
        mask = mask.view(-1)

    if ignore_index is not None:
        ig_mask = y_true != ignore_index
        if mask is not None:
            mask = mask * ig_mask
        else:
            mask = ig_mask

    if mask is not None:
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    nc = num_classes or (torch.max(y_true) + 1)
    inds = nc * y_true + y_pred
    # bincount only supports 1-d non-negative integral inputs
    bins = torch.bincount(inds, minlength=nc ** 2)
    return bins.reshape(nc, nc)


def mutlilabel_confusion_matrix(y_pred: torch.Tensor,
                                y_true: torch.Tensor,
                                num_labels: int = None,
                                threshold: float = 0.5,
                                ignore_index: Optional[int] = None):
    """
    Args；
        y_pred: tensor[bs, nc], output probility of every class
        y_true: tensor[bs, nc], onehot ground truth

    Returns:
        matrix: tensor[num_labels, 2, 2]
    """
    num_labels = num_labels or y_true.shape[-1]
    y_pred = (y_pred > threshold).type(torch.int64)
    y_pred = torch.movedim(y_pred, 1, -1).reshape(-1, num_labels)
    y_true = torch.movedim(y_true, 1, -1).reshape(-1, num_labels)
    if ignore_index is not None:
        y_pred = y_pred.clone()
        y_true = y_true.clone()
        # Make sure that when we map, it will always result in a negative number that we can filter away
        # Each label correspond to a 2x2 matrix = 4 elements per label
        mask = y_true == ignore_index
        y_pred[mask] = -4 * num_labels
        y_true[mask] = -4 * num_labels
    unique_mapping = ((2 * y_true + y_pred) + 4 * torch.arange(num_labels, device=y_pred.device)).flatten()
    unique_mapping = unique_mapping[unique_mapping >= 0]
    bins = torch.bincount(unique_mapping, minlength=4 * num_labels)
    return bins.reshape(num_labels, 2, 2)


@EVALUATOR_REGISTRY.register()
class MultiClassConfusionMatrixEvaluator(DatasetEvaluator):

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.model.num_classes
        }

    @configurable
    def __init__(self,
                 num_classes,
                 monitor="acc_global",
                 best_max=True,
                 mode="macro"):
        super().__init__(monitor, best_max=best_max)
        self.num_classes = num_classes
        assert mode in ("macro", "micro", "weight"), f"only support macro, micro, weight, but get {mode}"
        self.mode = mode
        self.register_buffer("mat", torch.zeros((num_classes, num_classes), dtype=torch.int64))

    def reset(self):
        self.mat.zero_()

    def forward(self, y_pred, y_true, mask=None):
        cm = multiclass_confusion_matrix(y_pred, y_true, num_classes=self.num_classes, mask=mask)
        self.mat += cm
        tp = torch.diag(self.mat)
        acc_global = tp.sum() / self.mat.sum()
        return acc_global

    def compute_metrics(self):
        h = self.mat.float()
        tp = torch.diag(h)
        acc_global = tp.sum() / h.sum()
        p = h.sum(0)
        n = h.sum(1)
        fp = p - tp
        fn = n - tp
        tn = h.sum() - fp - fn - tp
        acc = (tp + tn) / h.sum()
        # recall = tp / (fn + tp)
        recall = tp / n
        # precision = tp / (tp + fp)
        precision = tp / p
        f1_score = (2.0 * precision * recall) / (precision + recall)
        return {
            "acc_global": acc_global,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1_score
        }

    @torch.no_grad()
    def evaluate(self):
        if comm.is_parallel_world():
            comm.synchronize()
            group = comm.get_data_parallel_group()
            mats = comm.all_gather(self.mat.to("cpu"), group=group)
            self.mat = torch.stack(mats, dim=-1).sum(dim=-1).to(self.mat.device)

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


EVALUATOR_REGISTRY.register("ConfusionMatrixEvaluator", MultiClassConfusionMatrixEvaluator)

# Copyright (c) 2025, Tencent Inc. All rights reserved.

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from tgnn.utils import warn_rank_0


def _binary_clf_curve(
        preds: torch.Tensor,
        target: torch.Tensor,
        sample_weights: Optional[Union[Sequence, torch.Tensor]] = None,
        pos_label: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the TPs and false positives for all unique thresholds in the preds tensor.

    Adapted from
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py.

    Args:
        preds: 1d tensor with predictions
        target: 1d tensor with true values
        sample_weights: a 1d tensor with a weight per sample
        pos_label: integer determining what the positive class in target tensor is

    Returns:
        fps: 1d tensor with false positives for different thresholds
        tps: 1d tensor with true positives for different thresholds
        thresholds: the unique thresholds use for calculating fps and tps
    """
    if sample_weights is not None and not isinstance(sample_weights, torch.Tensor):
        sample_weights = torch.tensor(sample_weights, device=preds.device, dtype=torch.float)

    # remove class dimension if necessary
    if preds.ndim > target.ndim:
        preds = preds[:, 0]

    desc_score_indices = torch.argsort(preds, descending=True)
    preds = preds[desc_score_indices]
    target = target[desc_score_indices]

    weight = sample_weights[desc_score_indices] if sample_weights is not None else 1.0

    # pred typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = torch.where(preds[1:] - preds[:-1])[0]
    threshold_idxs = F.pad(distinct_value_indices, [0, 1], value=target.size(0) - 1)
    target = (target == pos_label).to(torch.long)
    tps = torch.cumsum(target * weight, dim=0)[threshold_idxs]

    if sample_weights is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = torch.cumsum((1 - target) * weight, dim=0)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps

    return fps, tps, preds[threshold_idxs]


def compute_binary_auc(preds: torch.Tensor, target: torch.Tensor, pos_label: int = 1):
    """
    ```python
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(target.long().cpu().numpy(), preds.float().cpu().numpy())
    ```
    """
    fps, tps, thres = _binary_clf_curve(preds=preds.float(), target=target.long(), pos_label=pos_label)
    # Add an extra threshold position to make sure that the curve starts at (0, 0)
    tps = torch.cat([torch.zeros(1, dtype=tps.dtype, device=tps.device), tps])
    fps = torch.cat([torch.zeros(1, dtype=fps.dtype, device=fps.device), fps])
    thres = torch.cat([torch.ones(1, dtype=thres.dtype, device=thres.device), thres])
    if fps[-1] <= 0:
        warn_rank_0(
            "No negative samples in targets, false positive value should be meaningless."
            " Returning zero tensor in false positive score",
            UserWarning,
        )
        fpr = torch.zeros_like(thres)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warn_rank_0(
            "No positive samples in targets, true positive value should be meaningless."
            " Returning zero tensor in true positive score",
            UserWarning,
        )
        tpr = torch.zeros_like(thres)
    else:
        tpr = tps / tps[-1]

    auc_score = torch.trapz(tpr, fpr, dim=-1)
    return auc_score

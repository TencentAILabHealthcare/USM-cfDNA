# Copyright (c) 2025, Tencent Inc. All rights reserved.

import numpy as np

from sklearn.metrics import roc_curve


def sensitivity_at_specificity(y_true, y_pred, specificity):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # Find threshold where specificity >= 0.98 (i.e., fpr <= 0.02)
    idx = np.where(fpr <= 1.0 - specificity)[0]
    if len(idx) == 0:
        return 0.0, 0.0, 0.0

    return tpr[idx[-1]], 1.0 - fpr[idx[-1]], thresholds[idx[-1]]


def sensitivity_at_specificity98(y_true, y_pred):
    return sensitivity_at_specificity(y_true, y_pred, 0.98)


def sensitivity_at_specificity95(y_true, y_pred):
    return sensitivity_at_specificity(y_true, y_pred, 0.95)

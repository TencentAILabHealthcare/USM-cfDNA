# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import numpy as np


def fit_vocab_size_to_dtype(vocab_size):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32

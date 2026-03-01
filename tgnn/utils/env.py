# Copyright (c) 2024, Tencent Inc. All rights reserved.

import logging
import os
import random
from datetime import datetime
import resource
import numpy as np
import torch


def get_torch_version():
    return tuple(int(x) for x in torch.__version__.split(".")[:2])


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (os.getpid() +
                int(datetime.now().strftime("%S%f")) +
                int.from_bytes(os.urandom(2), "big")
                )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))

    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))


def get_traced_memory():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

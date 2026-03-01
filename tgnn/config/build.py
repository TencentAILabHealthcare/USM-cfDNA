# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
from .default import _C, CONFIG_REGISTRY


def register_meta_config(name=None):
    if name is None:
        return _C
    
    cfg = CONFIG_REGISTRY[name](_C)
    return cfg


def get_config(name=None, clone=False):
    if name is not None:
        register_meta_config(name)

    if clone:
        return _C.clone()

    return _C

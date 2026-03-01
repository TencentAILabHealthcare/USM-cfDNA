# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import torch
import torch.nn as nn
from tgnn.utils import Registry, get_torch_version, get_logger

MODEL_REGISTRY = Registry("model")  # noqa F401 isort:skip

loger = get_logger()


def build_model(cfg):
    """build model by architecture name
    """
    arch = cfg.model.arch
    assert arch in MODEL_REGISTRY, f"{arch} not in model registry"
    model = MODEL_REGISTRY.get(arch)(cfg)
    if isinstance(model, nn.Module):
        model.to(getattr(torch, cfg.solver.dtype))
        if cfg.model.compile.enabled:
            loger.info("set model compiled")
            assert get_torch_version() >= [2, 0], f"only pytorch 2.0 support model compile, get {torch.__version__}"
            model = torch.compile(model, backend=cfg.model.compile.backend)
        model.to(cfg.solver.device)

    return model

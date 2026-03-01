# Copyright (c) 2025, Tencent Inc. All rights reserved.
import os

import lightgbm as lgb
import inspect
import numpy as np
from tgnn.config import configurable
from tgnn.model.build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class LGBMClassifier(lgb.LGBMClassifier):

    @classmethod
    def from_config(cls, cfg):
        signature = inspect.signature(lgb.LGBMClassifier.__init__)
        kwargs = {}
        parameters = signature.parameters
        for name in list(parameters.keys())[1:]:
            p = parameters[name]
            name = p.name
            if name in ("kwargs",):
                continue
            if hasattr(cfg.model, name):
                value = getattr(cfg.model, name)
            elif hasattr(cfg.solver, name):
                value = getattr(cfg.solver, name)
            elif hasattr(cfg.criterion.loss, name) and name in ("objective",):
                value = getattr(cfg.criterion.loss, name)
            else:
                value = p.default
            kwargs[name] = value

        print("lightgbm model parameters:", kwargs)

        return kwargs

    def to(self, device=None):
        if device is not None:
            self.set_params(device=device)

    def save_model(self, output):
        self.booster_.save_model(output)
        return self

    def load_model(self, filename):
        self.booster_ = lgb.Booster(model_file=filename)
        return self


LGBMClassifier.__init__ = configurable(LGBMClassifier.__init__)

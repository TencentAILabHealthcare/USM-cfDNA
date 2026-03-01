# Copyright (c) 2025, Tencent Inc. All rights reserved.
import os

import xgboost as xgb
import inspect
import numpy as np
from tgnn.config import configurable
from tgnn.model.build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class XGBClassifier(xgb.XGBClassifier):

    @classmethod
    def from_config(cls, cfg):
        signature = inspect.signature(xgb.XGBModel.__init__)
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
            elif hasattr(cfg.criterion.loss, name) and name in ("objective", "random_state"):
                value = getattr(cfg.criterion.loss, name)
            elif hasattr(cfg.criterion.evaluator, name) and name in ("eval_metric",):
                value = getattr(cfg.criterion.evaluator, name)
            else:
                value = p.default
            kwargs[name] = value

        print("xgboost model parameters:", kwargs)
        return kwargs

    @configurable
    def __init__(self, **kwargs):
        super(XGBClassifier, self).__init__(**kwargs)

    def to(self, device=None):
        self.device = device
        if "cuda" in self.device:
            self.tree_method = f"cuda_{self.tree_method.split('_')[-1]}"

    def __call__(self, x, **kwargs):
        return self.predict_proba(x, **kwargs)


@MODEL_REGISTRY.register()
class DaskXGBClassifier(xgb.dask.DaskXGBClassifier):

    @classmethod
    def from_config(cls, cfg):
        signature = inspect.signature(xgb.XGBModel.__init__)
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
            elif hasattr(cfg.criterion.loss, name) and name in ("objective", "random_state"):
                value = getattr(cfg.criterion.loss, name)
            elif hasattr(cfg.criterion.evaluator, name) and name in ("eval_metric",):
                value = getattr(cfg.criterion.evaluator, name)
            else:
                value = p.default
            kwargs[name] = value

        print("xgboost model parameters:", kwargs)
        return kwargs

    @configurable
    def __init__(self, **kwargs):
        super(XGBClassifier, self).__init__(**kwargs)

    def to(self, device=None):
        self.device = device
        if "cuda" in self.device:
            self.tree_method = f"cuda_{self.tree_method.split('_')[-1]}"

    def __call__(self, x, **kwargs):
        return self.predict_proba(x, **kwargs)


@MODEL_REGISTRY.register()
class EnsembleXGBClassifier:

    @classmethod
    def from_config(cls, cfg):
        signature = inspect.signature(xgb.XGBModel.__init__)
        kwargs = {}
        parameters = signature.parameters
        for name in list(parameters.keys())[1:]:
            p = parameters[name]
            value = cfg.model.get(p.name, p.default)
            kwargs[p.name] = value
        return cls(**kwargs)

    def __init__(self, num_models=5, **kwargs):
        self.models = [XGBClassifier(**kwargs) for _ in range(num_models)]
        
    def load_model(self, model_paths):
        for model_path, model in zip(model_paths, self.models):
            model.load_model(model_path)
        return self

    def save_model(self, model_path):
        save_dir, filename = os.path.split(model_path)
        name, ext = filename.split(".")
        for i, model in enumerate(self.models):
            model.save_model(os.path.join(save_dir, f"{name}_{i}.{ext}"))

    def to(self, device=None):
        for model in self.models:
            model.to(device)

    def __call__(self, x, **kwargs):
        predictions = []
        for model in self.models:
            predictions.append(model.predict_proba(x, **kwargs)[0])

        predictions = np.stack(predictions, axis=0).mean(axis=0)
        return predictions

    def fit(self, *args, **kwargs):
        for model in self.models:
            model.fit(*args, **kwargs)

        return self

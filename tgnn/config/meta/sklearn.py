# Copyright (c) 2025, Tencent Inc. All rights reserved.

from tgnn.config.config_node import CN
from tgnn.config.default import CONFIG_REGISTRY


@CONFIG_REGISTRY.register()
def sklearn(_C):
    _C.criterion.loss.objective = "binary:logistic"

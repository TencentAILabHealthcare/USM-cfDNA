# Copyright (c) 2024, Tencent Inc. All rights reserved.
from ml_collections import FieldReference

from tgnn.utils import Registry
from .config_node import CfgNode as CN

CONFIG_REGISTRY = Registry("config")

_C = CN()
_C.jobname = FieldReference("")
_C.disk_dir = FieldReference("/mnt")
_C.experiment_dir = "experiments"
_C.log_dir = _C.get_ref("experiment_dir") + "/logs"
_C.model_dir = _C.get_ref("experiment_dir") + "/models"
_C.cache_dir = _C.get_ref("disk_dir") + "/.cache"
_C.torchhub_dir = _C.get_ref("cache_dir") + "/torch/hub"

_C.rng_seed = 42
_C.log_freq = 1
_C.seq_len = 512
_C.server = CN()
_C.port = 8888

# -------------------------------dataloader----------------------#
_C.dataset = CN()
_C.dataset.name = ""
_C.dataset.data_dir = "bio_datasets"
_C.dataset.splits = None
_C.dataset.files = None
_C.dataset.seq_len = 512
_C.dataset.max_seqs = 48
_C.dataset.weights = None
# -------------------------------solver-------------------------#
_C.solver = CN()
_C.solver.device = "cuda"
_C.solver.dtype = "float32"  # param dtype
# -------------------------------criterion----------------------#
_C.criterion = CN()
_C.criterion.loss = CN()
_C.criterion.evaluator = CN()
# -------------------------------model-------------------------#
_C.model = CN()
_C.model.arch = "gpt"
_C.model.type = "gpt2"
_C.model.weights = None
# -------------------------------distributed-----------------------#
_C.distributed = CN()
_C.distributed.default_port = 6000
_C.distributed.rank = 0
_C.distributed.local_rank = None
_C.distributed.world_size = 1
_C.distributed.backend = "nccl"

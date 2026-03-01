# Copyright (c) 2025, Tencent Inc. All rights reserved.

from tgnn.config.config_node import CN
from tgnn.config.default import CONFIG_REGISTRY


@CONFIG_REGISTRY.register()
def nn(_C):
    _C.dataloader = CN()
    _C.dataloader.num_workers = 4
    _C.dataloader.pin_memory = False
    # when num_workers > 0, prefetch_factor is 2
    _C.dataloader.prefetch_factor = None

    _C.dataloader.batch_size = 1
    _C.dataloader.eval_batch_size = None

    _C.dataloader.train = CN()
    _C.dataloader.train.sampler = "DistributedSampler"
    _C.dataloader.train.batch_sampler = None
    _C.dataloader.train.shuffle = True
    _C.dataloader.train.drop_last = True
    _C.dataloader.train.sampler_weights = None

    _C.dataloader.eval = CN()
    _C.dataloader.eval.sampler = "InferenceSampler"
    _C.dataloader.eval.batch_sampler = None
    _C.dataloader.persistent_workers = True
    _C.dataloader.rampup_batch_size = None
    _C.dataloader.starmap_inputs = True
    # -------------------------------tokenizer----------------------#
    _C.tokenizer = CN()
    _C.tokenizer.name = "sentencepiece"
    _C.tokenizer.path = ""
    # -------------------------------model-------------------------#
    _C.model.pooler = None
    _C.model.num_layers = 6
    _C.model.num_kv_heads = None
    _C.model.num_heads = 6
    _C.model.num_row_kv_heads = None
    _C.model.num_col_kv_heads = None
    _C.model.num_hiddens = 768
    _C.model.dropout = 0.
    _C.model.droppath = 0.
    _C.model.bias = False
    _C.model.num_classes = 1
    _C.model.eps = 1e-5
    _C.model.sync_batch_norm = False
    _C.model.compile = CN({"enabled": False})
    _C.model.compile.backend = "inductor"  # hidet
    _C.model.compile.mode = None
    _C.model.attention_mode = "native"
    # -------------------------------distributed-----------------------#
    _C.distributed.ddp = CN()
    _C.distributed.ddp.size = 1

    # pipeline model parallel
    _C.distributed.pmp = CN()
    _C.distributed.pmp.virtual_size = None
    _C.distributed.pmp.size = 1

    # tensor model parallel
    _C.distributed.tmp = CN()
    _C.distributed.tmp.size = 1

    # sequence parallel
    _C.distributed.sqp = CN()
    _C.distributed.sqp.size = 1
    # -------------------------------deepspeed-------------------------#
    _C.deepspeed = CN()
    _C.deepspeed.enabled = True
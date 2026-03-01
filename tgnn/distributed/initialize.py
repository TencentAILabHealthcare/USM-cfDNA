# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.

import logging
import os

import deepspeed
import torch
from torch import distributed as dist

from . import parallel_state


def initialize_distributed(cfg):
    """Initialize torch.distributed and mpu."""
    device_count = torch.cuda.device_count()
    torch.set_float32_matmul_precision('high')
    loger = logging.getLogger(__name__)
    if dist.is_initialized():
        loger.info('> torch distributed is already initialized, skipping initialization ...')
        cfg.distributed.rank = dist.get_rank()
        cfg.distributed.world_size = dist.get_world_size()
    else:
        loger.info('> initializing torch distributed ...')
        cfg.distributed.rank = int(os.getenv("RANK", 0))
        cfg.distributed.world_size = int(os.getenv("WORLD_SIZE", 1))
        # Manually set the device ids.
        if device_count > 0:
            device = cfg.distributed.rank % device_count
            if cfg.distributed.local_rank is not None:
                assert cfg.distributed.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                cfg.distributed.local_rank = device
            torch.cuda.set_device(device)  # only do so when device_count > 0

        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', str(cfg.distributed.default_port))
        init_method = f"tcp://{master_ip}:{master_port}"
        loger.info(f"> init_method: {init_method}")
        if cfg.deepspeed.enabled:
            deepspeed.init_distributed(cfg.distributed.backend)
        else:
            dist.init_process_group(
                backend=cfg.distributed.backend,
                world_size=cfg.distributed.world_size,
                rank=cfg.distributed.rank,
                init_method=init_method)

    cfg.distributed.ddp.size = cfg.distributed.world_size // cfg.distributed.pmp.size
    loger.info(f"> initializing data parallel with size {cfg.distributed.ddp.size}.")

    # Set 3d parallel
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            loger.info('> model parallel is already initialized')
        else:
            tensor_model_parallel_size = cfg.distributed.tmp.size
            pipeline_model_parallel_size = cfg.distributed.pmp.size
            virtual_pipeline_model_parallel_size = cfg.distributed.pmp.virtual_size
            sequence_parallel_size = cfg.distributed.sqp.size
            parallel_state.initialize_model_parallel(tensor_model_parallel_size,
                                                     pipeline_model_parallel_size,
                                                     sequence_parallel_size,
                                                     virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size)

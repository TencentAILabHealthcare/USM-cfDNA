# Copyright (c) 2024, Tencent Inc. All rights reserved.

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from fvcore.common.file_io import PathManager

from tgnn.config import get_config, CfgNode, register_meta_config
from tgnn.distributed import comm, initialize_distributed
from tgnn.model import csrc
from tgnn.utils import print_rank_0, setup_logger, seed_all_rng
from .utils import auto_register_settings


def compile_dependencies(cfg):
    if comm.is_local_rank_0():
        start_time = time.time()
        print_rank_0('>>> compiling and loading fused kernels ...')
        if torch.cuda.device_count() > 0:  # Skip when CPU-only
            csrc.load(verbose=True)
        print_rank_0(
            f'>>> done with compiling and loading fused kernels. Compilation time: {time.time() - start_time:.3f} seconds')
        comm.synchronize()
    else:
        print_rank_0('>>> wating main rank compile kernel')
        comm.synchronize()
        if torch.cuda.device_count() > 0:
            csrc.load(verbose=False)
        print_rank_0('>>> loading fused kernels ...')


def setup_torchhub(cfg):
    torch.hub.set_dir(cfg.torchhub_dir)


def merge_config(cfg, cfg_file=None, options=None):
    if cfg_file is not None:
        load_cfg = CfgNode.load_from_file(cfg_file)
        print(load_cfg)
        meta = load_cfg.get("meta", None)
        register_meta_config(meta)
        cfg.merge_from_file(cfg_file)

    if options is not None:
        cfg.merge_from_list(options)

    if not cfg.jobname and cfg_file is not None:
        cfg.jobname = Path(cfg_file).stem

    if not os.path.isabs(cfg.experiment_dir):
        cfg.experiment_dir = f"{cfg.disk_dir}/{cfg.experiment_dir}"

    cfg.experiment_dir = f"{cfg.experiment_dir}/{cfg.jobname}"
    if not os.path.isabs(cfg.dataset.data_dir):
        cfg.dataset.data_dir = f"{cfg.disk_dir}/{cfg.dataset.data_dir}"

    return cfg


def default_setup(args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
    """
    cfg = get_config()
    cfg = merge_config(cfg, args.cfg, options=args.opts)
    setup_torchhub(cfg)
    rank = int(os.getenv("RANK", 0))
    if rank == 0:
        PathManager.mkdirs(cfg.log_dir)

    # user can specific log output
    log_file = cfg.get("log_file", None)
    if log_file is None:
        log_file = cfg.log_dir

    logger = setup_logger(output=log_file, distributed_rank=rank)
    dtype = cfg.solver.dtype
    if dtype == "bfloat16" and (not torch.cuda.is_bf16_supported(including_emulation=False)):
        logger.warning(f"gpu don't support bfloat16 training, auto covert to float16 training")
        cfg.solver.dtype = "float16"

    if cfg.model.attention_mode in ("v1", "v2", "flash_attn", "flash"):
        try:
            import flash_attn


        except ImportError:
            cfg.model.attention_mode = "native"
            if rank == 0:
                logger.warning(f"GPU not support flash attention, change to SDPA")

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)

    if major < 8:
        cfg.model.attention_mode = "native"
        if rank == 0:
            logger.warning(
                f"GPU compute capability sm{major}{minor} < sm80, "
                "flash attention not supported, fallback to SDPA"
            )

    logger.info(">>> set torch no grad enabled")
    torch.set_grad_enabled(False)
    logger.info(">>> command line arguments: " + str(args))

    return cfg


def default_argument_parser(ckpt=None, epilog=None):
    epilog = epilog or f"""
        Examples:
        
        Run on single machine:
            $ python3 script.py --cfg cfg.yaml
        
        Change some config options:
            $  python3 --cfg cfg.yaml model.weights /path/to/weight.pth
        
        Run on multiple machines:
            deepspeed --num_nodes=$HOST_NUM --num_gpus=$HOST_GPU_NUM python3 script.py --cfg cfg.yaml
        """
    parser = argparse.ArgumentParser(
        description=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", default=None, metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument("--ckpt", default=ckpt, help="checkpoint path")
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='local rank passed from distributed launcher')
    return parser


def launch(main_func, args=None):
    """initialize distributed envrionmnet"""
    cfg = get_config("nn")
    default_port = cfg.distributed.default_port
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(default_port))
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    main_dir = Path(sys.modules['__main__'].__file__).parent
    auto_register_settings(str(main_dir))
    print_rank_0(f">>> auto register main directory: {main_dir}")
    default_setup(args)
    initialize_distributed(cfg)
    print_rank_0(">>> set random seed by rank")
    seed_all_rng(cfg.rng_seed)
    print_rank_0(">>> compile_dependencies")
    compile_dependencies(cfg)
    print_rank_0(f">>> finish compile dependencies")
    main_func(args)

# Copyright (c) 2025, Tencent Inc. All rights reserved.

import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from fvcore.common.file_io import PathManager
from tgnn.model import build_model
from tgnn.utils import setup_logger, seed_all_rng
from tgnn.config import get_config
from tgnn.criterion.evaluator import build_evaluator
from tgnn.utils import get_logger
from .default import merge_config


class SklearnTrainer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = get_logger()
        self.model = None
        self.model_dir = self.cfg.model_dir
        self.evaluator = None

    def build_dataset(self, splits=None):
        raise NotImplemented

    def build_model(self):
        self.model = build_model(self.cfg)
        return self.model

    def build_sample_weight(self, y):
        weights = compute_sample_weight('balanced', y=y)
        return weights

    def build_class_weight(self, y):
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        weights = dict(zip(classes, weights))
        return weights

    def resume_or_load(self, path=None, resume=True):
        self.model = self.build_model()
        if resume:
            filename = f"{self.cfg.model_dir}/model.json"
        else:
            filename = path

        if filename is not None:
            self.logger.info(f"Loading model from {filename}")
            self.model.load_model(filename)

    def build_evaluator(self):
        self.evaluator = build_evaluator(self.cfg)

    def grid_search(self, cv=3):
        params = dict(self.cfg.model.search)
        gbm = GridSearchCV(self.model, params, cv=cv)
        train_x, train_y = self.build_dataset()
        return gbm.fit(train_x, train_y)

    def train(self, verbose=True, **kwargs):
        datasets = self.build_dataset(splits=["train", "eval"])
        train_x, train_y = datasets["train"]
        eval_x, eval_y = datasets["eval"]
        self.logger.info("start training....")
        self.logger.info(f"number of train samples: {len(train_x)}")
        self.model.fit(train_x, train_y,
                       eval_set=[eval_x, eval_y],
                       sample_weight=self.build_sample_weight(train_y),
                       verbose=verbose,
                       **kwargs)

    def evaluate(self, eval_set=None, split=None, overall=False, **kwargs):
        if eval_set is None:
            eval_set = self.build_dataset(splits=[split, ])[split]
            eval_set = [eval_set, ]

        if self.evaluator is None:
            self.build_evaluator()

        meters = []

        for (x, y) in eval_set:
            pred = self.model.predict_proba(x, **kwargs)
            self.evaluator(pred, y)
            m = self.evaluator.metrics_from_arrays(pred, y)
            meters.append(m)
            for k, v in m.items():
                if isinstance(v, float):
                    self.logger.info(f"{k}: {v: .3f}")
                elif isinstance(v, (np.ndarray, list, tuple)):
                    self.logger.info(f"{k}: {v}")
                else:
                    self.logger.info(f"{k}: {v}")

        return meters


def default_argument_parser(cfg_file=None, epilog=None):
    epilog = epilog or f"""
        Examples:

        Run on single machine:
            $ python3 --cfg cfg.yaml
        
        Change some config options:
            $  python3 --cfg cfg.yaml --resume model.weights /path/to/weight.pth solver.learning_rate 0.001
        """
    parser = argparse.ArgumentParser(
        description=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", default=cfg_file, metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument("--eval-only", "--eval", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    return parser


def default_setup(args):
    cfg = get_config()
    cfg = merge_config(cfg, args.cfg, options=args.opts)
    PathManager.mkdirs(cfg.log_dir)
    PathManager.mkdirs(cfg.model_dir)
    if args.eval_only:
        output = cfg.log_dir
    else:
        output = cfg.experiment_dir
    logger = setup_logger(output=output)
    path = os.path.join(cfg.experiment_dir, "config.yaml")
    with PathManager.open(path, "w") as f:
        f.write(cfg.dump())
    logger.info("Full config saved to {}".format(path))
    seed_all_rng(cfg.rng_seed)


def initialize_distributed(main_func, args=None):
    num_gpus = torch.cuda.device_count()
    import dask
    import dask.distributed as ddist
    from dask_cuda import LocalCUDACluster
    with LocalCUDACluster(n_workers=num_gpus, threads_per_worker=num_gpus) as cluster:
        with ddist.Client(cluster) as client, dask.config.set({"array.backend": "cupy"}):
            args.client = client
            main_func(args)

def launch(main_func, args=None):
    default_setup(args)
    main_func(args)

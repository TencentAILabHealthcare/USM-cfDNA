# Copyright (c) 2024, Tencent Inc. All rights reserved.

import itertools

import numpy as np
import torch
import torch.nn as nn
import tqdm

from tgnn.config import CN
from tgnn.criterion import build_evaluator
from tgnn.data import build_data_loader
from tgnn.distributed import comm
from tgnn.model import build_model
from tgnn.tokenizer import build_tokenizer
from tgnn.utils.io import to_device
from tgnn.utils.logger import get_logger


class Predictor(nn.Module):

    def __init__(self, cfg: CN):
        super(Predictor, self).__init__()
        self.cfg = cfg
        self.logger = get_logger()
        self.model = None

    def builder_tokenizer(self, cfg):
        return build_tokenizer(cfg)

    def build_model(self, cfg):
        self.logger.info(f"start building model")
        model = build_model(cfg)
        return model

    def build_dataloader(self, split="eval", collate_fn=None):
        loaders = build_data_loader(self.cfg, splits=(split,), collate_fn=collate_fn)
        return loaders[split]

    @torch.no_grad()
    def predict(self, dataloader):
        training = self.model.training
        device = self.model.device
        self.model.eval()
        pbar = tqdm.tqdm(enumerate(dataloader),
                         total=len(dataloader),
                         desc=f"prediction",
                         disable=not comm.is_rank_0())

        outputs = []
        for i, batch in pbar:
            batch_x = to_device(batch["inputs"], device)
            out = self.model(**batch_x)
            outputs.append(out)

        if comm.is_rank_0():
            outputs = itertools.chain(comm.gather(outputs, 0, group=comm.get_data_parallel_group()))
            outputs = list(outputs)

        self.model.train(training)
        return outputs

    def build_evaluator(self):
        return build_evaluator(self.cfg)

    def load_model(self, ckpt):
        state = torch.load(ckpt, map_location="cpu")
        model_state = state.get('model', state)
        model_state = model_state.get('module', model_state)
        config = state.get('config', {})
        self.cfg.update(config)
        if self.cfg.model.attention_mode in ("v1", "v2", "flash_attn", "flash"):
            try:
                import flash_attn
            except ImportError:
                self.cfg.model.attention_mode = "native"
                if comm.is_rank_0():
                    self.logger.warning(f"GPU not support flash attention, change to SDPA")

        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)

        if major < 8:
            self.cfg.model.attention_mode = "native"
            if comm.is_rank_0():
                self.logger.warning(
                    f"GPU compute capability sm{major}{minor} < sm80, "
                    "flash attention not supported, fallback to SDPA"
                )

        self.model = self.build_model(self.cfg)
        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"loading model states from {ckpt}")
        self.model.load_state_dict(model_state)
        return self.model

    @torch.inference_mode()
    def evaluate(self, dataloader=None, evaluator=None, tag="eval"):
        if dataloader is None:
            dataloader = self.build_dataloader()

        if evaluator is None:
            evaluator = self.build_evaluator()

        evaluator.reset()
        training = self.model.training
        device = self.model.device
        self.model.eval()
        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                         desc="evaluation", disable=not comm.is_rank_0())
        for i, (inputs, targets) in pbar:
            batch_x = to_device(inputs, device)
            outputs = self.model(**batch_x)
            targets = to_device(targets, device)
            results = evaluator(outputs, targets)
            if isinstance(results, dict) and "metric" in results:
                metric = results["metric"]
                pbar.set_description(f"{metric: .3f}")
            elif isinstance(results, (torch.Tensor, int, float)):
                if isinstance(results, torch.Tensor):
                    results = results.item()
                pbar.set_description(f"{results: .3f}")

        meters = evaluator.evaluate()
        self.model.train(training)
        for k, v in meters.items():
            if isinstance(v, float):
                self.logger.info(f"{k}: {v: .3f}")
            elif isinstance(v, (np.ndarray, torch.Tensor)):
                self.logger.info(f"{k}: {v.tolist()}")
            elif isinstance(v, (np.ndarray, list, tuple)):
                self.logger.info(f"{k}: {v}")
            else:
                self.logger.info(f"{tag}/{k}: {v}")

        self.logger.info(f"evaluate results: {evaluator.get_monitor()}")

        return evaluator.get_monitor()

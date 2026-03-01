# Copyright (c) 2024, Tencent Inc. All rights reserved.
import os
import sys

sys.path.append(".")

from functools import partial
from datetime import datetime
import numpy as np
import pysam
import torch
import tqdm
from multiprocessing.pool import Pool
from tgnn.utils.io import to_device
from tgnn.distributed import comm
from tgnn.config import get_config
from tgnn.data import MMapIndexedJsonlDatasetBuilder
from tgnn.engine import default_argument_parser, launch, Predictor
from tgnn.utils import get_logger
from tgnn.multiprocessing import get_cpu_cores
from tgnn.utils.env import get_torch_version
from tgnn.utils.pack_files import open_resource_text


def filter_cg_sits(cg_sits, bam_file, disable=False, quality_threshold=15, min_depth=3):
    af = pysam.AlignmentFile(bam_file, "rb")
    select_sits = []
    for contig, pos in tqdm.tqdm(cg_sits, disable=disable):
        counts = af.count_coverage(contig, pos, pos + 1, quality_threshold=quality_threshold)
        depth = np.array(counts).sum()
        c_counts = counts[1][0]
        if c_counts < 1:
            continue

        if depth < min_depth:
            continue

        select_sits.append((contig, pos))
    af.close()
    return select_sits


class MethylationCaller(Predictor):

    @classmethod
    def load_reference_cg_sits(cls, filename):
        cg_sits = []
        with open_resource_text(filename, "all_sites") as f:
            lines = f.readlines()
            if filename.endswith(".csv"):
                lines = lines[1:]

            for i, line in tqdm.tqdm(enumerate(lines)):
                if line.startswith("#"):
                    continue

                contig, pos = line.strip().split()[:2]
                cg_sits.append((contig, int(pos)))

        return cg_sits

    @classmethod
    def filter_cg_sits(cls,
                       cg_sits,
                       bam_file,
                       output,
                       quality_threshold=15,
                       min_depth=3,
                       num_workers=0):
        filter_fn = partial(filter_cg_sits, quality_threshold=quality_threshold, min_depth=min_depth)
        n = int(np.ceil(len(cg_sits) / num_workers))
        chunks = [cg_sits[i:i + n] for i in range(0, len(cg_sits), n)]
        arg_lists = [(sits, bam_file, True) for sits in chunks]
        arg_lists[0] = (arg_lists[0][0], arg_lists[0][1], False)
        select_sits = []
        builder = MMapIndexedJsonlDatasetBuilder(output)
        with Pool(processes=num_workers) as pool:
            for sits in pool.starmap(filter_fn, arg_lists):
                for (contig, position) in sits:
                    builder.add_item({"contig": contig, "position": position})
                    builder.end_document()
                select_sits.extend(sits)
        builder.finalize()
        return select_sits

    def load_model(self, ckpt):
        self.logger.info(f"loading model states from {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        model_state = state.get('model', state)
        model_state = model_state.get('module', model_state)
        config = state.get('config', {})
        dataset = config.pop("dataset", None)
        config["model"].pop("attention_mode", None)
        self.cfg.update(config)
        self.cfg.dataset.seq_len = dataset["seq_len"]
        self.cfg.dataset.max_seqs = dataset["max_seqs"]
        self.model = self.build_model(self.cfg)
        num_params = sum(p.numel() for p in self.model.parameters())
        self.model.load_state_dict(model_state, strict=False)
        return self.model

    @torch.no_grad()
    def predict(self, output="prediction.jsonl"):
        """inference jsonl dataset->test_result.jsonl"""
        loader = self.build_dataloader(split="test")
        assert self.model is not None, f"load model first."
        training = self.model.training
        device = self.model.device
        self.model.eval()
        rank = comm.get_rank()
        jsonl_path = output.replace(".jsonl", f".rank{rank}.jsonl")
        builder = MMapIndexedJsonlDatasetBuilder(jsonl_path)
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader), desc="prediction", disable=not comm.is_rank_0())
        for i, (inputs, targets) in pbar:
            batch_x = to_device(inputs, device)
            outputs = self.model(**batch_x)
            logits = outputs["logits"]
            probs = logits.softmax(dim=-1)[..., 1]
            # pbar.set_description(f"{probs.mean().item()}")
            metas = targets["meta"]
            for prob, meta in zip(probs, metas):
                builder.add_item({
                    "id": meta.pop("id"),  # chr1:100
                    "chr": meta.pop("contig"),
                    "position": meta.pop("position"),
                    "probability": prob.item(),
                    "depth": meta.pop("depth"),
                    "meta": meta
                })
                builder.end_document()

        builder.finalize()
        comm.synchronize()
        size = comm.get_data_parallel_world_size()
        if comm.is_rank_0():
            # merge results
            filenames = [output.replace(".jsonl", f".rank{rank}.jsonl") for rank in range(size)]
            MMapIndexedJsonlDatasetBuilder.merge_files(filenames, output)

        self.model.train(training)
        return output


def main(args):
    bam_file = args.bam
    ref_file = args.ref
    save_dir = args.output
    if save_dir is None:
        save_dir = os.path.dirname(bam_file)
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger()
    num_workers = get_cpu_cores() * 0.8 if args.num_workers is None else args.num_workers
    num_workers = int(num_workers)
    candidate_file = f"{save_dir}/methylation_candidates.jsonl"
    overwrite = args.overwrite
    start = datetime.now()
    if comm.is_rank_0():
        logger.info(f"start {datetime.now()}")
        if overwrite or not os.path.exists(f"{candidate_file}.idx"):
            cg_sits = MethylationCaller.load_reference_cg_sits(args.ref_sit)
            logger.info(f"#CG sits: {len(cg_sits)}")
            logger.info("start filter cg sits")
            MethylationCaller.filter_cg_sits(cg_sits, bam_file, candidate_file, num_workers=num_workers)
        else:
            logger.info(f"exist candidate file: {candidate_file}")

    can_tbp = datetime.now()
    comm.synchronize()
    logger.info(f"candidates selection time: {(can_tbp - start).seconds / 60} minutes")
    cfg = get_config()
    # set candidate files
    cfg.dataset.candidate_files = [candidate_file, ]
    cfg.dataset.bam_file = bam_file
    cfg.dataset.ref_file = ref_file
    cfg.dataset.name = "MethylationCandidateDataset"
    cfg.dataset.with_info = not args.without_info
    # gpu inference lazy loading checkpoint
    world_size = comm.get_world_size()
    cfg.dataloader.num_workers = num_workers // world_size
    caller = MethylationCaller(cfg)
    caller.load_model(args.ckpt)
    if args.seq_len is not None:
        cfg.dataset.seq_len = int(args.seq_len)

    cfg.dataloader.batch_size = int(args.batch_size)
    cfg.dataloader.prefetch_factor = 2
    if get_torch_version() > (2, 2):
        is_bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
    else:
        is_bf16_supported = torch.cuda.is_bf16_supported()

    dtype = torch.bfloat16 if is_bf16_supported else torch.float16
    caller.to(dtype)
    logger.info(f"start prediction methylation sits...")
    caller.predict(f"{save_dir}/methylation_prediction.jsonl")
    gpu_time = datetime.now() - start
    logger.info(f"gpu prediction time: {gpu_time.seconds / 60} minutes")
    logger.info(f"finish inference")


if __name__ == "__main__":
    parser = default_argument_parser(epilog="cfDNA Methylation Calling")
    parser.add_argument("--ref", "-r", required=True,
                        metavar="FILE", help="path to reference file")
    parser.add_argument("--bam", "-b", required=True,
                        metavar="FILE", help="path to bam file")
    parser.add_argument("--ref_sit", "-rs",
                        default=f"sites/",
                        metavar="FILE", help="path to reference cancer candidate sits")
    parser.add_argument("--output", "-o", default=None,
                        help="path to output directory")
    parser.add_argument("--seq_len", "-sl", default=None, type=int, help="path to output directory")
    parser.add_argument("--max_seqs", "-ms", default=None, type=int, help="path to output directory")
    parser.add_argument("--batch_size", "-bs", default=196, type=int, help="number of batch size of dataloader")
    parser.add_argument("--num_workers", "-n", default=None, help="number of workers")
    parser.add_argument("--without_info", "-woi", action="store_true", help="without quality info input")
    parser.add_argument("--overwrite", action="store_true", help="overwrite results")
    args = parser.parse_args()
    print("Command Line Args:", args)
    cfg = get_config()
    save_dir = args.output
    if save_dir is None:
        save_dir = os.path.dirname(args.bam)
    cfg.log_dir = save_dir
    cfg.log_file = f"{save_dir}/methylation_calling.log"
    launch(main, args=args)

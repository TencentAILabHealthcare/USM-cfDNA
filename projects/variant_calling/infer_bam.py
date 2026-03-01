# Copyright (c) 2024, Tencent Inc. All rights reserved.
import os
import sys
sys.path.append(".")

from datetime import datetime
from tgnn.utils import jload


import tqdm
import torch
from functools import partial
from tgnn.config import get_config
from tgnn.engine import default_argument_parser, launch, Predictor
from tgnn.distributed import comm
from tgnn.utils.logger import get_logger
from tgnn.utils.io import to_device
from tgnn.utils.env import get_torch_version
from tgnn.sci.parser.sam_parsing import index_bam, is_sorted_alignment, make_vcf_header, has_chrY
from tgnn.sci.parser.vcf_parsing import merge_vcf
from tgnn.multiprocessing import get_cpu_cores
from projects.variant_calling.dataset.output_utils import calling_model_output_to_variant
from projects.variant_calling.dataset.output_writer import VariantWriter


class VariantCaller(Predictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.input_mode = "MSA"

    @property
    def writer(self):
        if self._writer is not None:
            return self._writer

        rank = comm.get_data_parallel_rank()
        if self.output.endswith('.vcf'):
            output = f"{self.output[:-4]}.rank{rank}.vcf"
        else:
            output = f"{self.output[:-7]}.rank{rank}.vcf.gz"

        if os.path.exists(f"{output}.tbi"):
            os.remove(f"{output}.tbi")

        output_to_variant = partial(calling_model_output_to_variant,
                                    input_mode=self.input_mode,
                                    snp_threshold=self.snp_threshold,
                                    indel_threshold=self.indel_threshold)
        self._writer = VariantWriter(output,
                                     header=self.header,
                                     process_fn=output_to_variant,
                                     worker_starmap=True,
                                     num_workers=4)

        return self._writer

    def make_writer(self, filename, output, snp_threshold=50, indel_threshold=50):
        assert output.endswith(("vcf", "vcf.gz")), f"only support vcf format but got {os.path.basename(output)}"
        header = make_vcf_header(
            filename,
            info=[
                {"id": "IM", "number": 1, "type": "String", "description": "Model Input Mode"}
            ],
            filters=[
                {"id": "RefCall", "description": "Reference Call"},
                {"id": "LowQual", "description": "Low Quality Variant"}
            ]
        )
        world_size = comm.get_data_parallel_world_size()
        if world_size > 1:
            rank = comm.get_data_parallel_rank()
            if output.endswith('.vcf'):
                output = f"{output[:-4]}.rank{rank}.vcf"
            else:
                output = f"{output[:-7]}.rank{rank}.vcf.gz"

            if os.path.exists(f"{output}.tbi"):
                os.remove(f"{output}.tbi")

        haploid_contigs = None
        if has_chrY(filename):
            haploid_contigs = ("chrX", "chrY")
            self.logger.info(f"Haploid contigs: {haploid_contigs}")

        fn = partial(calling_model_output_to_variant,
                     input_mode=self.input_mode,
                     snp_threshold=snp_threshold,
                     indel_threshold=indel_threshold,
                     haploid_contigs=haploid_contigs)

        return VariantWriter(output,
                             header=header,
                             process_fn=fn,
                             worker_starmap=True,
                             num_workers=2)

    @torch.no_grad()
    def predict(
            self,
            bam_file,
            ref_file,
            output,
            chrs=None,
            bed_file=None,
            snp_af=0.08,
            indel_af=0.08,
            min_depth=4,
            min_mapq=5,
            snp_quality_threshold=50,
            indel_quality_threshold=50,
            without_info=True
    ):
        cfg.dataset.bam_file = bam_file
        cfg.dataset.ref_file = ref_file
        cfg.dataset.bed_file = bed_file
        cfg.dataset.with_info = not without_info
        cfg.dataset.snp_af = snp_af
        cfg.dataset.indel_af = indel_af
        cfg.dataset.min_depth = min_depth
        cfg.dataset.min_mapq = min_mapq
        cfg.dataset.chrs = chrs
        loader = self.build_dataloader(split="test")
        writer = self.make_writer(bam_file, output, snp_quality_threshold, indel_quality_threshold)
        assert self.model is not None, f"load model first."
        training = self.model.training
        device = self.model.device
        self.model.eval()
        world_size = comm.get_data_parallel_world_size()
        pbar = tqdm.tqdm(enumerate(loader), desc="prediction", disable=not comm.is_rank_0())
        for i, (inputs, targets) in pbar:
            batch_x = to_device(inputs, device)
            outputs = self.model(**batch_x)
            pred_ats = outputs["at"].softmax(dim=-1).cpu().float().numpy()
            metas = targets["meta"]
            writer.batch_write(zip(pred_ats, metas))
        writer.close()
        comm.synchronize()
        if world_size > 1 and comm.is_rank_0():
            if self.output.endswith('.vcf'):
                outputs = [f"{self.output[:-4]}.rank{i}.vcf" for i in range(world_size)]
            else:
                outputs = [f"{self.output[:-7]}.rank{i}.vcf.gz" for i in range(world_size)]
            merge_vcf(outputs, self.output, sort=True, index=True)

        self.model.train(training)
        return output


def get_tasks(args):
    tasks = []
    logger = get_logger()
    ref_file = args.ref
    if ref_file is not None:
        assert os.path.exists(ref_file), f"{ref_file} does not exist"
        logger.info(f"reference fasta file: {ref_file}")

    bed_file = args.bed
    if bed_file is not None:
        assert os.path.exists(bed_file), f"{bed_file} does not exist"
        logger.info(f"bed file: {bed_file}")

    chrs = args.chrs or ["chr" + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    config = {
        "ref_file": ref_file,
        "bed_file": bed_file,
        "chrs": chrs,
        "snp_af": args.snp_af,
        "indel_af": args.indel_af,
        "min_depth": args.min_depth,
        "min_mapq": args.min_mapq,
        "snp_quality_threshold": args.snp_quality_threshold,
        "indel_quality_threshold": args.indel_quality_threshold,
        "without_info": args.without_info
    }
    bam_file = args.bam
    if bam_file.endswith((".bam", ".sam", ".cram")):
        assert os.path.exists(bam_file), f"{bam_file} does not exist"
        assert is_sorted_alignment(bam_file), f"{bam_file} must be a sorted alignment"
        index_bam(bam_file, exist_ok=True)
        if args.output is None:
            save_dir = os.path.dirname(bam_file)
            output = f"{save_dir}/variant_prediction.vcf"
        elif args.output.endswith((".vcf", "vcf.gz")):
            output = args.output
        else:
            save_dir = args.output
            output = f"{save_dir}/variant_prediction.vcf"
        tasks.append({
            "input": bam_file,
            "output": output,
            **config
        })
    elif bam_file.endswith(".csv"):
        logger.warning(f"skipping {bam_file} as it is not a CSV file")
        import csv
        with open(bam_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tasks.append(row)
    elif bam_file.endswith(".json"):
        data = jload(bam_file)
        config.update(data.get("config", {}))
        for row in data["tasks"]:
            # overwrite global setting
            task = config.copy().update(row)
            tasks.append(task)
    else:
        raise ValueError(f"input bam file must be either .bam or .sam or .csv")

    return tasks


def main(args):
    logger = get_logger()
    tasks = get_tasks(args)
    logger.info(f"without quality information: {args.without_info}")
    num_workers = get_cpu_cores() * 0.8 if args.num_workers is None else args.num_workers
    num_workers = int(num_workers)
    logger.info(f"number of workers: {num_workers}")
    cfg = get_config()
    cfg.dataset.name = "CandidateTensorIterDataset"
    world_size = comm.get_world_size()
    cfg.dataloader.num_workers = num_workers // world_size
    cfg.dataloader.batch_size = args.batch_size
    cfg.dataloader.prefetch_factor = 2
    caller = VariantCaller(cfg)
    caller.load_model(args.ckpt)

    if get_torch_version() > (2, 2):
        is_bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
    else:
        is_bf16_supported = torch.cuda.is_bf16_supported()

    dtype = torch.bfloat16 if is_bf16_supported else torch.float16
    logger.info(f"inference on dtype: {dtype}\tdevice: {caller.model.device}")
    caller.to(dtype)
    for task in tasks:
        start = datetime.now()
        bam_file = task["input"]
        ref_file = task["ref_file"]
        output = task["output"]
        logger.info(f"input: {bam_file}")
        logger.info(f"output: {output}")
        logger.info(f"reference: {ref_file}")
        bed_file = task["bed_file"]
        if bed_file is not None:
            logger.info(f"bed file: {bed_file}")
        chrs = task["chrs"]
        logger.info(f"chrs: {chrs}")
        logger.info("parameters:")
        snp_af = task.get("snp_af", 0.08)
        indel_af = task.get("indel_af", 0.08)
        min_mapq = task.get("min_mapq", 5)
        min_depth = task.get("min_depth", 4)
        logger.info(f"snp_af: {snp_af}\tindel_af: {indel_af}\tmin_map: {min_mapq}\tmin_depth :{min_depth}")
        snp_quality_threshold = task.get("snp_quality_threshold", 50)
        indel_quality_threshold = task.get("indel_quality_threshold", 50)
        logger.info(f"filter quality snp: {snp_quality_threshold}\tindel: {indel_quality_threshold}")
        without_info = task.get("without_info", False)
        logger.info(f"without_info: {without_info}")
        caller.predict(bam_file,
                       ref_file,
                       output,
                       chrs=chrs,
                       bed_file=bed_file,
                       snp_af=snp_af,
                       indel_af=indel_af,
                       min_depth=min_depth,
                       snp_quality_threshold=snp_quality_threshold,
                       indel_quality_threshold=indel_quality_threshold,
                       without_info=without_info
                       )
        total_time = datetime.now() - start
        logger.info(f"total time: {total_time.seconds / 60} minutes")

    logger.info(f"finish inference")


if __name__ == "__main__":

    chrs = [f"chr{i}" for i in range(1, 23)] + ['chrX', 'chrY']
    parser = default_argument_parser(epilog="Variant Calling")
    parser.add_argument("--bam", "-b", required=True, metavar="FILE", help="path to bam file")
    parser.add_argument("--ref", "-r", required=True, metavar="FILE", help="path to reference file")
    parser.add_argument("--chrs", "-c", nargs='+', default=chrs, help="chromosomes for processing")
    parser.add_argument("--bed", default=None, metavar="FILE", help="path to bed file")
    parser.add_argument("--min_depth", default=4, type=int, help="min depth converage")
    parser.add_argument("--min_mapq", default=5, type=int, help="min mapping quality")
    parser.add_argument("--snp_af", default=0.08, type=float, help="snp af of candidate filter")
    parser.add_argument("--indel_af", default=0.08, type=float, help="indel af of candidate filter")
    parser.add_argument("--snp_quality_threshold", default=50, type=float, help="snp af of pass filter")
    parser.add_argument("--indel_quality_threshold", default=50, type=float, help="indel af of pass filter")
    parser.add_argument("--output", "-o", required=True, help="path to output directory")
    parser.add_argument("--num_workers", "-n", default=None, help="number of workers")
    parser.add_argument("--batch_size", "-bs", default=256, type=int, help="number of batch size of dataloader")
    parser.add_argument("--without_info", "-woi", action="store_true", help="without quality info input")
    args = parser.parse_args()
    print("Command Line Args:", args)
    # change default log dir
    cfg = get_config()
    output = args.output
    if output is None:
        save_dir = os.path.dirname(args.bam)
    elif output.endswith(("vcf.gz", "vcf")):
        save_dir = os.path.dirname(output)
    else:
        save_dir = output
    cfg.log_dir = save_dir
    cfg.log_file = f"{save_dir}/variant_calling.log"
    launch(main, args=args)

# Copyright (c) 2025, Tencent Inc. All rights reserved.
import atexit
import multiprocessing
import queue
import sys
import time
from typing import Optional
import pysam
from pysam import VariantFile
from tgnn.data import MMapIndexedJsonlDatasetBuilder
from .output_utils import get_sit_output_from_at
from .quality_utils import compute_gq, quality_score_from


class CallingModelOutputProcess(multiprocessing.Process):

    def __init__(self,
                 output_queue,
                 output_path,
                 header=None,
                 indel_threshold=4,
                 snp_threshold=4,
                 input_mode="profile",
                 **kwargs):
        super().__init__(**kwargs)
        self.output_queue = output_queue
        self.output_path = output_path
        self.header = header
        self.indel_threshold = indel_threshold
        self.snp_threshold = snp_threshold
        self.shutdown_flag = multiprocessing.Event()
        self.input_mode = input_mode

    def output_to_variant(self, output, target) -> Optional[pysam.VariantRecord]:
        """Convert a variant dictionary to VariantRecord"""
        pred_gt = output["gt"]
        pred_at = output["at"]
        gq, pls = compute_gq(pred_gt, return_pl=True)
        alt_bases = target["alt_bases"]
        ref_base = target["ref_base"]
        prob, gt, at, alleles, allele_counts = get_sit_output_from_at(
            pred_gt,
            pred_at,
            ref_base,
            alt_bases
        )
        if alleles is None or gt == (0, 0):
            return None

        depth = sum(b[1] for b in alt_bases.items())
        qual = quality_score_from(prob)
        if len(alleles) == 2:
            pls = pls[:3]
        else:
            pls = (pls[0], pls[1], pls[2], 0, pls[3], 0)
        is_indel = "I" in at or "D" in at
        threshold = self.indel_threshold if is_indel else self.snp_threshold
        try:
            record = self.header.new_record(
                contig=target["chrom"],
                start=target["pos"],
                alleles=alleles,
                qual=qual,
                filter="PASS" if qual > threshold else "RefCall",
                samples=[{
                    "GT": gt,
                    "GQ": gq,
                    "DP": depth,
                    "PL": pls,
                    "AD": allele_counts,
                    "AF": [cnt / depth for cnt in allele_counts[1:]]
                }],
                phased=False
            )
            record.info["IM"] = "P" if self.input_mode in ("profile",) else "M"
        except Exception as e:
            print(f"Error converting variant dict to record: {e}")
            return None

    def run(self):
        # Open output file with the same header as input
        if self.output_path.endswith(("vcf.gz", "vcf", "bcf", "bcf.gz")):
            with VariantFile(self.output_path, "w", header=self.header) as vf:
                while not self.shutdown_flag.is_set() or not self.output_queue.empty():
                    try:
                        # Non-blocking get with timeout to check shutdown_flag
                        output, target = self.output_queue.get(timeout=0.1)
                        variant = self.output_to_variant(output, target)
                        if variant is None:
                            continue
                        vf.write(variant)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"writer error: {e}")
                        break
        else:
            assert self.output_path.endswith(".jsonl"), f"{self.output_path} must be a jsonl file or variant file."
            builder = MMapIndexedJsonlDatasetBuilder(self.output_path)
            while not self.shutdown_flag.is_set() or not self.output_queue.empty():
                try:
                    # Non-blocking get with timeout to check shutdown_flag
                    output, target = self.output_queue.get(timeout=0.1)
                    variant = self.output_to_variant(output, target)
                    if variant is None:
                        continue
                    builder.add_item(variant)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"writer error: {e}")
                    break


class VariantWriter:
    def __init__(self,
                 output_path,
                 header=None,
                 process_fn=None,
                 write_fn=None,
                 num_workers=1,
                 worker_starmap=False,
                 writer_startmap=False,
                 max_size=10240):
        self.output_path = output_path
        self.input_queue = multiprocessing.Queue(maxsize=max_size)
        self.write_queue = multiprocessing.Queue(maxsize=max_size)
        self.process_fn = process_fn
        self.write_fn = write_fn
        self.num_workers = num_workers
        self.worker_starmap = worker_starmap
        self.writer_startmap = writer_startmap
        self.shutdown_flag = multiprocessing.Event()
        # Register cleanup at exit
        atexit.register(self.stop)
        self.header = header
        self.writer_process = None
        self._create_process()
        self._start_process()

    def _create_process(self):
        self.workers = [multiprocessing.Process(target=self._work_process, daemon=True) for _ in
                        range(self.num_workers)]
        self.writer_process = multiprocessing.Process(target=self._write_process, daemon=True)

    def _start_process(self):
        self.writer_process.start()
        for worker in self.workers:
            worker.start()

    def _work_process(self):
        while not self.shutdown_flag.is_set():
            try:
                data = self.input_queue.get(timeout=0.1)
                if self.worker_starmap:
                    output = self.process_fn(*data)
                else:
                    output = self.process_fn(data)
                self.write_queue.put(output)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}", file=sys.stderr)

    def _write_process(self):
        with VariantFile(self.output_path, "w", header=self.header) as vf:
            while not self.shutdown_flag.is_set():
                try:
                    data = self.write_queue.get(timeout=0.1)
                    if self.write_fn is not None:
                        if self.writer_startmap:
                            data = self.write_fn(*data)
                        else:
                            data = self.write_fn(data)
                    if data is None:
                        continue

                    if isinstance(data, pysam.VariantRecord):
                        vf.write(data)
                    else:
                        vf.write(self.header.new_record(**data))

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"writer error: {e}", file=sys.stderr)
                    break

    def batch_write(self, batches, **kwargs):
        return [self.write(x, **kwargs) for x in batches]

    def write(self, data, block=False, timeout=None):
        try:
            if block:
                self.input_queue.put(data, block=True, timeout=timeout)
            else:
                self.input_queue.put_nowait(data)
            return True
        except queue.Full:
            if not block:
                print("Warning: Variant queue full, record dropped")
            return False

    def flush(self):
        while not self.input_queue.empty():
            time.sleep(0.01)

        while not self.write_queue.empty():
            time.sleep(0.01)

    def close(self):
        self.flush()
        self.stop()

    def clean_queue(self, q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        self.shutdown_flag.set()
        for worker in self.workers:
            worker.join()
        self.writer_process.join()
        if self.input_queue.qsize() > 0:
            print(f"Warning: Variant queue empty, record dropped: {self.input_queue.qsize()}")
        if self.write_queue.qsize() > 0:
            print(f"Warning: Variant queue empty, record dropped: {self.input_queue.qsize()}")
        self.clean_queue(self.input_queue)
        self.clean_queue(self.write_queue)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Copyright (c) 2025, Tencent Inc. All rights reserved.
import subprocess
import sys

sys.path.append(".")

import os
from functools import partial
import argparse
from tgnn.sci.parser.sam_parsing import parse_alignment, get_reference_length
from tgnn.sci.parser.wig_parsing import FixStepCounter, parse_wig
from tgnn.multiprocessing import process_map, get_cpu_cores


def count_read(bam_file, chrom, start=None, end=None, window=1000_000, min_mapq=30):
    af = parse_alignment(bam_file)
    start = start or 0
    if end is None:
        end = af.get_reference_length(chrom)

    assert end <= af.get_reference_length(chrom), f"end must bed less than {af.get_reference_length(chrom)}"
    counter = FixStepCounter(chrom, window, start=start, end=end)
    for read in af.fetch(chrom, start, end):
        if (
                read.mapping_quality < min_mapq or
                read.is_duplicate or
                read.is_secondary or
                read.is_unmapped or
                read.is_qcfail or
                read.is_supplementary
        ):
            continue

        pos = read.reference_start
        if pos < start or pos >= end:
            continue

        counter.update(pos)
    return counter


def count_read_shell(bam_file, chrom, start=None, end=None, window=1000_000, min_mapq=30, readCounter="readCounter"):
    cmds = [readCounter, "--window", str(window), "--quality", str(min_mapq), "--chromosome", chrom, bam_file]
    print(" ".join(cmds))
    start = start or 0
    assert start == 0, f"start must be 0"
    p = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print(stdout.decode("utf-8"))
        print(stderr.decode("utf-8"))
        raise Exception(f"failed with return code {p.returncode}")

    return stdout.decode("utf-8")


def parse_count_read_shell(bam_file, chrom, start=None, end=None, window=1000_000, min_mapq=30,
                           readCounter="readCounter"):
    wig_string = count_read_shell(bam_file, chrom, start, end, window, min_mapq, readCounter)
    return FixStepCounter.from_string(wig_string)[chrom]


def read_counter(bam_file, chrs=None, window=1000, min_mapq=30, num_workers=None):
    """
    Args:
        bam_file: input bam file
        wig_file: output wig file
    """
    assert bam_file.endswith((".bam", ".sam")), f"{bam_file} is not a sam or bam file"
    af = parse_alignment(bam_file)
    if chrs is None:
        if af.get_reference_name(0).startswith("chr"):
            chrs = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        else:
            chrs = [f"{i}" for i in range(1, 23)] + ["X", "Y"]
    print("chromosomes:", chrs)
    for chrom in chrs:
        assert af.get_tid(chrom) >= 0, f"bam file {bam_file} does not contain {chrom}"

    # fn = partial(count_read, window=window, min_mapq=min_mapq)
    fn = partial(parse_count_read_shell, window=window, min_mapq=min_mapq)
    tasks = [(bam_file, chrom) for chrom in chrs]
    num_workers = get_cpu_cores() if num_workers is None else int(num_workers)
    num_procs = min(num_workers, len(tasks))
    print("num_procs:", num_procs)
    counters = []
    for (i, counter) in process_map(fn, enumerate(tasks), num_procs=num_procs, starmap=True):
        counters.append(counter)

    counters.sort(key=lambda c: (int(c.chrom.replace("chr", "")), c.start))
    return counters


def main(args):
    assert os.path.exists(args.input), f"{args.input} does not exist"
    if args.window.isdigit():
        window = int(args.window)
    elif args.window.lower().endswith("kb"):
        window = int(args.window[:-2]) * 1000
    elif args.window.lower().endswith("mb"):
        window = int(args.window[:-2]) * 1000000
    else:
        raise ValueError(f"{args.window} is not a valid window type")
    assert window > 0, f"window must be greater than 0"

    if args.output is None:
        wig_file = f"{args.input[:-4]}.{args.window.upper()}.Q{args.min_mapq}.wig"
    else:
        wig_file = args.output
    os.makedirs(os.path.dirname(wig_file), exist_ok=True)
    print("output wig file:", wig_file)
    counters = read_counter(args.input,
                            args.chrs,
                            window=window,
                            min_mapq=args.min_mapq,
                            num_workers=args.num_threads)
    wf = open(wig_file, "w")
    for counter in counters:
        wf.write(counter.to_string())
        wf.write("\n")
    wf.close()
    print(f"Successfully generated WIG file: {os.path.basename(wig_file)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate WIG file from BAM read coverage.')
    parser.add_argument('-i', '--input', required=True, help='Input bam or cram file')
    parser.add_argument('-o', '--output', default=None, help='Output WIG file (default: input name with wig extension)')
    parser.add_argument("--chrs", "-c", nargs='+', default=None, help="chromosomes for processing")
    parser.add_argument('-w', '--window', type=str, default="1MB", help='window size (default: 1000KB)')
    parser.add_argument('-m', '--min_mapq', type=int, default=30, help='minimum mapping quality (default: 30)')
    parser.add_argument('-t', '--num_threads', type=int, default=None, help='number of threads')
    args = parser.parse_args()
    main(args)

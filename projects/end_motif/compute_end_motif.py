# Copyright (c) 2025, Tencent Inc. All rights reserved.
import argparse
import os
import sys
from typing import Sequence, Union

sys.path.append(".")

import time

import numpy as np
import pysam
from collections import defaultdict
from intervaltree import IntervalTree, Interval
from functools import lru_cache
from tgnn.multiprocessing import process_map, get_cpu_cores
from tgnn.utils.io import jdump
from tgnn.sci.parser.sam_parsing import get_nm_value
from tgnn.utils.pack_files import open_resource_text

motif_types = ['ACG', 'TCG', 'CGT', 'CGA', 'GCG', 'CCG', 'CGG', 'CGC']


@lru_cache(maxsize=None)
def rev_comp(seq):
    return seq[::-1].translate(str.maketrans("ACGT", "TGCA"))


def infer_strand_by_flag(read1, read2):
    # READ1(83): FREAD1(64) + FREVERSE(16) + FPAIRED(1) + FPROPER_PAIR(2)
    # READ2(163): FREAD2(128) + FMREVERSE(32) + FPAIRED(1) + FPROPER_PAIR(2)

    # READ1(83): FREAD1(64) + FMREVERSE(32) + FPAIRED(1) + FPROPER_PAIR(2)
    # READ2(163): FREAD2(128) + FREVERSE(32) + FPAIRED(1) + FPROPER_PAIR(2)
    flags = {read1.flag, read2.flag}
    if flags == {99, 147}:
        return 'W'
    elif flags == {83, 163}:
        return 'C'
    else:
        return None


def parsing_bed(filename, resource_name, chrs=None, parse_header=True):
    """0-based interval tree [start, end)

    Args:
        filename: str, path to bed file
        chrs: list of chromosomes
    """
    tree = defaultdict(IntervalTree)
    fields = None
    with open_resource_text(filename, resource_name) as f:
        for line in f:
            if line.startswith(("browser", "track")):
                continue

            line = line.strip()
            if line.startswith("#"):
                if parse_header:
                    fields = line[1:].split()
                continue

            chrom, start, end, *other = line.split()
            if chrs is not None:
                if chrom not in chrs:
                    continue

            start = int(start)
            end = int(end)
            if start >= end:
                raise ValueError("start must be before end")

            if fields is not None and len(fields) == len(other) + 3:
                data = dict(zip(fields[3:], other))
            else:
                data = other

            tree[chrom].addi(start, end, data)

    return tree


def compute_bed_coverage(refs: IntervalTree, fragments: IntervalTree):
    """compute fragment, 5'end, 3'end profile"""
    bed = IntervalTree()
    for intva in refs:
        intvbs = fragments.overlap(intva)
        length = intva.length()  # 12BP
        profiles = np.zeros((length,), dtype=np.int32)
        watsons = np.zeros((length,), dtype=np.int32)
        cricks = np.zeros((length,), dtype=np.int32)
        for intvb in intvbs:
            begin = max(intvb.begin, intva.begin)
            end = min(intvb.end, intva.end)
            profiles[begin - intva.begin:end - intva.begin] += 1
            end5 = intvb.begin
            end3 = intvb.end - 1
            if intva.begin <= end5 < intva.end:
                watsons[end5 - intva.begin] += 1

            if intva.begin <= end3 < intva.end:
                cricks[end3 - intva.begin] += 1

        data = {
            **intva.data,
            "profile": profiles,
            "watson": watsons,
            "crick": cricks
        }
        bed.addi(intva.begin, intva.end, data)

    return bed


def compute_read_profile(
        bam_path,
        bed_path,
        resource_name,
        chrom,
        start=None,
        end=None,
        min_mapq=30,
        filter_clip=True
):
    fail_counter = defaultdict(int)
    fragments = IntervalTree()
    pair_buffer = {}
    with pysam.AlignmentFile(bam_path, "rb") as af:
        for read in af.fetch(chrom, start=start, end=end):
            # 1) unmapped / mate unmapped
            if read.is_unmapped or read.mate_is_unmapped or not read.is_paired:
                fail_counter["unmapped_or_unpaired"] += 1
                continue

            # 2) QC & DUPLICATION & Sencodary & Supplementary
            if read.is_qcfail or read.is_secondary or read.is_duplicate or read.is_supplementary:
                fail_counter["qc_secondary_duplicate_supplementary"] += 1
                continue

            # 3) mapping quality
            if read.mapping_quality < min_mapq:
                fail_counter["low_mapq"] += 1
                continue

            # 4) mismatch
            if get_nm_value(read) > 2:
                fail_counter["mismatches"] += 1
                continue

            # 5) non autosome
            if not read.reference_name.startswith("chr") or not read.reference_name[3:].isdigit():
                fail_counter["non_autosome"] += 1
                continue

            # 6) soft clip
            if filter_clip:
                cigars = read.cigartuples
                if cigars[0][0] in (pysam.CHARD_CLIP, pysam.CSOFT_CLIP) or cigars[-1][0] in (pysam.CHARD_CLIP,
                                                                                             pysam.CSOFT_CLIP):
                    fail_counter["read_clip"] += 1
                    continue

            qname = read.query_name
            if qname.endswith(("/1", "/2")):
                qname = qname[:-2]

            if qname in pair_buffer:
                if read.is_read2:
                    read1, read2 = pair_buffer.pop(qname), read
                else:
                    read1, read2 = read, pair_buffer.pop(qname)

                strand = infer_strand_by_flag(read1, read2)
                if strand is None:
                    fail_counter["failed_strand"] += 1
                    continue

                if read1.is_forward:
                    pos5 = read1.reference_start
                    pos3 = read2.reference_start + read2.query_length
                else:
                    pos5 = read2.reference_start
                    pos3 = read1.reference_start + read1.query_length

                if pos5 >= pos3:
                    continue

                fragments.append(Interval(pos5, pos3))
            else:
                pair_buffer[qname] = read

    ref_bed = parsing_bed(bed_path, resource_name, [chrom, ])[chrom]
    bed = compute_bed_coverage(ref_bed, fragments)

    return chrom, bed, fail_counter


def compute_feat(bed: Union[IntervalTree, Sequence[Interval]]) -> dict:
    fragments = np.stack([i.data["profile"] for i in bed], axis=0).sum(axis=0)
    profiles = fragments + fragments[::-1]
    watson_ends = np.stack([i.data["watson"] for i in bed], axis=0).sum(axis=0)
    crick_ends = np.stack([i.data["crick"] for i in bed], axis=0).sum(axis=0)
    motif_counts = {m: 0 for m in motif_types}
    for intervel in bed:
        data = intervel.data
        if data["NCG"] in motif_types:
            motif_counts[data["NCG"]] += data["watson"][4]

        if data["CGN"] in motif_types:
            motif_counts[data["CGN"]] += data["watson"][5]

        cgn_reverse = rev_comp(data["NCG"])
        if cgn_reverse in motif_counts:
            motif_counts[cgn_reverse] += data["crick"][6]

        ncg_reverse = rev_comp(data["CGN"])
        if ncg_reverse in motif_counts:
            motif_counts[ncg_reverse] += data["crick"][7]

    return {
        "profiles": profiles.tolist(),
        "watson_ends": watson_ends.tolist(),
        "crick_ends": crick_ends.tolist(),
        "motifs": motif_counts
    }


def group_region(bed_tree):
    """
    Note that create large IntervalTree(set) is slowly
    """
    bed_groups = defaultdict(list)
    for chrom in bed_tree:
        for region in bed_tree[chrom].all_intervals:
            cancers = region.data["cancer"].split(",")
            for cancer in cancers:
                group = cancer + "_" + chrom + "_" + region.data["type"]
                bed_groups[group].append(region)

    return bed_groups


def main(args):
    bam_file = args.input
    assert os.path.exists(bam_file), f"{bam_file} does not exist"
    print(f"input alignment file: {bam_file}")
    bed_file = args.bed
    resource_name = args.resource_name
    assert os.path.exists(bed_file), f"{bed_file} does not exist"
    output = args.output
    if args.output is None:
        save_dir, basename = os.path.split(bam_file)
        output = f"{save_dir}/{basename.split('.')[0]}.json"
    print(f"output file: {output}")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    num_workers = get_cpu_cores() if args.num_threads is None else args.num_threads
    chrs = args.chrs
    if chrs is None:
        chrs = [f"chr{i}" for i in range(1, 23)]

    print("chromosomes:", chrs)
    tasks = [(bam_file, bed_file, resource_name, c) for c in chrs]

    start_time = time.time()
    print(f"1. compute read profile")
    num_procs = min(num_workers, len(tasks))
    print("#workers:", num_procs)
    bed_tree = {}
    fail_counter = defaultdict(int)
    for i, (chrom, bed, counter) in process_map(compute_read_profile,
                                                enumerate(tasks),
                                                num_procs=num_procs,
                                                starmap=True):
        print(f"finish computing {chrom} profile, number of sits: {len(bed)}")
        bed_tree[chrom] = bed
        for fname, fcount in counter.items():
            fail_counter[fname] += fcount

    for name, count in fail_counter.items():
        print(f"{name}: {count}")

    step1_time = time.time() - start_time
    print(f"Step1 Time：{step1_time:.4f} s")
    print("2. compute group feature")
    groups = group_region(bed_tree)
    print("#groups:", len(groups))
    num_procs = min(num_workers, len(groups))
    print("#workers:", num_procs)
    names = list(groups.keys())
    beds = [(groups[n],) for n in names]
    features = {}
    for i, feat in process_map(compute_feat,
                               enumerate(beds),
                               num_procs=num_procs,
                               starmap=True):
        name = names[i]
        features[name] = feat

    jdump(features, output)
    step2_time = time.time() - start_time - step1_time
    print(f"Step 2 Time：{step2_time:.4f} s")
    print("finish computing end motif")
    print(f"Total Time: {time.time() - start_time:.4f} s")
    print(f"output file: {output}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Compute End Motif")
    parser.add_argument('-i', '--input', required=True, help='Input bam file')
    parser.add_argument('-r', '--resource_name', required=True, help='reference cg bed file')
    parser.add_argument('-b', '--bed', required=True, help='reference cg bed file')
    parser.add_argument("--chrs", "-c", nargs='+', default=None, help="chromosomes for processing")
    parser.add_argument('-o', '--output', default=None, help='Path to output json')
    parser.add_argument('-t', '--num_threads', type=int, default=None, help='number of threads')
    parser.add_argument('-q', '--min_maq', type=int, default=30, help='minimum mapping quality')
    main(parser.parse_args())

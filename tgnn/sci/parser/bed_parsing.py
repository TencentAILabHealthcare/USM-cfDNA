# Copyright (c) 2024, Tencent Inc. All rights reserved.

import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict
import numpy as np
from intervaltree import IntervalTree


def region_to_tree(regions,
                   contig=None,
                   start=None,
                   end=None):
    """0-based interval tree [start, end)
    Args:
        filename: str, path to bed file
        contig: str or None, bed contig name
    """
    tree = defaultdict(IntervalTree)
    for (ctg_name, ctg_start, ctg_end) in regions:
        if ctg_name != contig:
            continue

        if start and end:
            if ctg_end < start or ctg_start > end:
                continue

        if ctg_start == ctg_end:
            ctg_end += 1

        tree[ctg_name].addi(ctg_start, ctg_end)

    return tree


def bed_to_tree(filename,
                contig=None,
                start=None,
                end=None):
    """0-based interval tree [start, end), note that bed file is 0-based index
    Args:
        filename: str, path to bed file
        contig: str or None, bed contig name
    """
    tree = defaultdict(IntervalTree)
    bed_start, bed_end = float('inf'), 0
    for row_id, row in enumerate(open(filename, mode="r").readlines()):
        if row.startswith(('#', 'track', 'browser')) or row.strip() == '':
            continue

        columns = row.strip().split()
        ctg_name = columns[0]
        if not ctg_name.startswith("chr"):
            ctg_name = f"chr{ctg_name}"

        ctg_start, ctg_end = int(columns[1]), int(columns[2])

        if contig is not None and ctg_name != contig:
            continue

        if ctg_end < ctg_start or ctg_start < 0 or ctg_end < 0:
            raise ValueError(f"[ERROR] Invalid bed input in {row_id}-th row {ctg_name} {ctg_start} {ctg_end}")

        if start and end:
            if ctg_end < start or ctg_start > end:
                continue

        bed_start = min(ctg_start, bed_start)
        bed_end = max(ctg_end, bed_end)

        if ctg_start == ctg_end:
            ctg_end += 1

        tree[ctg_name].addi(ctg_start, ctg_end)

    return tree


def region2string(contig, start=None, end=None):
    """python format [0-based, 0-based) to samtools format [1-based, 1-based]
    half-open intervals
    """
    assert contig.startswith("chr"), f"{contig} is invalid chromosome name"
    if start is None and end is None:
        return contig

    if start is None and end is not None:
        return f"{contig}:1-{end}"

    assert end is not None, f"invalid region {contig}:{start}-{end}"
    return "{}:{}-{}".format(contig, start + 1, end)


def string2region(region):
    """samtools format region to python format"""
    contig, rs = region.split(":")
    assert contig.startswith("chr"), f"{contig} is invalid chromosome name"
    if not rs:
        return contig, None, None

    start = None
    end = None
    rs = rs.split("-")
    assert 1 <= len(rs) <= 2, f"{region} is invalid region"

    if len(rs) >= 1:
        start = int(rs[0]) - 1

    if len(rs) == 2:
        end = int(rs[1])
        assert end >= start, f"region start: {start} must less than end: {end}"

    return contig, start, end


def is_interval_in(tree: IntervalTree, start: int = None, end: int = None):
    if hasattr(tree, 'at'):
        return len(
            tree.at(start)
            if end is None else
            tree.overlap(begin=start, end=end)
        ) > 0
    # intervaltree version 2
    return len(tree.search(begin=start, end=end, strict=False)) > 0


def is_region_in(tree: Dict[str, IntervalTree],
                 contig: str,
                 start: int = None,
                 end: int = None):
    if not tree or (contig is None) or (contig not in tree):
        return False

    itree = tree[contig]
    start = start or 0
    return is_interval_in(itree, start, end)


def fetch_region(bed_file,
                 contig=None,
                 start=None,
                 end=None):
    if Path(bed_file).suffix == "gz":
        open_fn = gzip.GzipFile
    else:
        open_fn = open

    regions = []
    with open_fn(bed_file, mode="r") as f:
        for line in f.readlines():
            if line[0] == "#":
                continue
            ctg, ctg_start, ctg_end = line.strip().split()
            ctg_start, ctg_end = int(ctg_start), int(ctg_end)
            if contig is not None and ctg != contig:
                continue

            if start and ctg_end < start:
                continue

            # note that bed file, postion is ascending order
            if end and ctg_start > end:
                break

            regions.append((contig, start, end))

    return regions


def sorted_region(regions, reverse=False, chrs=None):
    chrs = chrs or [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    def sort_fn(s):
        chr = chrs.index(s[0])
        start = s[1]
        return chr, start

    return sorted(regions, key=sort_fn, reverse=reverse)


def padding_region(regions, padding=33, sorted=True):
    if sorted:
        regions = sorted_region(regions)

    pre_end, pre_start = -1, -1
    extend_regions = []
    for (contig, start, end) in regions:
        if pre_start == -1:
            pre_start = start - padding
            pre_end = end + padding
            continue

        if pre_end >= start - padding:
            pre_end = end + padding
            continue
        else:
            extend_regions.append((contig, pre_start, pre_end))
            pre_start = start - padding
            pre_end = end + padding

    return extend_regions


def bed_coverage(bed1: IntervalTree, bed2: IntervalTree):
    bed = IntervalTree()
    for intva in bed1:
        intvbs = bed2.overlap(intva)
        mask = np.zeros((intva.end - intva.begin), dtype=np.int32)
        for intvb in intvbs:
            begin = max(intvb.begin, intva.begin)
            end = min(intvb.end, intva.end)
            mask[begin:end] += 1

        data = {
            **intva.data,
            "profile": mask
        }
        bed.addi(intva.begin, intva.end, data)

    return bed

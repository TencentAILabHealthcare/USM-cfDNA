# Copyright (c) 2024, Tencent Inc. All rights reserved.

from datetime import datetime
import os
import pysam
import itertools
from tgnn.utils import is_tool
from typing import Union, Iterator
from ..constants import base_constants as bc
from .bed_parsing import is_region_in, bed_to_tree


def parse_variant(filename, threads=0):
    if isinstance(filename, str):
        return pysam.VariantFile(filename, "r", threads=threads)

    return filename


def get_header(filename):
    return parse_variant(filename).header


def get_variants(vf: Union[pysam.VariantFile, str],
                 contig: str = None,
                 start=None,
                 end=None,
                 bed_tree=None,
                 reopen=False,
                 **kwargs
                 ) -> Iterator[pysam.VariantRecord]:
    """get variant records from vcf file
    """
    vf = parse_variant(vf, **kwargs)
    # Record (CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT, sample1, sample2,..)
    variants = vf.fetch(contig=contig, start=start, end=end, reopen=reopen)
    if bed_tree is not None:
        if isinstance(bed_tree, str):
            bed_tree = bed_to_tree(bed_tree, contig=contig, start=start, end=end)

        variants = [var for var in variants if is_region_in(bed_tree, var.chrom, var.pos - 1)]

    return variants


def variants_to_dict(variants: Iterator[pysam.VariantRecord]):
    variant_dict = {}
    for variant in variants:
        # always 0-based positions
        name = f"{variant.chrom}:{variant.pos - 1}"
        variant_dict[name] = variant

    return variant_dict


def sorted_variant(variants, reverse=False, chrs=None):
    if len(variants) == 0:
        return []

    var = variants[0]
    if isinstance(var, pysam.VariantRecord) or hasattr(var, "chrom"):
        long_chr = var.chrom.startswith("chr")
    else:
        chrom = var["chrom"] if "chrom" in var else var["chr"]
        long_chr = chrom.startswith("chr")

    if chrs is None:
        if long_chr:
            chrs = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        else:
            chrs = [f"{i}" for i in range(1, 23)] + ["X", "Y"]

    def sort_fn(var):
        if isinstance(var, pysam.VariantRecord) or hasattr(var, "chrom"):
            chrom = chrs.index(var.chrom)
            start = var.pos
        else:
            if "chrom" in var:
                chr_name = "chrom"
            elif "contig" in var:
                chr_name = "contig"
            else:
                chr_name = "chr"
            chrom = var[chr_name]
            chrom = chrs.index(chrom)
            start = var["pos"] if "pos" in var else var["position"]
        return chrom, start

    return sorted(variants, key=sort_fn, reverse=reverse)


def get_genotype(variant: pysam.VariantRecord, index=-1):
    samples: pysam.VariantRecordSamples = variant.samples
    sample: pysam.VariantRecordSample = samples.values()[index]
    gt1, gt2 = sample['GT']
    if gt1 is None or gt2 is None:
        return gt1, gt2

    if int(gt1) > int(gt2):
        gt1, gt2 = gt2, gt1

    return gt1, gt2


def genotype_to_index(gt1, gt2=None):
    if isinstance(gt1, (list, tuple)):
        assert gt2 is None
        gt1, gt2 = gt1

    if gt1 > gt2:
        gt2, gt1 = gt1, gt2

    gt = f"{gt1}/{gt2}"
    assert gt in bc.genotypes, f"{gt} not in 0/0, 1/1, 0/1, 1/2"

    return bc.genotypes.index(gt)


def get_info(variant: pysam.VariantRecord, key, default=None):
    info: pysam.VariantRecordInfo = variant.info
    return info.get(key, default)


def index_vcf(filename, force=False, **kwargs):
    if is_tool("tabix"):
        cmd = ["tabix", "-p", "vcf"]
        if force:
            cmd.append("-f")
        cmd.append(filename)
        cmd = " ".join(cmd)
        print(f"CMD: {cmd}")
        os.system(cmd)
    else:
        pysam.tabix_index(filename, force=force, preset="vcf", **kwargs)


def merge_vcf(filenames, output, sort=False, index=False, **kwargs):
    assert len(filenames) >= 1
    header = get_header(filenames[0])
    with pysam.VariantFile(output, mode="w", header=header) as f:
        variants = [get_variants(path) for path in filenames]
        variants = itertools.chain.from_iterable(variants)
        if sort:
            variants = sorted_variant(list(variants), **kwargs)

        for var in variants:
            f.write(var)

    if index:
        index_vcf(output, force=True)
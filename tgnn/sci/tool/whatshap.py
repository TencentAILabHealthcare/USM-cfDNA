# Copyright (c) 2025, Tencent Inc. All rights reserved.

import os
import sys
from functools import partial

import numpy as np
import pysam
from whatshap.cli.haplotag import (
    run_haplotag, VcfReader, PhasedInputReader, NumericSampleIds, load_chromosome_variants,
    prepare_haplotag_information
)
from whatshap.cli.phase import run_whatshap
from whatshap.cli.unphase import run_unphase

from tgnn.multiprocessing import process_map
from tgnn.sci.parser.sam_parsing import get_platform, merge_bam_files, reference_startswith_chr
from tgnn.sci.parser.sam_parsing import get_sample_name, index_bam
from tgnn.sci.parser.vcf_parsing import get_variants, get_genotype, parse_variant, sorted_variant, index_vcf, merge_vcf
from .longphase import LongPhase


def unphase(vcf_file, output):
    run_unphase(vcf_file, output)


whatshap_unphase = unphase


def whatshap_phase(bam_file,
                   vcf_file,
                   reference,
                   output=sys.stdout,
                   chromosomes=None,
                   **kwargs):
    if isinstance(bam_file, str):
        bam_file = [bam_file, ]

    if isinstance(chromosomes, str):
        chromosomes = [chromosomes, ]

    run_whatshap(bam_file,
                 vcf_file,
                 reference=reference,
                 output=output,
                 chromosomes=chromosomes,
                 **kwargs)


phase = whatshap_phase


def whatshap_haplotag(variant_file,
                      alignment_file,
                      output=None,
                      reference=False,
                      regions=None,
                      **kwargs):
    if isinstance(regions, str):
        regions = [regions, ]

    run_haplotag(variant_file, alignment_file, output, reference, regions,
                 **kwargs)


def parse_haplotag(
        alignment_file,
        variant_file,
        reference,
        contig,
        start=None,
        end=None,
        ploidy: int = 2
):
    vcf_reader = VcfReader(variant_file, only_snvs=False, phases=True, ploidy=ploidy)
    sample = get_sample_name(alignment_file)
    phased_input_reader = PhasedInputReader(
        [alignment_file, ],
        reference,
        NumericSampleIds(),
        ignore_read_groups=True,
        only_snvs=False,
        duplicates=True,
    )
    if phased_input_reader.has_alignments and reference is None:
        raise ("A reference FASTA needs to be provided with -r/--reference; "
               "or use --no-reference at the expense of phasing quality.")

    regions = [(start, end), ]
    variant_table = load_chromosome_variants(vcf_reader, contig, regions)
    read_to_haplotype = prepare_haplotag_information(
        variant_table,
        [sample, ],
        phased_input_reader,
        regions,
        ignore_linked_read=True,
        linked_read_cutoff=50000,
        ploidy=ploidy
    )[1]

    return {str(k.read_name): v for k, v in read_to_haplotype.items()}


haplotag = whatshap_haplotag


def select_het_variant(vcf_file,
                       contig,
                       output,
                       ratio=None,
                       only_snp=False,
                       threshold=90):
    """select high quality heterozygosity snp for phasing"""
    vf = parse_variant(vcf_file)
    variants = get_variants(vf, contig)
    total_het_variants = 0
    hets = []
    for var in variants:
        gt = get_genotype(var)
        if gt not in [(0, 1), (1, 2)]:
            continue

        total_het_variants += 1
        if var.qual < threshold:
            continue

        alleles = var.alleles
        ref_base, alt_bases = alleles[0], alleles[1:]
        if only_snp:
            if len(ref_base) != 1 and len(alt_bases) != 1:
                continue

        hets.append((var, var.qual))

    hets.sort(key=lambda x: -x[1])  # inplace sort
    num_variants = len(hets)
    if ratio is not None:
        num_variants = int(num_variants * ratio)

    cutoff = hets[num_variants - 1][1]
    quals = np.array([qual for var, qual in hets[:num_variants]])
    variants = [var for var, qual in hets[:num_variants]]
    with pysam.VariantFile(output, mode="w", header=vf.header) as writer:
        for var in sorted_variant(variants):
            writer.write(var)

    if isinstance(vcf_file, str):
        vf.close()

    print(f"{contig} select {num_variants} variants for phasing, "
          f"mean quality: {np.mean(quals)}, "
          f"max quality: {np.max(quals)}, "
          f"quality cutoff: {cutoff}, "
          f"ratio: {len(variants) / total_het_variants}")

    return output


def phase_haplotag_chrom(contig,
                         bam_file,
                         vcf_file,
                         ref_file,
                         output,
                         phaser="whatshap",
                         unphase_vcf=False,
                         phase_with_het=False,
                         phasing_ratio=0.8,
                         overwrite=False):
    os.makedirs(output, exist_ok=True)
    phased_bam_file = f"{output}/phased.haplotag.{contig}.bam"
    if not overwrite and os.path.exists(f"{phased_bam_file}.bai"):
        return phased_bam_file

    if unphase_vcf:
        whatshap_unphase(vcf_file, f"{output}/input.vcf.gz")
        vcf_file = f"{output}/input.vcf.gz"

    if phase_with_het:
        het_file = f"{output}/het.{contig}.vcf.gz"
        select_het_variant(vcf_file, contig, het_file, ratio=phasing_ratio)
        index_vcf(het_file, force=True)
        vcf_file = het_file

    print(f"phasing vcf: {vcf_file}")
    phased_vcf_file = f"{output}/phased.{contig}.vcf.gz"
    if phaser == "whatshap":
        whatshap_phase(bam_file,
                       vcf_file,
                       reference=ref_file,
                       output=phased_vcf_file,
                       chromosomes=contig,
                       ignore_read_groups=True,
                       distrust_genotypes=True)
    elif phaser == "longphase":
        LongPhase(ref_file).phase(bam_file, vcf_file, output=phased_vcf_file)
    else:
        raise f"no phaser name {phaser}"

    index_vcf(phased_vcf_file, force=True)
    print(f"phasing bam({os.path.basename(bam_file)}) with vcf({os.path.basename(phased_vcf_file)})")
    whatshap_haplotag(phased_vcf_file,
                      bam_file,
                      reference=ref_file,
                      output=phased_bam_file,
                      regions=contig,
                      ignore_read_groups=True)
    index_bam(phased_bam_file, num_threads=8)
    return phased_bam_file


def phase_chrom(contig,
                bam_file,
                vcf_file,
                ref_file,
                output,
                phaser=None,
                unphase_vcf=False,
                phase_with_het=True,
                phasing_ratio=None,
                phasing_quality_threshold=90,
                overwrite=False):
    phased_vcf_file = f"{output}/phased.{contig}.vcf.gz"
    if not overwrite and os.path.exists(f"{phased_vcf_file}.tbi"):
        return output

    if unphase_vcf:
        whatshap_unphase(vcf_file, f"{output}/input.vcf.gz")
        vcf_file = f"{output}/input.vcf.gz"

    if phase_with_het:
        het_file = f"{output}/het.{contig}.vcf.gz"
        select_het_variant(
            vcf_file, contig, het_file,
            ratio=phasing_ratio,
            threshold=phasing_quality_threshold
        )
        index_vcf(het_file, force=True)
        vcf_file = het_file

    if phaser is None:
        platform = get_platform(bam_file)
        if platform in ("illumina", "ILLUMINA"):
            phaser = "whatshap"
        else:
            phaser = "longphase"

    if phaser == "whatshap":
        whatshap_phase(bam_file,
                       vcf_file,
                       reference=ref_file,
                       output=phased_vcf_file,
                       chromosomes=contig,
                       ignore_read_groups=True,
                       distrust_genotypes=True)
    elif phaser == "longphase":
        LongPhase(ref_file).phase(bam_file, vcf_file, output=phased_vcf_file)
    else:
        raise f"no phaser name {phaser}"

    return phased_vcf_file


def phase_vcf_file(
        bam_file,
        vcf_file,
        ref_file,
        output,
        bed_tree=None,
        phaser=None,
        phase_with_het=False,
        phasing_ratio=0.7,
        phasing_quality_threshold=90,
        merge=False,
        overwrite=False):
    merged_vcf_file = f"{output}/phased.vcf.gz"
    if not overwrite and os.path.exists(merged_vcf_file):
        return merged_vcf_file

    startswith_chr = reference_startswith_chr(bam_file)
    if bed_tree is None:
        if startswith_chr:
            contigs = [f"chr{chr}" for chr in range(1, 23)] + ["chrX", "chrY"]
        else:
            contigs = [f"{chr}" for chr in range(1, 23)]
    else:
        contigs = sorted(bed_tree.keys())

    if not os.path.exists(f"{vcf_file}.tbi"):
        print(f"no exist vcf index file, indexing {vcf_file}")
        index_vcf(vcf_file)
    output_dir = f"{output}/phased"
    os.makedirs(output_dir, exist_ok=True)
    phase_fn = partial(phase_chrom,
                       bam_file=bam_file,
                       vcf_file=vcf_file,
                       ref_file=ref_file,
                       output=output_dir,
                       phaser=phaser,
                       phase_with_het=phase_with_het,
                       phasing_ratio=phasing_ratio,
                       phasing_quality_threshold=phasing_quality_threshold,
                       overwrite=overwrite)
    results = process_map(phase_fn, enumerate(contigs))
    vcf_files = {}
    for i, path in results:
        print(f"finished phasing {path}")
        vcf_files[contigs[i]] = path

    if merge:
        print(f"merge phased file: {merged_vcf_file}")
        merge_vcf(list(vcf_files.values()), merged_vcf_file, sort=True, index=True)
        return merged_vcf_file
    else:
        return vcf_files


def phase_haplotag_bam(bam_file,
                       vcf_file,
                       ref_file,
                       output,
                       bed_tree=None,
                       phaser=None,
                       phase_with_het=False,
                       phasing_ratio=0.8,
                       num_procs=None,
                       merge=False,
                       overwrite=True):
    merged_bam_file = f"{output}/phased.haplotag.bam"
    if not overwrite and os.path.exists(merged_bam_file):
        return merged_bam_file

    startswith_chr = reference_startswith_chr(bam_file)
    if bed_tree is None:
        if startswith_chr:
            contigs = [f"chr{chr}" for chr in range(1, 23)] + ["chrX", "chrY"]
        else:
            contigs = [f"{chr}" for chr in range(1, 23)]
    else:
        contigs = sorted(bed_tree.keys())

    if phaser is None:
        platform = get_platform(bam_file)
        if platform in ("illumina", "ILLUMINA"):
            phaser = "whatshap"
        else:
            phaser = "longphase"

    if not os.path.exists(f"{vcf_file}.tbi"):
        print(f"no exist vcf index file, indexing {vcf_file}")
        index_vcf(vcf_file)

    phase_haplotag_fn = partial(phase_haplotag_chrom,
                                bam_file=bam_file,
                                vcf_file=vcf_file,
                                ref_file=ref_file,
                                output=f"{output}/phased",
                                phaser=phaser,
                                phase_with_het=phase_with_het,
                                phasing_ratio=phasing_ratio,
                                overwrite=overwrite)

    results = process_map(phase_haplotag_fn, enumerate(contigs))
    bam_files = {}
    for i, path in results:
        print(f"finished phasing {path}")
        bam_files[contigs[i]] = path

    if merge:
        merge_bam_files(bam_files.values(), merged_bam_file, index=True, num_threads=num_procs)
        return merged_bam_file
    else:
        return bam_files

# Copyright (c) 2025, Tencent Inc. All rights reserved.

from datetime import datetime

import math
import pysam
from collections import OrderedDict, defaultdict
import os
from tgnn.utils import is_tool
from tgnn.multiprocessing import get_cpu_cores


def parse_fasta(filename):
    if isinstance(filename, str):
        ff = pysam.FastaFile(filename)
    else:
        ff = filename
    return ff


def parse_alignment(filename, **kwargs):
    if isinstance(filename, pysam.AlignmentFile):
        return filename

    if filename.endswith(".bam"):
        mode = "rb"
    elif filename.endswith(".sam"):
        mode = "r"
    elif filename.endswith(".cram"):
        mode = "rc"
    else:
        raise ValueError("Unknown file format")

    return pysam.AlignmentFile(filename, mode=mode, **kwargs)


def has_chrY(filename):
    af = get_alignment_header(filename)
    if af.get_tid("chrY") != -1 or af.get_tid("Y") != -1:
        return True

    return False


def split_regions(ref_file, chrs, bin_size=None, num_bins=None):
    if bin_size is None and num_bins is None:
        raise ValueError("Either bin size or number of bins should be specified")

    if isinstance(ref_file, pysam.FastaFile) or ref_file.endswith(("fasta", "fa", "fna")):
        f = parse_fasta(ref_file)
    else:
        f = parse_alignment(ref_file)

    if bin_size is None:
        total_len = 0
        for chrom in chrs:
            total_len += f.get_reference_length(chrom)
        bin_size = int(math.ceil(total_len / num_bins))

    regions = []
    for chr in chrs:
        seq_len = f.get_reference_length(chr)
        starts = range(0, seq_len, bin_size)
        ends = [min(s + bin_size, seq_len) for s in starts]
        regions.extend([chr, s, e] for (s, e) in zip(starts, ends))

    if isinstance(ref_file, str):
        f.close()

    return regions


def get_alignment_header(filename):
    if isinstance(filename, str):
        af = parse_alignment(filename)
    else:
        af = filename

    return af.header


def get_reference_length(filename, contig=None):
    header = get_alignment_header(filename)
    if isinstance(contig, str):
        return header.get_reference_length(contig)

    contigs = contig
    if contigs is None:
        start_with_chr = header.get_reference_name(0).startswith("chr")
        if start_with_chr:
            contigs = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
        else:
            contigs = [f"{i}" for i in range(1, 23)] + ["X", "Y"]

    return OrderedDict([(chrom, header.get_reference_length(chrom)) for chrom in contigs])


def is_sorted_alignment(filename):
    header = get_alignment_header(filename)
    if "HD" not in header:
        return False

    so = header["HD"]["SO"]
    return so in ("coordinate",)


def get_sample_name(filename):
    header = get_alignment_header(filename)
    if "RG" not in header:
        return None

    rgs = header["RG"]
    rg = None
    for rg in rgs:
        if "SM" in rg:
            break
    if rg is None:
        return None

    return rg["SM"]


def get_platform(filename):
    header = get_alignment_header(filename)
    if "RG" not in header:
        return None

    rgs = header["RG"]
    rg = None
    for rg in rgs:
        if "PL" in rg:
            break
    if rg is None:
        return None

    return rg["PL"]


def index_bam(filename, exist_ok=True, num_threads=0, samtools="samtools"):
    if os.path.exists(f"{filename}.bai") and exist_ok:
        return

    if is_tool(samtools):
        cmd = f"{samtools} index -@{num_threads} {filename}"
        print(cmd)
        os.system(cmd)

    pysam.index(f"-@{num_threads}", filename)


def merge_bam_files(files, output, samtools="samtools", index=False, num_threads=0):
    if num_threads is None:
        num_threads = min(get_cpu_cores(), len(files))

    filename = " ".join(files)
    if is_tool(samtools):
        cmd = f"{samtools} merge -@{num_threads} {output} {filename}"
        print(cmd)
        os.system(cmd)
    else:
        pysam.merge(f"-@{num_threads}", output, files)

    if index:
        index_bam(output, num_threads=num_threads)
    return output


def reference_startswith_chr(filename):
    header = get_alignment_header(filename)
    return header.get_reference_name(0).startswith("chr")


def get_mean_coverage(bam_file, regions):
    if isinstance(regions, str):
        results = pysam.samtools.coverage(bam_file, "-r", regions)
        print(results)
        meandepth = results.splitlines()[1].split()[6]
        return float(meandepth)

    depths = [get_mean_coverage(bam_file, r) for r in regions]
    return depths


def get_nm_value(read):
    if read.has_tag("NM"):
        return int(read.get_tag("NM"))

    cigars = read.cigartuples
    if not cigars:
        return 1e10  # If cIGAR string is empty, return a large number to indicate no mismatches

    mismatch_count = 0
    for operation, length in cigars:
        if operation in (pysam.CDIFF, pysam.CINS, pysam.CDEL):  # note that M corresponds to match or mismatch
            mismatch_count += length

    return mismatch_count


def make_vcf_header(filenames, formats=None, info=None, filters=None):
    """make vcf file head

    Args:
        info: list of dict, meta-information of variant record, item must have key id, number, type, description
        formats: list of dict, meta-information of every sample, item must have key id, number, type, description
    """
    if isinstance(filenames, str):
        filenames = [filenames, ]

    assert len(filenames) >= 1, f"{filenames} must have at least one file"
    header = pysam.VariantHeader()
    header.add_meta("filedate", datetime.today().strftime('%Y-%m-%d'))
    contig_lengths = {}
    for filename in filenames:
        af = parse_alignment(filename)
        lengths = get_reference_length(af)
        for name, length in lengths.items():
            if name not in contig_lengths:
                contig_lengths[name] = length
            else:
                assert contig_lengths[
                           name] == length, f"chrom ({name}) have different length {contig_lengths[name]} != {length}"
        sample = get_sample_name(filename)
        header.samples.add(sample)
        af.close()

    for chr, length in contig_lengths.items():
        header.contigs.add(chr, length)

    if filters is not None:
        for data in filters:
            header.filters.add(data["id"], None, None, data["description"])

    if info is not None:
        for data in info:
            if "description" not in data:
                data["description"] = ""
            header.info.add(data["id"], data["number"], data["type"], data["description"])

    formats = [
        {"id": "GT", "number": 1, "type": "String", "description": "Genotype"},
        {"id": "GQ", "number": 1, "type": "Integer", "description": "Genotype Quality"},
        {"id": "DP", "number": 1, "type": "Integer", "description": "Read Depth"},
        {"id": "AD", "number": "R", "type": "Integer", "description": "Read depth for each allele"},
        {"id": "AF", "number": "A", "type": "Float", "description": "Estimated allele frequency in the range of [0,1]"},
        {"id": "PL", "number": "G", "type": "Integer",
         "description": "Phred-scaled genotype likelihoods rounded to the closest integer"},
        {"id": "GP", "number": 4, "type": "Float", "description": "genotype probabilities"},
        {"id": "AP", "number": 1, "type": "Float", "description": "allele type probability"},
    ] if formats is None else formats

    for data in formats:
        if "description" not in data:
            data["description"] = ""
        header.formats.add(data["id"], data["number"], data["type"], data["description"])

    return header

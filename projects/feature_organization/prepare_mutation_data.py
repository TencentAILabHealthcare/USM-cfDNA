# Copyright (c) 2025, Tencent Inc. All rights reserved.

import sys
sys.path.append(".")

from collections import defaultdict
from tgnn.sci.parser.vcf_parsing import parse_variant, get_genotype
from tgnn.sci.parser.sam_parsing import parse_fasta
from functools import partial, lru_cache

BASES = ["A", "C", "G", "T"]

SBS6_TYPES = ["C_to_A", "C_to_G", "C_to_T", "T_to_A", "T_to_C", "T_to_G"]
SBS12_TYPES = [f"{r}_to_{a}" for r in BASES for a in BASES if r != a]
SBS96_TYPES = [f"{l}_{m}_{r}" for m in SBS6_TYPES for l in BASES for r in BASES]
SBS192_TYPES = [f"{l}_{m}_{r}" for m in SBS12_TYPES for l in BASES for r in BASES]

translator = str.maketrans("ACGT", "TGCA")


@lru_cache(maxsize=1024)
def rev_comp(seq):
    return seq[::-1].translate(translator)


def load_variants(vcf_file):
    snp = []
    indel = []
    vf = parse_variant(vcf_file)
    for var in vf:
        filt = list(var.filter.keys())[0]
        if filt in {"RefCall", "LowQual"}:
            continue

        chrom = var.chrom
        pos = var.pos - 1
        gt = get_genotype(var)
        if gt == (0, 0):
            continue

        alleles = var.alleles
        ref = alleles[0]
        alts = alleles[1:]

        for alt in alts:
            if ref == alt:
                continue

            if len(ref) < len(alt):
                indel.append((chrom, pos, "INS", ref, alt))
            elif len(ref) > len(alt):
                indel.append((chrom, pos, "DEL", ref, alt))
            else:
                snp.append((chrom, pos, "SNP", ref[0], alt[0]))

    vf.close()
    return snp, indel


# ----------------------------------------- 1mb ---------------------------------------#
def vcf_to_1mb_features(snp, indel, window=1_000_000):
    snp_counts = defaultdict(int)
    indel_counts = defaultdict(int)
    for chrom, pos, vt, ref, alt in snp:
        bin_id = pos // window
        start = bin_id * window
        end = start + window
        key = f"snp_{chrom}:{start}-{end}"
        snp_counts[key] += 1

    for chrom, pos, vt, ref, alt in indel:
        bin_id = pos // window
        start = bin_id * window
        end = start + window
        key = f"indel_{chrom}:{start}-{end}"
        indel_counts[key] += 1

    features = {**snp_counts, **indel_counts}
    return features


# ----------------------------------------- sbs ---------------------------------------#
def sbs6_key(ref: str, alt: str) -> str:
    if ref in ("A", "G"):
        ref = rev_comp(ref)
        alt = rev_comp(alt)

    return f"{ref}_to_{alt}"


def sbs12_key(ref: str, alt: str) -> str:
    return f"{ref}_to_{alt}"


def sbs96_key(left: str, ref: str, alt: str, right: str) -> str:
    if ref in ("A", "G"):
        ref = rev_comp(ref)
        alt = rev_comp(alt)
        left, right = rev_comp(right), rev_comp(left)
    return f"{left}_{ref}_to_{alt}_{right}"


def sbs192_key(left: str, ref: str, alt: str, right: str) -> str:
    return f"{left}_{ref}_to_{alt}_{right}"


def vcf_to_sbs_features(snp,
                        ref_file,
                        want=("sbs6", "sbs12", "sbs96", "sbs192"),
                        by_chrom=True):
    want = set(want)
    global6 = defaultdict(int)
    global12 = defaultdict(int)
    global96 = defaultdict(int)
    global192 = defaultdict(int)
    per_chr6 = defaultdict(lambda: defaultdict(int))
    per_chr12 = defaultdict(lambda: defaultdict(int))
    per_chr96 = defaultdict(lambda: defaultdict(int))
    per_chr192 = defaultdict(lambda: defaultdict(int))
    fa = parse_fasta(ref_file)
    ref_seqs = {}
    for chrom, pos, vt, ref, alt in snp:
        if "sbs6" in want:
            k6 = sbs6_key(ref, alt)
            if k6 in SBS6_TYPES:
                global6[k6] += 1
                if by_chrom:
                    per_chr6[chrom][k6] += 1

        if "sbs12" in want:
            k12 = sbs12_key(ref, alt)
            if k12 in SBS12_TYPES:
                global12[k12] += 1
                if by_chrom:
                    per_chr12[chrom][k12] += 1

        if ("sbs96" in want) or ("sbs192" in want):
            if chrom not in ref_seqs:
                ref_seqs[chrom] = fa.fetch(chrom)

            tri = ref_seqs[chrom][pos - 1:pos + 2].upper()
            if len(tri) != 3:
                continue

            left, ref_base, right = tri
            if ref_base != ref:
                print(f"ref not qual ref: {ref}, tri: {tri}")
                continue

            if left not in BASES or right not in BASES:
                continue

            if "sbs192" in want:
                k192 = sbs192_key(left, ref, alt, right)
                if k192 in SBS192_TYPES:
                    global192[k192] += 1
                    if by_chrom:
                        per_chr192[chrom][k192] += 1

            if "sbs96" in want:
                k96 = sbs96_key(left, ref, alt, right)
                if k96 in SBS96_TYPES:
                    global96[k96] += 1
                    if by_chrom:
                        per_chr96[chrom][k96] += 1

    feats = {}
    if "sbs6" in want:
        for k in SBS6_TYPES:
            feats[f"sbs6_{k}"] = global6.get(k, 0)
    if "sbs12" in want:
        for k in SBS12_TYPES:
            feats[f"sbs12_{k}"] = global12.get(k, 0)
    if "sbs96" in want:
        for k in SBS96_TYPES:
            feats[f"sbs96_{k}"] = global96.get(k, 0)
    if "sbs192" in want:
        for k in SBS192_TYPES:
            feats[f"sbs192_{k}"] = global192.get(k, 0)

    if by_chrom:
        if "sbs6" in want:
            for chrom, d in per_chr6.items():
                for k in SBS6_TYPES:
                    feats[f"sbs6_{chrom}_{k}"] = d.get(k, 0)
        if "sbs12" in want:
            for chrom, d in per_chr12.items():
                for k in SBS12_TYPES:
                    feats[f"sbs12_{chrom}_{k}"] = d.get(k, 0)
        if "sbs96" in want:
            for chrom, d in per_chr96.items():
                for k in SBS96_TYPES:
                    feats[f"sbs96_{chrom}_{k}"] = d.get(k, 0)

        if "sbs192" in want:
            for chrom, d in per_chr192.items():
                for k in SBS192_TYPES:
                    feats[f"sbs192_{chrom}_{k}"] = d.get(k, 0)
    fa.close()
    return feats


# ----------------------------------------- dbs ---------------------------------------#
def vcf_to_dbs_features(snp):
    snp_sorted = sorted(snp, key=lambda x: (x[0], x[1]))
    genome = defaultdict(int)
    per_chr = defaultdict(lambda: defaultdict(int))
    i = 0
    while i < len(snp_sorted) - 1:
        c1, p1, _, r1, a1 = snp_sorted[i]
        c2, p2, _, r2, a2 = snp_sorted[i + 1]
        if c1 == c2 and p2 == p1 + 1:
            ref2 = r1 + r2
            alt2 = a1 + a2
            ref_key, alt_key = ref2, alt2
            if ref2[0] in ("A", "G"):
                ref_key = rev_comp(ref2)
                alt_key = rev_comp(alt2)
            key = f"{ref_key}>{alt_key}"
            genome[key] += 1
            per_chr[c1][key] += 1
            i += 2
        else:
            i += 1

    feats = {f"dbs_{k}": v for k, v in genome.items()}
    for chrom, d in per_chr.items():
        for k, v in d.items():
            feats[f"dbs_{chrom}_{k}"] = v

    return feats

def get_mutation_feature(vcf_file, ref_file, window=1_000_000):
    snp, indel = load_variants(vcf_file)
    features = vcf_to_1mb_features(snp, indel, window=window)
    features.update(vcf_to_sbs_features(snp, ref_file=ref_file, want=("sbs6", "sbs12", "sbs96", "sbs192")))
    features.update(vcf_to_dbs_features(snp))

    return features
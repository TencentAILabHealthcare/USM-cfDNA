# Copyright (c) 2025, Tencent Inc. All rights reserved.
from collections import defaultdict
import numpy as np
from tgnn.sci.constants import base_constants as bc
from tgnn.sci.parser.bed_parsing import is_region_in
from tgnn.sci.constants.alt24 import allele_types, allele_type_to_genotype, allele_prob_to_genotype
from tgnn.sci.constants.chr_constants import XY_PAR
from .quality_utils import compute_gq


def get_indel_alt(alt_base_ids, ref_seq):
    ref_seq = ref_seq.upper()
    bases = []
    for i, (base_id, ref_base) in enumerate(zip(alt_base_ids, ref_seq)):
        base = bc.alt_base_types[base_id]
        if base in (".",):
            break

        if i == 0 and base == "-":
            base = "*"

        if i > 0 and base not in "acgtn-":
            break

        if i > 1:
            last_base = bc.alt_base_types[alt_base_ids[i - 1]]
            if last_base in "-" and base in "acgtn":
                break

            if last_base in "acgtn" and base == "-":
                break

        if base in "-":
            base = ref_base

        bases.append(base)

    if len(bases) <= 1:
        return None

    alt = "".join(bases)
    if bases[1].islower():
        return "+" + alt.upper()
    else:
        return "-" + alt.upper()


def calling_model_output_to_variant(
        pred_at,
        meta,
        input_mode="msa",
        indel_threshold=55.0,
        snp_threshold=55.0,
        haploid_contigs=None
):
    ref_base = meta["ref_base"]
    pred_gt = allele_prob_to_genotype(ref_base, pred_at)
    gq, pl = compute_gq(pred_gt, return_pl=True)
    gq = int(gq)
    pl = [int(x) for x in pl]
    alt_bases = meta["alt_bases"]
    alt_bases = {k: v for k, v in alt_bases.items()}

    diploid = True
    haploid_contigs = haploid_contigs or ()
    if meta["chrom"] in haploid_contigs:
        # position in PAR or female chrX is diploid
        diploid = is_region_in(XY_PAR, meta["chrom"], meta["pos"])

    prob, gt, at, alleles, allele_counts = get_sit_output_from_at(
        pred_at.tolist(),
        ref_base,
        alt_bases,
        diploid=diploid
    )

    if alleles is None or allele_counts is None:
        return None

    depth = sum(b[1] for b in alt_bases.items())
    # qual = quality_score_from(prob)
    qual = round(prob * 100, 2)
    pls = (pl[0], pl[2], pl[1], pl[3])
    if len(alleles) == 2:
        pls = pls[:3]
    else:
        pls = (pls[0], pls[1], pls[2], 0, pls[3], 0)

    af = [cnt / depth for cnt in allele_counts[1:]]
    ad = allele_counts
    is_indel = "I" in at or "D" in at
    threshold = indel_threshold if is_indel else snp_threshold

    if gt == (0, 0):
        filter = "RefCall"
    else:
        filter = "PASS" if qual > threshold else "LowQual"

    record = dict(
        contig=meta["chrom"],
        start=meta["pos"],
        alleles=alleles,
        qual=qual,
        filter=filter,
        samples=[{
            "GT": gt,
            "GQ": gq,
            "DP": depth,
            "PL": pls,
            "AD": ad,
            "AF": af,
            "AP": round(prob, 2),
            "GP": [round(p, 2) for p in pred_gt.tolist()]
        }],
        info={"IM": "P" if input_mode in ("profile",) else "M"}
    )
    return record


def match_indel_alt(pred_alts, gt_alts, n=1):
    if pred_alts is None:
        return gt_alts[:n]

    matches = []
    for pred_alt in pred_alts:
        for gt_alt in gt_alts:
            if gt_alt.startswith(pred_alt):
                if gt_alt not in matches:
                    matches.append(gt_alt)
                    # best match
                    break
    if len(matches) >= n:
        return matches[:n]

    others = [a for a in gt_alts if a not in matches]
    return matches + others[:n - len(matches)]


def get_sit_output_from_at(pred_at, ref_base, alt_bases, pred_indel_alts=None,
                           diploid=True):
    alt_bases = {k: v for k, v in alt_bases.items() if v > 0}
    del_seqs = []
    ins_seqs = []
    base_seqs = defaultdict(int)
    for base, n in alt_bases.items():
        # first base in indel alt_bases is also base in column
        if base[0] == "-":
            del_seqs.append((base[1:], n))
            base_seqs[base[1]] += n
        elif base[0] == "+":
            ins_seqs.append((base[1:], n))
            base_seqs[base[1]] += n
        else:
            base_seqs[base] += n

    for b, n in base_seqs.items():
        if b not in alt_bases:
            alt_bases[b] = n

    if ref_base not in alt_bases:
        alt_bases[ref_base] = 0

    del_seqs = sorted(del_seqs, key=lambda x: -x[1])  # sort by counts
    del_seqs = [b[0] for b in del_seqs]
    ins_seqs = sorted(ins_seqs, key=lambda x: -x[1])
    ins_seqs = [b[0] for b in ins_seqs]

    base_seqs = [b for b, n in base_seqs.items()]
    # 1/1
    homo_snp_types = [t for t in ["AA", "CC", "GG", "TT"] if ref_base not in t]
    # 0/1
    hetero_snp_types = [t for t in ["AC", "AG", "AT", "CG", "CT", "GT"] if ref_base in t]
    # 1/2
    ma_snp_types = [t for t in ["AC", "AG", "AT", "CG", "CT", "GT"] if ref_base not in t]
    # 0/1 RI
    refins = "".join(sorted(["I", ref_base]))
    # 0/1 RD
    refdel = "".join(sorted(["D", ref_base]))
    # 1/2 BI
    basei_types = ["".join(sorted(["I", base])) for base in "AGCT" if base != ref_base]
    # 1/2 BD
    basedel_types = ["".join(sorted(["D", base])) for base in "AGCT" if base != ref_base]
    if pred_indel_alts is not None:
        pred_indel_alts = {alt for alt in pred_indel_alts if alt}
        pred_ins_alts = {alt[1:] for alt in pred_indel_alts if alt[0] == "+"}
        pred_del_alts = {alt[1:] for alt in pred_indel_alts if alt[0] == "-"}
    else:
        pred_ins_alts = None
        pred_del_alts = None

    prob = 0
    gt = None
    at = None
    alleles = None
    ref_count = alt_bases[ref_base]
    allele_counts = None
    at_indices = np.argsort(np.array(pred_at))[::-1]
    for i, at_id in enumerate(at_indices):
        at = allele_types[at_id]
        prob = pred_at[at_id]
        gt = allele_type_to_genotype(ref_base, at)

        if gt in ((0, 1), (1, 2)) and not diploid:
            continue

        # homo refeerence
        if gt == (0, 0):
            if ref_base not in base_seqs:
                continue

            alleles = [ref_base, ref_base]
            allele_counts = [ref_count, ref_count]
            break

        # SNP
        if gt == (0, 1) and at in hetero_snp_types:
            if ref_base not in base_seqs:
                continue

            alt_base = at[1] if at[0] == ref_base else at[0]
            if alt_base not in base_seqs:
                continue

            alleles = [ref_base, alt_base]
            allele_counts = [ref_count, alt_bases[alt_base]]
            break

        if gt == (1, 1) and at in homo_snp_types:
            alt_base = at[0]
            if alt_base not in base_seqs:
                continue

            alleles = [ref_base, alt_base]
            allele_counts = [ref_count, alt_bases[alt_base]]
            break

        if gt == (1, 2) and at in ma_snp_types:
            if at[0] not in base_seqs or at[1] not in base_seqs:
                continue

            alleles = [ref_base, at[0], at[1]]
            allele_counts = [ref_count, alt_bases[at[0]], alt_bases[at[1]]]
            break

        # INS
        if gt == (0, 1) and at == refins:
            if ref_base not in base_seqs:
                continue

            if len(ins_seqs) < 1:
                continue

            iseq = match_indel_alt(pred_ins_alts, ins_seqs, n=1)[0]
            alleles = [ref_base, iseq]
            allele_counts = [ref_count, alt_bases["+" + iseq]]
            break

        if gt == (1, 2) and at in basei_types:
            base = at[1] if at[0] == "I" else at[0]
            if len(ins_seqs) < 1 or base not in base_seqs:
                continue

            iseq = match_indel_alt(pred_ins_alts, ins_seqs, n=1)[0]
            alleles = [ref_base, base, ins_seqs[0]]
            allele_counts = [ref_count, alt_bases[base], alt_bases["+" + iseq]]
            break

        if gt == (1, 1) and at == "II":
            if len(ins_seqs) < 1:
                continue

            iseq = match_indel_alt(pred_ins_alts, ins_seqs, n=1)[0]
            alleles = [ref_base, iseq]
            allele_counts = [ref_count, alt_bases["+" + iseq]]
            break

        if gt == (1, 2) and at == "iI":
            if len(ins_seqs) < 2:
                continue

            iseq1, iseq2 = match_indel_alt(pred_ins_alts, ins_seqs, n=2)
            alleles = [ref_base, iseq1, iseq2]
            allele_counts = [ref_count, alt_bases["+" + iseq1], alt_bases["+" + iseq2]]
            break

        # DEL
        if gt == (0, 1) and at == refdel:
            if ref_base not in base_seqs:
                continue

            if len(del_seqs) < 1:
                continue

            dseq = match_indel_alt(pred_del_alts, del_seqs, n=1)[0]
            alleles = [dseq, ref_base]
            allele_counts = [ref_count, alt_bases["-" + dseq], ]
            break

        if gt == (1, 2) and at in basedel_types:
            if len(del_seqs) < 1:
                continue

            base = at[1] if at[0] == "D" else at[0]
            if base not in base_seqs:
                continue

            dseq = match_indel_alt(pred_del_alts, del_seqs, n=1)[0]
            alleles = [dseq, base + dseq[1:], ref_base]
            allele_counts = [ref_count, alt_bases[base], alt_bases["-" + dseq]]
            break

        if gt == (1, 1) and at == "DD":
            if len(del_seqs) < 1:
                continue

            dseq = match_indel_alt(pred_del_alts, del_seqs, n=1)[0]
            alleles = [dseq, ref_base]
            allele_counts = [ref_count, alt_bases["-" + dseq]]
            break

        if gt == (1, 2) and at == "dD":
            if len(del_seqs) < 2:
                continue

            del1, del2 = match_indel_alt(pred_del_alts, del_seqs, n=2)
            if len(del1) < len(del2):
                del1, del2 = del2, del1

            n = len(del1) - len(del2)
            alleles = [del1, ref_base, ref_base + del1[-n:]]
            allele_counts = [ref_count, alt_bases["-" + del1], alt_bases["-" + del2]]
            break

        if gt == (1, 2) and at == "DI":
            if len(del_seqs) < 1 or len(ins_seqs) < 1:
                continue

            dseq = match_indel_alt(pred_del_alts, del_seqs, n=1)[0]
            iseq = match_indel_alt(pred_ins_alts, ins_seqs, n=1)[0]
            alleles = [dseq, ref_base, iseq + dseq[1:]]
            allele_counts = [ref_count, alt_bases["-" + dseq], alt_bases["+" + iseq]]
            break

    return prob, gt, at, alleles, allele_counts

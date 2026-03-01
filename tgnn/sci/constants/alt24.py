# Copyright (c) 2025, Tencent Inc. All rights reserved.

import numpy as np

allele_types = [
    "NN",
    "AA",
    "AC",
    "AG",
    "AT",
    "AI",  # I: Ins
    "AD",  # D: Del
    "CC",
    "CG",
    "CT",
    "CI",
    "CD",
    "GG",
    "GT",
    "GI",
    "DG",
    "TT",
    "IT",
    "DT",
    "DD",
    "DI",
    "II",
    "dD",  # different deletion
    "iI",  # different insertion
]


def _partial_label_from(ref, alt):
    if "*" in alt:
        alt = ""

    if len(ref) > len(alt):
        return "D"

    elif len(ref) < len(alt):
        return "I"

    return alt[0]


def allele_type_to_genotype(ref_base, at):
    assert len(at) == 2, f"only support diploid, get {at}"

    if ref_base not in "AGCT":
        return (0, 0)

    if "N" in at:
        return (0, 0)

    if ref_base in at:
        if len(set(at)) == 1:
            return (0, 0)
        else:
            return (0, 1)

    if len(set(at)) == 1:
        return (1, 1)

    return (1, 2)


def allele_prob_to_genotype(ref_base, probs):
    gt_probs = [0, 0, 0, 0]
    genotypes = ("0/0", "1/1", "0/1", "1/2")
    for at, prob in zip(allele_types, probs):
        gt1, gt2 = allele_type_to_genotype(ref_base, at)
        if gt1 > gt2:
            gt2, gt1 = gt1, gt2

        gt = f"{gt1}/{gt2}"
        gt_probs[genotypes.index(gt)] += prob
    return np.array(gt_probs)


def variant_to_allele_type(ref, alt0, alt1):
    g0 = _partial_label_from(ref, alt0)
    g1 = _partial_label_from(ref, alt1)

    g0, g1 = sorted([g0, g1])
    if g0 == g1 and g0 in ("D", "I") and alt0 != alt1:
        g0 = g0.lower()

    allele_type = g0 + g1
    return allele_type_to_index(allele_type)


def allele_type_to_index(allele_type):
    return allele_types.index(allele_type)

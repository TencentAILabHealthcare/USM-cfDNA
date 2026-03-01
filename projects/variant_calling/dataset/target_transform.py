# Copyright (c) 2025, Tencent Inc. All rights reserved.
from tgnn.sci.constants.alt24 import variant_to_allele_type


def build_variant_calling_label(variant):
    genotype = tuple(variant["genotype"])
    alleles = variant["alleles"]
    if genotype == (2, 2):
        genotype = (1, 1)
        alleles = (alleles[0], alleles[2])

    if "*" in alleles[1:]:
        if genotype == (1, 2):
            genotype = (1, 1)
        else:
            genotype = (0, 1)

    ref_bases = alleles[0]
    alts = alleles[1:]
    if len(alts) == 1:
        alts = [ref_bases, alts[0]] if 0 in genotype else [alts[0], alts[0]]

    if "*" in alts:
        alt = alts[0] if alts[0] != "*" else alts[1]
        at = variant_to_allele_type(ref_bases, alt, alt)
    else:
        at = variant_to_allele_type(ref_bases, alts[0], alts[1])

    return {"at": at}

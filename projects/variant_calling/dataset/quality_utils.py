# Copyright (c) 2025, Tencent Inc. All rights reserved.
from tgnn.utils import tensor as top


def quality_score_from(p, eps=1e-10):
    """
    make a modification for quality score calculation. did not apply quality square for computation.
    """
    p = top.clip(p, 0, 1.0)
    tmp = -10 * top.log10((1.0 - p + eps) / (p + eps)) + 10
    return float(round(max(tmp, 0), 2))


def compute_qual(gt_probs, eps=1e-8):
    """QUAL is the Phred-scaled probability that the site has no variant and is computed as:
    QUAL = -10*log10 (posterior genotype probability of a homozygous-reference genotype (GT=0/0))
    That is, QUAL = GP (GT=0/0), where GP = posterior genotype probability in Phred scale.
    QUAL = 20 means there is 99% probability that there is a variant at the site. The GP values are also given in Phred-scale in the VCF file.

    Args:
        gt_probs: [*, num_genotypes]

    Returns:
        qual: [*]
    """
    qual = -10 * top.log10(gt_probs[..., 0] + eps)
    return top.clip(qual, 0, 99)


def compute_pl(gt_probs, normalize=True, eps=1e-8):
    """normalized Phred-scaled likelihoods of the possible genotypes.
    Args:
        gt_probs: [*, num_genotypes]

    Returns:
        pl: [*, num_genotypes]
    """
    pl = -10 * top.log10(gt_probs + eps)
    if normalize:
        pl = pl - top.amin(pl, dim=-1, keepdim=True)

    return pl


def compute_gq(gt_probs, return_pl=False):
    """The value of GQ is simply the difference between the second lowest PL and the lowest PL(always 0 when normalized PL)

    GQ is the Phred-scaled Probability that the call is incorrect. GQ=-10*log10(p), where p is the probability that the call is incorrect.

    Refs:
        1. https://gatk.broadinstitute.org/hc/en-us/articles/360035890451-Calculation-of-PL-and-GQ-by-HaplotypeCaller-and-GenotypeGVCFs
        2. https://jp.support.illumina.com/content/dam/illumina-support/help/Illumina_DRAGEN_Bio_IT_Platform_v3_7_1000000141465/Content/SW/Informatics/Dragen/QUAL_QD_GQ_Formulation_fDG.htm

    Args:
        gt_probs: [*, num_genotypes]

    Returns:
        gq: [*]
    """
    pl = compute_pl(gt_probs)
    sorted_pl = top.sort(pl, dim=-1)
    gq = top.clip(sorted_pl[..., 1], 0, 99)
    if return_pl:
        return gq, pl

    return gq

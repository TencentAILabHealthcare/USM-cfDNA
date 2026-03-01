# Copyright (c) 2024, Tencent Inc. All rights reserved.

import sys
import warnings

from collections import defaultdict
from typing import Dict
from collections import Counter
from functools import lru_cache
import pysam
import numpy as np
from intervaltree import IntervalTree
from tgnn.utils.type import DefaultOrderedDict
from ..parser.bed_parsing import is_region_in
from ..parser.sam_parsing import parse_fasta, parse_alignment
from ..parser.vcf_parsing import get_variants, variants_to_dict, parse_variant, get_genotype
from ..constants import base_constants as bc


def check_base(base):
    na = base.upper()
    if na not in (".", "-", "*", "#", "A", "G", "C", "T", "N", "U"):
        if na in bc.iupac_to_base:
            na = bc.iupac_to_base[na][0]
        else:
            na = "N"
            print(f"error base: {base}", file=sys.stderr)

    if base.islower():
        na = na.lower()

    return na


@lru_cache(maxsize=100)
def check_seq(seq):
    new_seq = [check_base(base) for base in seq]
    return "".join(new_seq)


def check_msa(msa):
    new_msa = []
    for seq in msa:
        new_msa.append(check_seq(seq))
    return new_msa


def in_region(index, region):
    if index >= region[1] or index < region[0]:
        return False

    return True


def parse_align_read(read, start, end, ref_seq=None):
    ref_pos = read.reference_start
    is_forward = read.is_forward
    if ref_pos >= end:
        return None

    cigars = read.cigartuples
    if cigars is None or read.query_sequence is None:
        return None

    query_seq = check_seq(read.query_sequence.upper())
    base_qualities = read.query_qualities
    seq = []
    bq = []
    insertions = []
    deletions = []
    query_pos = 0
    last_pos = ref_pos
    for operation, length in cigars:
        if operation in (pysam.CSOFT_CLIP,):
            query_pos += length

        elif operation in (pysam.CREF_SKIP,):
            ref_pos += length

        elif operation in (pysam.CMATCH, pysam.CEQUAL, pysam.CDIFF):
            length = min(ref_pos + length, end) - ref_pos
            if start < ref_pos + length:
                p = query_pos + max(ref_pos, start) - ref_pos
                seq.extend(query_seq[p:query_pos + length])
                bq.extend(base_qualities[p:query_pos + length])
            ref_pos += length
            query_pos += length
            last_pos = ref_pos
        elif operation == pysam.CINS:
            if ref_pos >= start:
                ins_len = min(ref_pos + length, end) - ref_pos
                ins_seq = query_seq[query_pos: query_pos + ins_len]

                if len(seq):
                    # seq[-1] = f"{seq[-1]}+{ins_len}{ins_seq}"
                    insertions.append((
                        ref_pos - start,
                        ins_seq,
                        np.mean(base_qualities[query_pos: query_pos + ins_len])
                    ))

            query_pos += length
        elif operation == pysam.CDEL:
            length = min(ref_pos + length, end) - ref_pos
            if start < ref_pos + length:
                del_len = min(ref_pos + length - start, length)
                try:
                    qual = base_qualities[query_pos]
                except IndexError:
                    qual = 0

                if len(seq):
                    if ref_seq is None:
                        del_seq = "N" * del_len
                    else:
                        del_seq = ref_seq[ref_pos - start:ref_pos - start + del_len]
                    # seq[-1] = f"{seq[-1]}-{del_len}" + del_seq
                    deletions.append((
                        ref_pos - start,
                        del_seq,
                        qual
                    ))

                seq.extend(["-"] * del_len)
                bq.extend([qual, ] * del_len)

            ref_pos += length
            last_pos = ref_pos
        # else:
        #     operation in (pysam.CHARD_CLIP, pysam.CPAD):
        #     hard clipping, clipped sequence not present in query seq
        if ref_pos >= end:
            break

    if len(seq) == 0:
        return None

    output = {
        "ref_pos": last_pos - len(seq),
        "bases": seq,
        "qualities": bq,
        "insertions": insertions,
        "deletions": deletions
    }

    return output


def make_sequencing_msa(
        align_reads,
        start,
        end,
        ref_seq=None,
        min_bq=0,
        min_mq=0,
        max_depth=None,
        joint_fragment=False):
    window = end - start
    msa = DefaultOrderedDict(lambda: {
        "base_qualities": [0] * window,
        "mapping_qualities": [0] * window,
        "strands": [3] * window,
        "seqs": ["."] * window,
        "insertions": [],
        "deletions": []
    })
    for cnt, align in enumerate(align_reads):
        mapping_quality = align.mapping_quality
        if mapping_quality < min_mq:
            continue

        fragment_id = align.query_name
        if fragment_id is None:
            continue

        output = parse_align_read(align, start, end, ref_seq=ref_seq)
        if output is None:
            continue

        ref_start = output["ref_pos"]
        if not joint_fragment:
            fragment_id = f"read{cnt}"

        line = msa[fragment_id]
        is_forward = align.is_forward
        bases = line["seqs"]
        base_qualities = line["base_qualities"]
        mapping_qualities = line["mapping_qualities"]
        strands = line["strands"]
        line["insertions"].extend(output["insertions"])
        line["deletions"].extend(output["deletions"])
        for i, (base2, qual2) in enumerate(zip(output["bases"], output["qualities"]), start=ref_start - start):
            base1 = line["seqs"][i]
            qual1 = line["base_qualities"][i]
            if qual2 < min_bq:
                continue

            if qual2 >= qual1 or base1 in (".", "N"):
                bases[i] = base2
                base_qualities[i] = qual2

            if strands[i] == 3:
                strands[i] = 0 if is_forward else 1
                mapping_qualities[i] = mapping_quality
            else:
                strands[i] = 2
                mapping_qualities[i] = max(mapping_quality, mapping_qualities[i])

        if max_depth is not None and len(msa) > max_depth:
            break

    if len(msa) == 0:
        return None

    names = list(msa.keys())
    all_msa = defaultdict(list)
    for n in names:
        lines = msa[n]
        for key, line in lines.items():
            if len(line) > 0 and isinstance(line[0], str):
                line = "".join(line)
            all_msa[key].append(line)

    all_msa = dict(all_msa)
    return all_msa


def make_sequencing_insert_msa(insertions, seq_len):
    mas_seqs = [['.'] * seq_len for _ in range(len(insertions))]
    for i, ins_info in enumerate(insertions):
        seq = mas_seqs[i]
        for (ins_idx, ins_seq, *_) in ins_info:
            seq[ins_idx:ins_idx + len(ins_seq)] = list(ins_seq)
        mas_seqs[i] = "".join(seq)

    return mas_seqs


def get_profile5_from_msa(msa):
    seq_len = len(msa[0])
    profiles = defaultdict(lambda: np.zeros((seq_len,), dtype=np.int32))
    for seq in msa:
        for i, nt in enumerate(seq):
            profiles[nt][i] += 1
    return profiles


def get_profile24_from_msa(msa, strands, insertions):
    seq_len = len(msa[0])
    profiles = defaultdict(lambda: np.zeros((seq_len,), dtype=np.int32))
    for i, (seq, strand) in enumerate(zip(msa, strands, insertions)):
        for b, s in zip(seq, strand):
            if s == 3:
                continue

            if b == "-":
                b = "D"

            if strand in (0, 2):
                profiles[b][i] += 1

            if strand in (1, 2):
                profiles[b.lower()][i] += 1

        for insertion in insertions:
            ins_pos, ins_seq, *_ = insertion
            s = strand[ins_pos]
            if s == 3:
                continue

            if strand in (0, 2):
                profiles["I"][i] += 1
                for ii, ibase in enumerate(ins_seq, start=ins_pos):
                    if ii >= seq_len:
                        continue
                    profiles[f"{ibase}+"][i] += 1
            if strand in (1, 2):
                profiles["i"][i] += 1

                for ii, ibase in enumerate(ins_seq, start=ins_pos):
                    if ii >= seq_len:
                        continue
                    profiles[f"{ibase.lower()}+"][i] += 1


def get_profile_from_msa(msa, strand_msa, insertions, deletions):
    seq_len = len(msa[0])
    profiles = defaultdict(lambda: np.zeros((seq_len,), dtype=np.int32))
    for seq, strands, ins_info, del_info in zip(msa, strand_msa, insertions, deletions):
        for i, (base, strand) in enumerate(zip(seq, strands)):
            assert base in "AGCTN*-.", f"{base} not in AGCTN*-"
            if base in ".-":
                continue

            if strand == 0:
                profiles[base][i] += 1
            elif strand == 1:
                profiles[base.lower()][i] += 1
            elif strand == 2:
                profiles[base][i] += 1
                profiles[base.lower()][i] += 1
            else:
                raise ValueError(f"illegal strand id: {strand}")

        for info in ins_info:
            ins_idx, ins_seq = info[:2]
            if ins_idx > seq_len:
                continue

            ins_pos = ins_idx - 1
            if ins_pos < 0:
                continue

            strand = strands[ins_pos]
            ie_pos = ins_pos + len(ins_seq)
            if strand == 0:
                profiles["IS"][ins_pos] += 1
                if ie_pos < seq_len:
                    profiles["IE"][ie_pos] += 1
            elif strand == 1:
                profiles["is"][ins_pos] += 1
                if ie_pos < seq_len:
                    profiles["ie"][ie_pos] += 1
            elif strand == 2:
                profiles["is"][ins_pos] += 1
                profiles["IS"][ins_pos] += 1
                if ie_pos < seq_len:
                    profiles["IE"][ie_pos] += 1
                    profiles["ie"][ie_pos] += 1
            else:
                raise ValueError(f"illegal strand id: {strand}")

            strand = strands[ins_pos]
            for ii, ibase in enumerate(ins_seq, start=ins_idx):
                if ii >= seq_len:
                    break

                if strand == 0:
                    profiles[f"{ibase}+"][ii] += 1
                elif strand == 1:
                    profiles[f"{ibase}+".lower()][ii] += 1
                elif strand == 2:
                    profiles[f"{ibase}+"][ii] += 1
                    profiles[f"{ibase}+".lower()][ii] += 1
                else:
                    raise ValueError(f"illegal strand id: {strand}")

        for info in del_info:
            del_idx, del_seq = info[:2]
            if del_idx > seq_len:
                continue

            del_pos = del_idx - 1
            if del_pos < 0:
                continue

            strand = strands[del_pos]
            de_pos = del_pos + len(del_seq)
            if strand == 0:
                profiles["DS"][del_pos] += 1
                if de_pos < seq_len:
                    profiles["DE"][de_pos] += 1

            elif strand == 1:
                profiles["ds"][del_pos] += 1
                if de_pos < seq_len:
                    profiles["de"][de_pos] += 1

            elif strand == 2:
                profiles["DS"][del_pos] += 1
                profiles["ds"][del_pos] += 1
                if de_pos < seq_len:
                    profiles["DE"][de_pos] += 1
                    profiles["de"][de_pos] += 1
            else:
                raise f"illegal strand id: {strand}"

            for di, ibase in enumerate(del_seq, start=del_idx):
                if di >= seq_len:
                    break

                if strand == 0:
                    profiles[f"{ibase}-"][di] += 1
                elif strand == 1:
                    profiles[f"{ibase.lower()}-"][di] += 1
                elif strand == 2:
                    profiles[f"{ibase}-"][di] += 1
                    profiles[f"{ibase.lower()}-"][di] += 1
                else:
                    raise f"illegal strand id: {strand}"

    return profiles


@lru_cache(maxsize=256)
def split_indel_base(base):
    first_base, indel = base[:2]
    base = base[2:]
    ns = []
    for n in base:
        if not n.isdigit():
            break
        ns.append(n)
    length = int("".join(ns))
    seq = base[len(ns):]
    assert length == len(seq), f"{seq} length != {length}"
    return f"{first_base}{seq}"


def candidate_sit_generator(bam_file,
                            ref_file,
                            contig,
                            start=None,
                            end=None,
                            bed_tree: Dict[str, IntervalTree] = None,
                            vcf_file=None,
                            min_depth=4,
                            min_mapq=5,
                            min_bq=0,
                            snp_af=0.08,
                            indel_af=0.08,
                            flag_filter=2316,
                            max_depth=200,
                            truncate=True,
                            **kwargs):
    af = parse_alignment(bam_file)
    start = start or 0
    end = end or af.get_reference_length(contig)

    if bed_tree is not None and not is_region_in(bed_tree, contig, start, end):
        return

    if isinstance(ref_file, pysam.FastaFile):
        ref_seq = ref_file.fetch(contig, start, end).upper()
    else:
        with pysam.FastaFile(ref_file) as ff:
            ref_seq = ff.fetch(contig, start, end).upper()

    variants = {}
    if vcf_file is not None:
        vf = parse_variant(vcf_file)
        variants = get_variants(vf, contig, start, end, bed_tree=bed_tree, reopen=not isinstance(vcf_file, str))
        variants = variants_to_dict(variants)

    for col in af.pileup(contig,
                         start=start,
                         end=end,
                         truncate=truncate,
                         min_base_quality=min_bq,
                         min_mapping_quality=min_mapq,
                         flag_filter=flag_filter,
                         max_depth=max_depth,
                         **kwargs):
        ref_pos = col.reference_pos  # zero-based
        if bed_tree is not None:
            if bed_tree is not None and not is_region_in(bed_tree, contig, ref_pos):
                continue

        ref_base = ref_seq[ref_pos - start]
        if ref_base not in "ACGT":
            continue

        query_seq = col.get_query_sequences(add_indels=True)
        query_counter = Counter(query_seq)
        base_counts = defaultdict(int)
        alt_bases = defaultdict(int)
        for qbase, count in query_counter.items():
            qbase = qbase.upper()
            if qbase in "ACGT":
                base_counts[qbase] += count
                alt_bases[qbase] += count
            elif "+" in qbase:
                base_counts["I"] += count
                base_counts[qbase[0]] += count
                seq = split_indel_base(qbase)
                alt_bases[f"+{seq}"] += count
            elif "-" in qbase:
                base_counts["D"] += count
                base_counts[qbase[0]] += count
                seq = split_indel_base(qbase)
                seq = ref_seq[ref_pos - start: ref_pos - start + len(seq)]
                alt_bases[f"-{seq}"] += count
            else:
                # "*" is deletion in pileup
                base_counts["O"] += count
                alt_bases[qbase] += count

        depth = sum([base_counts[b] for b in "ACGTIDO"])
        if depth < min_depth:
            continue
        # remove another case
        base_counts = [(b, n) for b, n in base_counts.items() if b != "O"]
        if len(base_counts) == 0:
            continue

        data = {"contig": contig, "position": ref_pos, "ref_base": ref_base, "alt_bases": alt_bases}
        if vcf_file is not None:
            name = f"{contig}:{ref_pos}"
            if name in variants:
                var = variants.pop(name)
                genotype = get_genotype(var)
                data.update({
                    "genotype": genotype,
                    "alleles": var.alleles,
                    "variant_type": var.alleles_variant_types
                })
            else:
                data.update({
                    "genotype": (0, 0),
                })

        base_counts.sort(key=lambda x: -x[1])  # sort base_count descendingly
        if base_counts[0][0] != ref_base:
            yield data
            continue

        pass_af = False
        for item, count in base_counts:
            if item == ref_base:
                continue

            if item in 'ID':
                if count / depth >= indel_af:
                    pass_af = True
                    break
            else:
                if count / depth >= snp_af:
                    pass_af = True
                    break

        if pass_af:
            yield data

    # release bam file
    if isinstance(bam_file, str):
        af.close()

    if isinstance(vcf_file, str):
        vf.close()


def candidate_msa_generator(
        bam_file,
        ref_file,
        contig,
        start=None,
        end=None,
        bed_tree: Dict[str, IntervalTree] = None,
        min_depth=4,
        min_mapq=5,
        min_bq=0,
        snp_af=0.08,
        indel_af=0.08,
        num_flanking=16,
        join_fragment=False
):
    af = parse_alignment(bam_file)
    rf = parse_fasta(ref_file)
    seq_len = rf.get_reference_length(contig)
    seq = rf.fetch(contig)
    for candidate in candidate_sit_generator(
            af,
            rf,
            contig,
            start=start,
            end=end,
            bed_tree=bed_tree,
            min_mapq=min_mapq,
            min_bq=min_bq,
            snp_af=snp_af,
            indel_af=indel_af,
            min_depth=min_depth
    ):
        center = candidate["position"]
        cs = max(center - num_flanking, 0)
        ce = min(center + num_flanking + 1, seq_len)
        if (ce - cs) != (2 * num_flanking + 1):
            continue

        aln_reads = af.fetch(contig, start=cs, end=ce)
        msa = make_sequencing_msa(aln_reads, start, end, join_fragment=join_fragment)
        data = {
            **candidate,
            "start": cs,
            "end": ce,
            "ref_seq": check_seq(seq[cs:ce]),
            **msa
        }
        yield data

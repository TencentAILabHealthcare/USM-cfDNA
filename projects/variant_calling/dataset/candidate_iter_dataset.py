# Copyright (c) 2025, Tencent Inc. All rights reserved.
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset
from tgnn.sci.parser.sam_parsing import parse_alignment, split_regions, parse_fasta
from tgnn.sci.parser.bed_parsing import bed_to_tree
from tgnn.sci.data_transform.bam_processing import candidate_sit_generator, make_sequencing_msa, check_seq, \
    make_sequencing_insert_msa, get_profile_from_msa
from tgnn.distributed.comm import get_dataloader_world_size, get_dataloader_rank
from tgnn.sci.parser.vcf_parsing import parse_variant
from tgnn.data.dataset import DATASET
from tgnn.tokenizer import build_tokenizer
from tgnn.sci.constants import base_constants as bc
from tgnn.utils.tensor import collate_dense_tensors
from .target_transform import build_variant_calling_label


@DATASET.register()
class CandidateSitIterDataset(IterableDataset):

    @classmethod
    def build_dataset(cls, cfg, splits=None):
        bam_file = cfg.dataset.bam_file
        ref_file = cfg.dataset.ref_file
        vcf_file = cfg.dataset.get("vcf_file", None)
        bed_file = cfg.dataset.get("bed_file", None)
        chrs = cfg.dataset.get("chrs", None)
        if chrs is None:
            chrs = [f"chr{i}" for i in range(1, 23)]
            print("input chrs: ", chrs)

        ds = cls(bam_file, ref_file,
                 vcf_file=vcf_file,
                 bed_file=bed_file,
                 chrs=chrs,
                 min_depth=cfg.dataset.get("min_depth", 4),
                 min_bq=cfg.dataset.get("min_bq", 0),
                 snp_af=cfg.dataset.get("snp_af", 0.08),
                 indel_af=cfg.dataset.get("indel_af", 0.08),
                 min_mapq=cfg.dataset.get("min_mapq", 5)
                 )

        return {
            "eval": ds,
            "test": ds
        }

    def __init__(self,
                 bam_file,
                 ref_file,
                 chrs=None,
                 bed_file=None,
                 vcf_file=None,
                 min_depth=4,
                 min_mapq=4,
                 min_bq=0,
                 snp_af=0.08,
                 indel_af=0.08,
                 flag_filter=2316,
                 max_depth=200,
                 **kwargs):
        self.chrs = chrs
        self.bam_file = bam_file
        self.ref_file = ref_file
        self.bed_file = bed_file
        self.vcf_file = vcf_file
        self.max_depth = max_depth
        self.flag_filter = flag_filter
        self.min_depth = min_depth
        self.min_bq = min_bq
        self.snp_af = snp_af
        self.indel_af = indel_af
        self.min_mapq = min_mapq
        self.kwargs = kwargs
        self.bed_tree = bed_to_tree(self.bed_file) if self.bed_file is not None else None
        # support mutli-process
        self.vf = parse_variant(self.vcf_file)
        self.rf = None
        self.ref_seqs = {}

    def get_ref(self):
        if self.rf is None:
            self.rf = parse_fasta(self.ref_file)
        return self.rf

    def get_ref_seq(self, chrom, start=None, end=None):
        rf = self.get_ref()
        if chrom not in self.ref_seqs:
            self.ref_seqs[chrom] = rf.fetch(chrom)

        ref_seq = self.ref_seqs[chrom]
        if start is None:
            start = 0

        if end is None:
            end = len(ref_seq)

        assert start < end, f"out of range, start: {start}, end: {end}"
        return ref_seq[start:end]

    def get_local_regions(self, ref_file):
        world_size = get_dataloader_world_size()
        rank = get_dataloader_rank()
        regions = split_regions(ref_file, self.chrs, num_bins=world_size)
        chunk_size = math.ceil(len(regions) / world_size)
        local_regions = regions[rank * chunk_size: (rank + 1) * chunk_size]
        return local_regions

    def __iter__(self):
        af = parse_alignment(self.bam_file)
        rf = self.get_ref()
        regions = self.get_local_regions(rf)
        for (chrom, start, end) in regions:
            for data in candidate_sit_generator(
                    af, rf,
                    chrom, start, end,
                    bed_tree=self.bed_tree,
                    vcf_file=self.vf,
                    min_depth=self.min_depth,
                    min_bq=self.min_bq,
                    min_mapq=self.min_mapq,
                    snp_af=self.snp_af,
                    indel_af=self.indel_af,
                    flag_filter=self.flag_filter,
                    max_depth=self.max_depth,
                    **self.kwargs
            ):
                yield data
        af.close()

    def collate_fn(self, batch):
        return batch


@DATASET.register()
class CandidateMSAIterDataset(CandidateSitIterDataset):

    @classmethod
    def build_dataset(cls, cfg, splits=None):
        bam_file = cfg.dataset.bam_file
        ref_file = cfg.dataset.ref_file
        vcf_file = cfg.dataset.get("vcf_file", None)
        bed_file = cfg.dataset.get("bed_file", None)
        chrs = cfg.dataset.get("chrs", None)
        if chrs is None:
            chrs = [f"chr{i}" for i in range(1, 23)]
            print("input chrs: ", chrs)
        ds = cls(bam_file, ref_file,
                 vcf_file=vcf_file,
                 bed_file=bed_file,
                 chrs=chrs,
                 min_depth=cfg.dataset.get("min_depth", 4),
                 min_bq=cfg.dataset.get("min_bq", 0),
                 snp_af=cfg.dataset.get("snp_af", 0.08),
                 indel_af=cfg.dataset.get("indel_af", 0.08),
                 min_mapq=cfg.dataset.get("min_mapq", 5),
                 num_flanking=cfg.dataset.get("num_flanking", 16)
                 )
        return {"eval": ds}

    def __init__(self,
                 bam_file,
                 ref_file,
                 num_flanking=16,
                 joint_fragment=True,
                 **kwargs):
        super().__init__(bam_file, ref_file, **kwargs)
        self.num_flanking = num_flanking
        self.joint_fragment = joint_fragment
        # fetch support multiprocess
        self.af = parse_alignment(self.bam_file)

    def __iter__(self):
        num_flanking = self.num_flanking
        joint_fragment = self.joint_fragment
        for candidate in super().__iter__():
            chrom = candidate["contig"]
            pos = candidate["position"]
            cs = max(pos - num_flanking, 0)
            ref_seq = self.get_ref_seq(chrom)
            ce = min(pos + num_flanking + 1, len(ref_seq))
            if (ce - cs) != (2 * num_flanking + 1):
                continue

            reads = self.af.fetch(chrom, start=cs, end=ce, multiple_iterators=True)
            msa = make_sequencing_msa(reads, start=cs, end=ce, ref_seq=ref_seq, joint_fragment=joint_fragment)
            data = {
                **candidate,
                "start": cs,
                "end": ce,
                "ref_seq": check_seq(ref_seq[cs:ce]),
                **msa
            }
            yield data


@DATASET.register()
class CandidateTensorIterDataset(CandidateMSAIterDataset):

    @classmethod
    def build_dataset(cls, cfg, splits=None):
        bam_file = cfg.dataset.bam_file
        ref_file = cfg.dataset.ref_file
        vcf_file = cfg.dataset.get("vcf_file", None)
        bed_file = cfg.dataset.get("bed_file", None)
        chrs = cfg.dataset.get("chrs", None)
        if chrs is None:
            chrs = [f"chr{i}" for i in range(1, 23)]
            print("input chrs: ", chrs)

        max_seqs = cfg.dataset.max_seqs
        tokenizer = build_tokenizer(cfg)
        ds = cls(bam_file,
                 ref_file,
                 tokenizer=tokenizer,
                 max_seqs=max_seqs,
                 vcf_file=vcf_file,
                 bed_file=bed_file,
                 chrs=chrs,
                 min_depth=cfg.dataset.get("min_depth", 4),
                 min_bq=cfg.dataset.get("min_bq", 0),
                 snp_af=cfg.dataset.get("snp_af", 0.08),
                 indel_af=cfg.dataset.get("indel_af", 0.08),
                 min_mapq=cfg.dataset.get("min_mapq", 5),
                 num_flanking=cfg.dataset.get("num_flanking", 16),
                 )
        return {
            "eval": ds,
            "test": ds
        }

    def __init__(self,
                 bam_file,
                 ref_file,
                 tokenizer,
                 max_seqs=56,
                 with_info=True,
                 **kwargs):
        super().__init__(bam_file, ref_file, **kwargs)
        self.max_seqs = max_seqs
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad
        self.with_info = with_info

    def get_quality(self, data):
        mapping_qualities = data.get("mapping_qualities", None)
        base_qualities = data.get("base_qualities", None)
        if mapping_qualities is None or base_qualities is None:
            return None

        seq_len = len(data["ref_seq"])
        mq_msa = [[0, ] * seq_len, ] + list(mapping_qualities)
        mq_msa = np.array(mq_msa, dtype=np.float32)
        bq_msa = [[0, ] * seq_len, ] + list(base_qualities)
        bq_msa = np.array(bq_msa, dtype=np.float32)
        qual = np.stack([bq_msa, mq_msa], axis=-1)
        return torch.tensor(qual, dtype=torch.float32)

    def get_strand(self, data):
        strands = data.get("strands", None)
        if strands is None:
            return None

        seq_len = len(data["ref_seq"])
        strand_msa = [[0, ] * seq_len, ] + list(strands)
        return torch.tensor(strand_msa, dtype=torch.long)

    def get_profiles(self, data):
        msa = data["seqs"]
        strands = data["strands"]
        insertions = data["insertions"]
        deletions = data["deletions"]
        profiles = get_profile_from_msa(msa, strands, insertions, deletions)
        base_types = bc.base38_types
        seq_len = len(msa[0])
        counts = torch.zeros((seq_len, len(base_types)), dtype=torch.float32)
        for i, t in enumerate(base_types):
            if t in profiles:
                counts[:, i] = torch.tensor(profiles[t], dtype=torch.float32)
        return counts

    def get_targets(self, data):
        info = {
            "chrom": data["contig"],
            "pos": data["position"],
            "ref_base": data["ref_base"],
            "start": data["start"],
            "ref_seq": data["ref_seq"],
            "alt_bases": data["alt_bases"]}

        if "genotype" not in data:
            return [info, ]

        ref_seq = data["ref_seq"]
        seq_len = len(ref_seq)
        if "alleles" not in data:
            ref_base = ref_seq[seq_len // 2]
            data["alleles"] = [ref_base, ref_base]

        meta = build_variant_calling_label(data)
        at = meta["at"]
        return (info, at)

    def __iter__(self):
        for data in super().__iter__():
            ref_seq = data["ref_seq"]
            seq_len = len(ref_seq)
            insertions = data["insertions"]
            msa = [ref_seq, ] + list(data["seqs"])
            ins_msa = [ref_seq, ] + make_sequencing_insert_msa(insertions, seq_len)
            strands = self.get_strand(data)
            if self.with_info:
                qual = self.get_quality(data) / 100.0

            profiles = self.get_profiles(data) / 100.0
            if len(msa) > self.max_seqs:
                msa = msa[:self.max_seqs + 1]
                ins_msa = ins_msa[:self.max_seqs + 1]
                strands = strands[:self.max_seqs + 1]
                if self.with_info:
                    qual = qual[:self.max_seqs + 1]

            msa_token_ids = torch.stack([self.tokenizer(seq) for seq in msa])
            ins_msa_token_ids = torch.stack([self.tokenizer(seq) for seq in ins_msa])
            inputs = [msa_token_ids, ins_msa_token_ids, strands, profiles]
            if self.with_info:
                inputs.append(qual)

            targets = self.get_targets(data)
            yield inputs, targets

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        msa_token_ids, ins_msa_token_ids, strand_msa, profiles, *info = zip(*inputs)
        inputs = {
            "msa_token_ids": collate_dense_tensors(msa_token_ids, self.pad_id),
            "ins_msa_token_ids": collate_dense_tensors(ins_msa_token_ids, self.pad_id),
            "strand_ids": collate_dense_tensors(strand_msa, 3),
            "profiles": collate_dense_tensors(profiles, self.pad_id),
        }
        if self.with_info:
            qual_msa = info[0]
            inputs["qualities"] = collate_dense_tensors(qual_msa, 0)

        info, *other_tagets = zip(*targets)
        targets = {
            "meta": info
        }

        if len(other_tagets) > 0:
            ats = other_tagets[0]
            targets.update({
                "at": torch.tensor(ats, dtype=torch.long)
            })

        return inputs, targets

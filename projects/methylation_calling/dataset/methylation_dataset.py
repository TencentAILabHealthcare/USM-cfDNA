# Copyright (c) 2025, Tencent Inc. All rights reserved.
import os

import numpy as np
import pysam
import torch
from torch.utils.data import ConcatDataset

from tgnn.data import MMapIndexedJsonlDataset
from tgnn.data.dataset import BaseDataset, DATASET
from tgnn.sci.data_transform.bam_processing import check_seq, check_msa, make_sequencing_msa, \
    make_sequencing_insert_msa, get_profile_from_msa, get_profile5_from_msa
from tgnn.tokenizer import build_tokenizer
from tgnn.utils import print_rank_0
from tgnn.utils.tensor import collate_dense_tensors
from tgnn.sci.constants.base_constants import base14_types, base38_types


class MethylationDataset(BaseDataset):

    def __init__(self,
                 dataset,
                 tokenizer,
                 max_len=256,
                 max_seqs=48,
                 profile_dim=5,
                 with_info=True,
                 split="test"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.training = split == "train"
        self.split = split
        self.pad_id = self.tokenizer.pad
        self.max_len = max_len
        self.max_seqs = max_seqs
        self.profile_dim = profile_dim
        assert self.profile_dim in [5, 14, 38]
        self.with_info = with_info

    def __len__(self):
        return len(self.dataset)

    def get_data(self, index):
        data = self.dataset[index]
        return data

    def get_profile(self, data):
        if self.profile_dim == 5:
            base_types = ["N", "A", "C", "G", "T"]
            profiles = get_profile5_from_msa([data["ref_seq"], ] + list(data["msa"]))
        elif self.profile_dim == 14:
            base_types = base14_types
            profiles = get_profile_from_msa(data["msa"], data["strands"], data["insertions"], data["deletions"])
        elif self.profile_dim == 38:
            base_types = base38_types[:self.profile_dim]
            profiles = get_profile_from_msa(data["msa"], data["strands"], data["insertions"], data["deletions"])
        else:
            raise f"profile_dim {self.profile_dim} not supported"

        profiles = [profiles[b] for b in base_types]
        profiles = torch.tensor(np.array(profiles, dtype=np.float32)).transpose(0, 1)
        return profiles

    def __getitem__(self, index):
        data = self.get_data(index)
        ref_seq = check_seq(data["ref_seq"])
        seq_len = len(ref_seq)
        msa = [ref_seq, ] + list(data["msa"])
        ins_msa = [ref_seq, ] + make_sequencing_insert_msa(data["insertions"], seq_len)
        profiles = self.get_profile(data) / 100.0
        if self.with_info:
            strand_msa = data["strands"]
            strand_msa = [[0, ] * seq_len, ] + list(strand_msa)
            mq_msa = data["mapping_qualities"]
            mq_msa = [[0, ] * seq_len, ] + list(mq_msa)
            bq_msa = data["base_qualities"]
            bq_msa = [[0, ] * seq_len, ] + list(bq_msa)

        num_alignments = len(msa)
        if num_alignments > self.max_seqs:
            indices = np.random.randint(1, num_alignments, size=self.max_seqs - 1)
            indices = [0, ] + list(indices)
            msa = [msa[idx] for idx in indices]
            ins_msa = [ins_msa[idx] for idx in indices]
            if self.with_info:
                strand_msa = [strand_msa[idx] for idx in indices]
                mq_msa = [mq_msa[idx] for idx in indices]
                bq_msa = [bq_msa[idx] for idx in indices]

        msa_token_ids = torch.stack([self.tokenizer(seq) for seq in msa])
        ins_msa_token_ids = torch.stack([self.tokenizer(seq) for seq in ins_msa])
        if self.with_info:
            qual_msa = torch.stack([
                torch.tensor(bq_msa, dtype=torch.float),
                torch.tensor(mq_msa, dtype=torch.float)
            ], dim=-1) / 100.0
            strand_msa = torch.tensor(strand_msa, dtype=torch.long)

        if seq_len > self.max_len:
            start = (seq_len - self.max_len) // 2
            msa_token_ids = msa_token_ids[..., start:start + self.max_len]
            ins_msa_token_ids = ins_msa_token_ids[..., start:start + self.max_len]
            profiles = profiles[start:start + self.max_len]
            if self.with_info:
                qual_msa = qual_msa[..., start: start + self.max_len, :]
                strand_msa = strand_msa[..., start: start + self.max_len]

        meta = data.get("meta", {})
        targets = []
        if self.split in ("train", "eval"):
            target = meta["target"]
            status = meta["methy_status"]
            targets.append(torch.tensor(target, dtype=torch.long))
            targets.append(torch.tensor(status, dtype=torch.float32))
        targets.append(meta)

        inputs = [msa_token_ids, ins_msa_token_ids, profiles]
        if self.with_info:
            inputs.extend([qual_msa, strand_msa])

        return inputs, targets

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        msa_token_ids, ins_msa_token_ids, *other = zip(*inputs)
        info = {}
        profiles, *other = other
        info = {
            "profiles": collate_dense_tensors(profiles, 0),
            **info
        }

        if self.with_info:
            qual_msa, strand_msa = other
            info = {
                **info,
                "qualities": collate_dense_tensors(qual_msa, 0),
                "strand_ids": collate_dense_tensors(strand_msa, 3)
            }
        inputs = {
            "msa_token_ids": collate_dense_tensors(msa_token_ids, self.pad_id),
            "ins_msa_token_ids": collate_dense_tensors(ins_msa_token_ids, self.pad_id),
            **info
        }

        if self.split in ("train", "eval"):
            target, status, meta = zip(*targets)
            targets = {
                "target": torch.stack(target),
                "status": torch.stack(status),
                "meta": meta
            }
        else:
            (meta,) = zip(*targets)
            targets = {"meta": meta}

        return inputs, targets


@DATASET.register()
class MethylationMSADataset(MethylationDataset):

    @classmethod
    def build_dataset(cls, cfg, splits=None):
        tokenizer = build_tokenizer(cfg)
        sets = {"test", "eval"}
        if splits is not None:
            splits = set(splits)
            assert splits.issubset(sets), f"splits({splits}) not in sets({sets})"
        max_len = cfg.dataset.seq_len
        print_rank_0(f"seq_len: {max_len}")
        max_seqs = cfg.dataset.max_seqs
        with_info = cfg.dataset.get("with_info", True)
        profile_dim = cfg.model.get("profile_dim", 5)
        datasets = {}
        if "test" in splits:
            datasets["test"] = cls(cfg.dataset.files,
                                   tokenizer,
                                   max_len=max_len,
                                   max_seqs=max_seqs,
                                   with_info=with_info,
                                   split="test")

        if "eval" in splits:
            datasets["eval"] = cls(cfg.dataset.files,
                                   tokenizer,
                                   max_len=max_len,
                                   max_seqs=max_seqs,
                                   with_info=with_info,
                                   profile_dim=profile_dim,
                                   split="eval")

        return datasets

    def __init__(self,
                 filenames,
                 tokenizer,
                 max_len=256,
                 max_seqs=48,
                 with_info=True,
                 profile_dim=5,
                 split="test"):
        if isinstance(filenames, str):
            filenames = [filenames, ]
        dataset = ConcatDataset([MMapIndexedJsonlDataset(path) for path in filenames])
        super().__init__(dataset,
                         tokenizer,
                         max_len=max_len,
                         max_seqs=max_seqs,
                         with_info=with_info,
                         profile_dim=profile_dim,
                         split=split)


class CandidateToMSADataset(BaseDataset):
    """convert candidate to msa"""

    def __init__(self,
                 candidate_file,
                 bam_file,
                 ref_file,
                 max_len=128):
        assert os.path.exists(candidate_file)
        assert os.path.exists(bam_file)
        assert os.path.exists(ref_file)
        self.dataset = MMapIndexedJsonlDataset(candidate_file)
        self.af = pysam.AlignmentFile(bam_file, "rb")
        self.ref_file = ref_file
        assert max_len % 2 == 0, f"length must be divisible by 2: {max_len}"
        self.num_flanking = max_len // 2
        chrs = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
        self.contig_lenghts = {chr: self.af.get_reference_length(chr) for chr in chrs}
        self.ref_seqs = {}

    def __len__(self):
        return len(self.dataset)

    def get_ref_seq(self, chr, start=None, end=None):
        if chr not in self.ref_seqs:
            with pysam.FastaFile(self.ref_file) as rf:
                self.ref_seqs[chr] = rf.fetch(chr)

        ref_seq = self.ref_seqs[chr]
        if start is None:
            start = 0

        if end is None:
            end = len(rf.fetch(chr))
        assert start < end, f"out of range, start: {start}, end: {end}"
        return ref_seq[start:end]

    def __getitem__(self, index):
        data = self.dataset[index]
        contig = data["contig"]
        pos = data["position"]  # 0-based, CG or GC position
        start = pos - self.num_flanking + 1
        end = pos + self.num_flanking + 1
        align_reads = self.af.fetch(contig, start=start, end=end, multiple_iterators=True)
        ref_seq = self.get_ref_seq(contig, start=start, end=end)
        all_msa = make_sequencing_msa(align_reads, start=start, end=end, ref_seq=ref_seq, joint_fragment=True)
        data = {
            "ref_seq": check_seq(ref_seq),
            "msa": check_msa(all_msa["seqs"]),
            "insertions": all_msa["insertions"],
            "deletions": all_msa["deletions"],
            "strands": all_msa["strands"],
            "mapping_qualities": all_msa["mapping_qualities"],
            "base_qualities": all_msa["base_qualities"],
            "meta": {
                "id": f"{contig}:{pos}",
                "contig": contig,
                "position": pos,
                "depth": len(all_msa["seqs"]),
            }
        }
        return data


@DATASET.register()
class MethylationCandidateDataset(MethylationDataset):

    @classmethod
    def build_dataset(cls, cfg, splits=None):
        sets = {"test", "eval"}
        if splits is not None:
            splits = set(splits)
            assert splits.issubset(sets), f"splits({splits}) not in sets({sets})"

        tokenizer = build_tokenizer(cfg)
        profile_dim = cfg.model.get("profile_dim", 5)
        max_len = cfg.dataset.seq_len
        max_seqs = cfg.dataset.max_seqs
        with_info = cfg.dataset.get("with_info", True)
        print(f"max_len={max_len}\tmax_seqs={max_seqs}\twith_info={with_info}")
        datasets = {}
        if "test" in splits:
            datasets["test"] = cls(cfg.dataset.candidate_files,
                                   cfg.dataset.bam_file,
                                   cfg.dataset.ref_file,
                                   tokenizer,
                                   max_len=max_len,
                                   max_seqs=max_seqs,
                                   with_info=with_info,
                                   profile_dim=profile_dim,
                                   split="test")

        if "eval" in splits:
            datasets["eval"] = cls(cfg.dataset.candidate_files,
                                   cfg.dataset.bam_file,
                                   cfg.dataset.ref_file,
                                   tokenizer,
                                   max_len=max_len,
                                   max_seqs=max_seqs,
                                   with_info=with_info,
                                   profile_dim=profile_dim,
                                   split="eval")

        return datasets

    def __init__(self,
                 candidate_files,
                 bam_file,
                 ref_file,
                 tokenizer,
                 max_len=256,
                 max_seqs=48,
                 with_info=True,
                 profile_dim=5,
                 split="test"):
        if isinstance(candidate_files, str):
            candidate_files = [candidate_files, ]

        factory_kwargs = {"tokenizer": tokenizer,
                          "max_len": max_len,
                          "max_seqs": max_seqs,
                          "split": split}
        dataset = ConcatDataset(
            [CandidateToMSADataset(path, bam_file, ref_file, max_len=max_len) for path in
             candidate_files])
        super().__init__(dataset, with_info=with_info, profile_dim=profile_dim,
                         **factory_kwargs)

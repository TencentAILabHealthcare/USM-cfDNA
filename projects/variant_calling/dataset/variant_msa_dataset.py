# Copyright (c) 2024, Tencent Inc. All rights reserved.
import numpy as np
import torch
from torch.utils.data import ConcatDataset

from tgnn.data.dataset import BaseDataset, DATASET, MMapIndexedJsonlDataset
from tgnn.tokenizer import build_tokenizer
from tgnn.utils.tensor import collate_dense_tensors
from tgnn.sci.constants import base_constants as bc
from tgnn.sci.data_transform.bam_processing import make_sequencing_insert_msa, get_profile_from_msa
from .target_transform import build_variant_calling_label


@DATASET.register()
class VariantMSADataset(BaseDataset):
    """variant calling msa with wrong profile info, strand、bq、 mq should be msa format.
    this version should not be used.
    """

    @classmethod
    def build_dataset(cls, cfg, splits=None):
        tokenizer = build_tokenizer(cfg)
        sets = {"test", "eval"}
        if splits is not None:
            splits = set(splits)
            assert splits.issubset(sets), f"splits({splits}) not in sets({sets})"
        max_seqs = cfg.dataset.max_seqs
        max_len = cfg.dataset.max_len
        max_len = 33
        print(f"tokenizer: {cfg.tokenizer.path}, pad id {tokenizer.pad_id}, max_seqs: {max_seqs}, max_len: {max_len}")
        datasets = {}
        if "test" in splits:
            datasets["test"] = cls.concat_dataset(cfg.dataset.files,
                                                  tokenizer,
                                                  max_len=max_len,
                                                  max_seqs=max_seqs,
                                                  split="test")

        if "eval" in splits:
            datasets["eval"] = cls.concat_dataset(cfg.dataset.files,
                                                  tokenizer,
                                                  max_len=max_len,
                                                  max_seqs=max_seqs,
                                                  split="eval")

        return datasets

    @classmethod
    def concat_dataset(cls,
                       filenames,
                       tokenizer,
                       max_len=33,
                       max_seqs=56,
                       split="test"):
        if isinstance(filenames, str):
            filenames = [filenames, ]
        dataset = ConcatDataset([MMapIndexedJsonlDataset(path, verbose=False) for path in filenames])
        return cls(dataset, tokenizer, max_len=max_len, max_seqs=max_seqs, split=split)

    def __init__(self,
                 dataset,
                 tokenizer,
                 max_len=33,
                 max_seqs=56,
                 max_depth=200,
                 profile_dim=38,
                 split="test"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.training = split == "train"
        self.split = split
        self.pad_id = self.tokenizer.pad
        self.max_len = max_len
        self.max_seqs = max_seqs
        self.max_depth = max_depth
        self.profile_dim = profile_dim

    def __len__(self):
        return len(self.dataset)

    def get_data(self, index):
        data = self.dataset[index]
        return data

    def get_quality(self, data):
        mapping_qualities = data["mapping_qualities"]
        base_qualities = data["base_qualities"]
        seq_len = len(data["ref_seq"])
        mq_msa = [[0, ] * seq_len, ] + list(mapping_qualities)
        mq_msa = np.array(mq_msa, dtype=np.float32)
        bq_msa = [[0, ] * seq_len, ] + list(base_qualities)
        bq_msa = np.array(bq_msa, dtype=np.float32)
        qual = np.stack([bq_msa, mq_msa], axis=-1)
        return torch.tensor(qual, dtype=torch.float32)

    def get_strand(self, data):
        strands = data["strands"]
        seq_len = len(data["ref_seq"])
        strand_msa = [[0, ] * seq_len, ] + list(strands)
        return torch.tensor(strand_msa, dtype=torch.long)

    def get_profiles(self, data):
        msa = data["seqs"] if "seqs" in data else data["msa"]
        profiles = get_profile_from_msa(msa, data["strands"], data["insertions"], data["deletions"])
        base_types = bc.base38_types[:self.profile_dim]
        seq_len = len(msa[0])
        counts = torch.zeros((seq_len, len(base_types)), dtype=torch.float32)
        for i, t in enumerate(base_types):
            if t in profiles:
                counts[:, i] = torch.tensor(profiles[t], dtype=torch.float32)

        return counts

    def get_meta(self, data):
        meta = data.get("meta", {})
        ref_seq = data["ref_seq"]
        meta["ref_seq"] = ref_seq
        return meta

    def get_targets(self, data):
        ref_seq = data["ref_seq"]
        seq_len = len(ref_seq)
        if "meta" in data:
            data = data["meta"]["variants"][0]

        if "alleles" not in data:
            ref_base = ref_seq[seq_len // 2]
            data["alleles"] = [ref_base, ref_base]

        meta = build_variant_calling_label(data)
        at = meta["at"]
        return (at,)

    def get_msa(self, data):
        ref_seq = data["ref_seq"]
        if "seqs" in data:
            msa = [ref_seq, ] + list(data["seqs"])
        else:
            msa = [ref_seq, ] + list(data["msa"])

        return msa

    def get_insert_msa(self, data):
        ref_seq = data["ref_seq"]
        seq_len = len(ref_seq)
        insertions = data["insertions"]
        ins_msa = [ref_seq, ] + make_sequencing_insert_msa(insertions, seq_len)
        return ins_msa

    def clip_data(self, data, max_depth):
        if len(data["strands"]) <= max_depth:
            return data

        if "seqs" in data:
            data["seqs"] = data["seqs"][:max_depth]
        else:
            data["msa"] = data["msa"][:max_depth]

        data["strands"] = data["strands"][:max_depth]
        data["mapping_qualities"] = data["mapping_qualities"][:max_depth]
        data["base_qualities"] = data["base_qualities"][:max_depth]
        data["insertions"] = data["insertions"][:max_depth]
        data["deletions"] = data["deletions"][:max_depth]

        return data

    def __getitem__(self, index):
        data = self.get_data(index)
        data = self.clip_data(data, self.max_depth)
        profiles = self.get_profiles(data) / 100.0
        ref_seq = data["ref_seq"]
        seq_len = len(ref_seq)
        msa = self.get_msa(data)
        ins_msa = self.get_insert_msa(data)

        strands = self.get_strand(data)
        qual_msa = self.get_quality(data) / 100.0

        if len(msa) > self.max_seqs:
            msa = msa[:self.max_seqs + 1]
            ins_msa = ins_msa[:self.max_seqs + 1]
            strands = strands[:self.max_seqs + 1]
            qual_msa = qual_msa[:self.max_seqs + 1]

        msa_token_ids = torch.stack([self.tokenizer(seq) for seq in msa])
        ins_msa_token_ids = torch.stack([self.tokenizer(seq) for seq in ins_msa])
        if seq_len > self.max_len:
            start = (seq_len - self.max_len) // 2
            clip_len = seq_len - 2 * start
            msa_token_ids = msa_token_ids[..., start:start + clip_len]
            ins_msa_token_ids = ins_msa_token_ids[..., start:start + clip_len]
            strands = strands[..., start: start + clip_len]
            qual_msa = qual_msa[..., start: start + clip_len, :]
            profiles = profiles[start: start + clip_len]

        inputs = [msa_token_ids, ins_msa_token_ids, strands, qual_msa, profiles]
        targets = self.get_targets(data)

        return inputs, targets

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        msa_token_ids, ins_msa_token_ids, strand_msa, qual_msa, profiles = zip(*inputs)
        inputs = {
            "msa_token_ids": collate_dense_tensors(msa_token_ids, self.pad_id),
            "ins_msa_token_ids": collate_dense_tensors(ins_msa_token_ids, self.pad_id),
            "strand_ids": collate_dense_tensors(strand_msa, 3),
            "profiles": collate_dense_tensors(profiles, self.pad_id),
            "qualities": collate_dense_tensors(qual_msa, 0)
        }

        (ats,) = zip(*targets)
        targets = {
            "at": torch.tensor(ats, dtype=torch.long)
        }

        return inputs, targets

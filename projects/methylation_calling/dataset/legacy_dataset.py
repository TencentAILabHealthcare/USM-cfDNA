# Copyright (c) 2024, Tencent Inc. All rights reserved.
import os

import numpy as np
import pysam
import torch
from torch.utils.data import ConcatDataset

from tgnn.data import MMapIndexedJsonlDataset
from tgnn.data.dataset import BaseDataset, DATASET
from tgnn.sci.data_transform.bam_processing import check_seq
from tgnn.tokenizer import build_tokenizer
from tgnn.utils.tensor import collate_dense_tensors


def create_fragment_msa(align_reads,
                        center,
                        half_window_size=256,
                        half_valid_window_size=32):
    """Create a multiple sequence alignment (MSA) by fragment for the given reads."""
    window = 2 * half_window_size
    start = center - half_window_size + 1  # e.g. CG site position is 255 and 256, start = 255 - 256 + 1 = 0
    end = center + half_window_size + 1  # e.g. CG site position is 255 and 256, end = 255 + 256 + 1 = 512

    # Group reads (r1 and r2) by their fragment (pair)
    fragment_map = {}
    for align in align_reads:
        fragment_length = abs(align.template_length)
        # QC 1: Skip fragments that are too long
        if fragment_length > 600:
            continue

        # QC 2: Skip fragments that are too short
        if fragment_length < 20:
            continue

        ref_start = align.reference_start  # absolute position
        cigars = align.cigartuples  # Get the CIGAR tuples
        if cigars is None:
            continue

        ref_pos = ref_start
        query_pos = 0
        sel_mas_seq = ['.'] * window  # default is padding
        query_seq = align.query_sequence  # Get the query sequence

        for operation, length in cigars:
            if operation == pysam.CSOFT_CLIP:  # Skip soft-clipped bases
                query_pos += length
            elif operation == pysam.CREF_SKIP:  # Skip intron bases
                ref_pos += length
            elif operation in (pysam.CMATCH, pysam.CEQUAL, pysam.CDIFF):  # Match, equal, or mismatch
                for i in range(length):
                    if start <= ref_pos < end:  # Only consider the region around the center
                        sel_mas_seq[ref_pos - start] = query_seq[query_pos]
                    ref_pos += 1
                    query_pos += 1

        fragment_id = align.query_name  # Get the fragment ID
        if fragment_id not in fragment_map:
            fragment_map[fragment_id] = [sel_mas_seq]
        else:  # Merge R1 and R2
            fragment_map[fragment_id].append(sel_mas_seq)

    msa_seqs = []
    for sequences in fragment_map.values():
        if len(sequences) == 1:  # If there is only one read, use it directly
            merged_sequence = sequences[0]
        else:
            merged_sequence = sequences[0]
            seq = sequences[1]
            for i in range(window):
                if seq[i] != '.':
                    merged_sequence[i] = seq[i]  # Replace with the base if it's not a gap

            fragment_start = next((i for i in range(window) if merged_sequence[i] != '.'),
                                  None)  # Find the start position of the fragment
            if fragment_start is not None:
                fragment_end = next((i for i in range(window - 1, -1, -1) if merged_sequence[i] != '.'),
                                    None) + 1  # Find the end position of the fragment

                for i in range(fragment_start, fragment_end):  # Replace the gaps with 'x'
                    if merged_sequence[i] == '.':
                        merged_sequence[i] = 'x'
            assert len(merged_sequence) == window

        # QC 3: Check if the fragment contains valid bases between the valid start and end positions
        for i in range(half_window_size - half_valid_window_size, half_window_size + half_valid_window_size):
            if merged_sequence[i] != '.':
                msa_seqs.append("".join(merged_sequence))
                break

    return msa_seqs


class MethylationLegacyDataset(BaseDataset):

    def __init__(self,
                 dataset,
                 tokenizer,
                 max_len=512,
                 max_seqs=48,
                 split="test"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.training = split == "train"
        self.split = split
        self.pad_id = self.tokenizer.pad
        self.max_len = max_len
        self.max_seqs = max_seqs

    def __len__(self):
        return len(self.dataset)

    def get_data(self, index):
        data = self.dataset[index]
        return data

    def __getitem__(self, index):
        data = self.get_data(index)
        ref_seq = check_seq(data["ref_seq"])
        seq_len = len(ref_seq)
        msa = [ref_seq, ] + list(data["msa"])
        num_alignments = len(msa)
        if num_alignments > self.max_seqs:
            indices = np.random.randint(1, num_alignments, size=self.max_seqs - 1)
            # keep sorted msa
            indices = np.append(0, indices)
            indices = np.sort(indices)
            msa = [msa[idx] for idx in indices]

        msa_token_ids = torch.stack([self.tokenizer(seq) for seq in msa])
        if seq_len > self.max_len:
            if self.training:
                start = np.random.randint(0, seq_len - self.max_len + 1)
            else:
                start = (seq_len - self.max_len) // 2

            msa_token_ids = msa_token_ids[..., start:start + self.max_len]

        meta = data.get("meta", {})
        targets = []
        if self.split in ("train", "eval"):
            target = data["target"]
            status = data["methy_status"]
            targets.append(torch.tensor(target, dtype=torch.long))
            targets.append(torch.tensor(status, dtype=torch.float32))
        return msa_token_ids, targets, meta

    def collate_fn(self, batch):
        msa_token_ids, targets, meta = zip(*batch)
        msa_token_ids = collate_dense_tensors(msa_token_ids, self.pad_id)
        data = {
            "inputs": {"token_ids": msa_token_ids},
            "meta": meta
        }
        if self.split in ("train", "eval"):
            target, status = zip(*targets)
            data["targets"] = {
                "target": torch.stack(target),
                "status": torch.stack(status)
            }

        return data


@DATASET.register()
class MethylationMSALegacyDataset(MethylationLegacyDataset):

    @classmethod
    def build_dataset(cls, cfg, splits=None):
        tokenizer = build_tokenizer(cfg)
        sets = {"test", "eval"}
        if splits is not None:
            splits = set(splits)
            assert splits.issubset(sets), f"splits({splits}) not in sets({sets})"
        max_len = cfg.dataset.seq_len
        max_seqs = cfg.dataset.max_seqs
        datasets = {}
        if "test" in splits:
            datasets["test"] = cls(cfg.dataset.files, tokenizer,
                                   max_len=max_len,
                                   max_seqs=max_seqs,
                                   split="test")

        if "eval" in splits:
            datasets["eval"] = cls(cfg.dataset.files,
                                   tokenizer,
                                   max_len=max_len,
                                   max_seqs=max_seqs,
                                   split="eval")

        return datasets

    def __init__(self, filenames, tokenizer, max_len=512, max_seqs=48, split="test"):
        if isinstance(filenames, str):
            filenames = [filenames, ]
        dataset = ConcatDataset([MMapIndexedJsonlDataset(path) for path in filenames])
        super().__init__(dataset, tokenizer, max_len=max_len, max_seqs=max_seqs, split=split)


class CandidateToMSALegacyDataset(BaseDataset):
    """convert candidate to msa"""

    def __init__(self,
                 candidate_file,
                 bam_file,
                 ref_file,
                 half_window_size=256,
                 half_valid_window_size=32):
        assert os.path.exists(candidate_file)
        assert os.path.exists(bam_file)
        assert os.path.exists(ref_file)
        self.dataset = MMapIndexedJsonlDataset(candidate_file)
        self.af = pysam.AlignmentFile(bam_file, "rb")
        self.ref_file = ref_file
        self.half_window_size = half_window_size
        self.half_valid_window_size = half_valid_window_size
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
        pos = data["position"]  # 0-based
        half_window_size = self.half_window_size
        center = pos
        start = center - half_window_size + 1  # e.g. CG site position is 255 and 266, start = 255 - 256 + 1 = 0
        end = center + half_window_size + 1  # e.g. CG site position is 255 and 266, end = 255 + 256 + 1 = 512
        align_reads = list(self.af.fetch(contig, start=start + 1, end=end + 1, multiple_iterators=True))
        ref_seq = self.get_ref_seq(contig, start=start, end=end)
        msa = create_fragment_msa(align_reads, center=center,
                                  half_window_size=half_window_size,
                                  half_valid_window_size=self.half_valid_window_size)
        data = {
            "ref_seq": check_seq(ref_seq),
            "msa": msa,
            "meta": {
                "id": f"{contig}:{pos + 1}",
                "contig": contig,
                "position": pos
            }
        }
        return data


@DATASET.register()
class MethylationCandidateLegacyDataset(MethylationLegacyDataset):

    @classmethod
    def build_dataset(cls, cfg, splits=None):
        tokenizer = build_tokenizer(cfg)
        sets = {"test", "eval"}
        if splits is not None:
            splits = set(splits)
            assert splits.issubset(sets), f"splits({splits}) not in sets({sets})"

        max_len = cfg.dataset.seq_len
        max_seqs = cfg.dataset.max_seqs
        datasets = {}
        if "test" in splits:
            datasets["test"] = cls(cfg.dataset.candidate_files,
                                   cfg.dataset.bam_file,
                                   cfg.dataset.ref_file,
                                   tokenizer,
                                   max_len=max_len,
                                   max_seqs=max_seqs,
                                   split="test")

        if "eval" in splits:
            datasets["eval"] = cls(cfg.dataset.candidate_files,
                                   cfg.dataset.bam_file,
                                   cfg.dataset.ref_file,
                                   tokenizer,
                                   max_len=max_len,
                                   max_seqs=max_seqs,
                                   split="eval")

        return datasets

    def __init__(self,
                 candidate_files,
                 bam_file,
                 ref_file,
                 tokenizer,
                 max_len=512,
                 max_seqs=48,
                 split="test"):
        if isinstance(candidate_files, str):
            candidate_files = [candidate_files, ]

        factory_kwargs = {"tokenizer": tokenizer,
                          "max_len": max_len,
                          "max_seqs": max_seqs,
                          "split": split}
        dataset = ConcatDataset([CandidateToMSALegacyDataset(path, bam_file, ref_file) for path in candidate_files])
        super().__init__(dataset, **factory_kwargs)

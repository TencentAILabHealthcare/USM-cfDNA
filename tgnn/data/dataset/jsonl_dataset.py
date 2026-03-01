# Copyright (c) 2024, Tencent Inc. All rights reserved.

import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from tgnn.utils.io import get_file_timestamp, set_file_timestamp
from .index_dataset import MMIndex, MMapIndexedDataset, MMapIndexedDatasetBuilder


class JsonlDataset(torch.utils.data.Dataset):

    def __init__(self, filename):
        self.records = []
        with open(filename, encoding="utf-8") as f:
            for line in tqdm(f):
                data = json.loads(line)
                self.records.append(data)

    @classmethod
    def shuffle_dataset(cls, filename, seed=42):
        ds = cls(filename)
        indices = np.arange(len(ds))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        with open(filename, mode="xb") as f:
            for data in tqdm(ds):
                bytes = json.dumps(data).encode("utf-8")
                f.write(bytes)
        return filename

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class JsonlIndex(MMIndex):
    _HDR_MAGIC = b'JSONIDX\x00\x00'

    @classmethod
    def build_index(cls, filename):
        assert filename.endswith("jsonl"), f"file is not jsonl file: {filename}"

        def generator():
            i = 0
            while True:
                yield i
                i += 1

        sizes = []
        with open(filename, 'r', encoding="utf-8") as f:
            for _ in tqdm(generator(), desc="indexing"):
                line = f.readline()
                if not line:
                    break
                length = len(line.encode('utf-8'))
                sizes.append(length)

        idx_filename = f"{filename}.idx"
        with cls.writer(idx_filename) as index:
            index.write(sizes)

        bin_ts = get_file_timestamp(filename)
        set_file_timestamp(idx_filename, bin_ts)

    @classmethod
    def writer(cls, path, dtype=np.uint8):
        return super().writer(path, dtype)


class MMapIndexedJsonlDataset(MMapIndexedDataset):

    def read_index(self, path, skip_warmup=True):
        if os.path.isfile(path):
            assert JsonlIndex.is_index(path), f"{path} is not valid index file"
        else:
            print(f"not exist index file, start building index: {path}")
            JsonlIndex.build_index(self.bin_file)

        return JsonlIndex(path, skip_warmup=skip_warmup, verbose=self.verbose)

    def to_json(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_json(item) for item in data]
        else:
            try:
                data = json.loads(data.tobytes())
            except:
                print(data, file=sys.stderr)
                raise ValueError()
            return data

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return self.to_json(data)


class MMapIndexedJsonlDatasetBuilder(MMapIndexedDatasetBuilder):
    IndexClass = JsonlIndex
    IndexDataset = MMapIndexedJsonlDataset

    def __init__(self, out_file, dtype=np.uint8):
        super().__init__(out_file, dtype=dtype)

    def add_item(self, jsonline, encode="utf-8", **kwargs):
        if isinstance(jsonline, dict):
            jsonline = json.dumps(jsonline, **kwargs) + "\n"

        if encode is not None:
            bytes = jsonline.encode("utf-8")
        else:
            bytes = jsonline

        super().add_item(bytes)

    def add_items(self, lines, verbose=True):
        desc = os.path.basename(self._filename)
        line_iter = tqdm(lines, desc=desc) if verbose else lines
        for line in line_iter:
            self.add_item(line)
            self.end_document()

    @classmethod
    def merge_files(cls, files, filename):
        for path in tqdm(files, "scan jsonl index"):
            if not os.path.exists(path + ".idx"):
                cls.IndexClass.build_index(path)
        super(MMapIndexedJsonlDatasetBuilder, cls).merge_files(files, filename)

    def merge_file(self, filename):
        index_file = filename + ".idx"
        if not os.path.isfile(index_file):
            JsonlIndex.build_index(index_file)
        super().merge_file(filename)

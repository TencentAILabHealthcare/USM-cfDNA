# Copyright (c) 2024, Tencent Inc. All rights reserved.

import gzip
import json
import os
import re
import sys
from functools import partial

import numpy as np
import torch
from Bio import SeqIO, Seq
from tqdm import tqdm

from tgnn.utils import get_file_timestamp, set_file_timestamp
from .index_dataset import MMIndex, MMapIndexedDatasetBuilder, MMapIndexedDataset


def parse_fasta_record(path):
    """Parse the FASTA file.

    Args:
        path: path to the FASTA file (could be GZIP-compressed)

    Returns:
        dict of record {id: record}
    """
    # parse all the lines in the FASTA file
    assert os.path.exists(path), f'FASTA file does not exist: {path}'
    open_fn = partial(gzip.open, mode="rt") if path.endswith(".gz") else partial(open, mode="r")
    count = 0
    records = {}
    with open_fn(path) as ff:
        parser = SeqIO.parse(ff, 'fasta')
        for record in parser:
            rid = record.id
            if rid in records:
                rid = f"{rid}{count}"
                count += 1
            records[rid] = record

    return records


class FastaDataset(torch.utils.data.Dataset):

    def __init__(self,
                 filename):
        self.records = parse_fasta_record(filename)
        self.seq_ids = list(self.records.keys())

    def ids(self):
        return self.seq_ids

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        seq_id = index if isinstance(index, str) else self.seq_ids[index]
        record: SeqIO.SeqRecord = self.records[seq_id]
        seq = record.seq
        jsonline = re.findall(r"{(.+?)}", record.description)
        if jsonline:
            try:
                description = json.loads("{" + jsonline[0] + "}")
            except:
                print(f"cannot parse description as json line", file=sys.stderr)
                description = record.description
        else:
            description = record.description
        data = {"id": seq_id, "seq": seq, "description": description}
        return data


class FastaDatasetBuilder:
    def __init__(self, filename):
        self.ff = open(filename, mode="w")
        self.seq_ids = []

    def add_item(self,
                 seq, id,
                 name="<unknown name>",
                 description="<unknown description>"):
        if isinstance(description, dict):
            description = json.dumps(description)
        try:
            record = SeqIO.SeqRecord(Seq.Seq(seq),
                                     id=id,
                                     name=name,
                                     description=description)
        except:
            print(seq, file=sys.stderr, flush=True)
            raise f"can not write record, seq: {seq}, id: {id}, name: {name}, description: {description}"

        SeqIO.write(record, self.ff, format="fasta")

    def __del__(self):
        self.ff.close()


class FastaIndex(MMIndex):
    _HDR_MAGIC = b'FASTAIDX\x00\x00'

    @classmethod
    def build_index(cls, filename):
        assert filename.endswith(("fasta", "fa", "fna")), f"file is not fasta file: {filename}"

        def generator():
            i = 0
            while True:
                yield i
                i += 1

        sizes = []
        with open(filename, 'r', encoding="utf-8") as f:
            fasta_lines = []
            for _ in tqdm(generator(), desc="indexing"):
                line = f.readline()
                if not line:
                    if fasta_lines:
                        length = len("".join(fasta_lines).encode('utf-8'))
                        sizes.append(length)
                    break

                if line.startswith(">") and fasta_lines:
                    length = len("".join(fasta_lines).encode('utf-8'))
                    sizes.append(length)
                    fasta_lines = []
                fasta_lines.append(line)

        idx_filename = f"{filename}.idx"
        with cls.writer(f"{filename}.idx") as index:
            index.write(sizes)

        bin_ts = get_file_timestamp(filename)
        set_file_timestamp(idx_filename, bin_ts)

    @classmethod
    def writer(cls, path, dtype=np.uint8):
        return super().writer(path, dtype)


class MMapIndexedFastaDataset(MMapIndexedDataset):

    def read_index(self, path, skip_warmup=True):
        if os.path.isfile(path):
            assert FastaIndex.is_index(path), f"{path} is not valid index file"
        else:
            print(f"not exist index file, start building index: {path}")
            FastaIndex.build_index(self.bin_file)

        return FastaIndex(path, skip_warmup=skip_warmup)

    def to_fasta(self, data):
        if isinstance(data, (tuple, list)):
            return [self.to_fasta(item) for item in data]

        fasta_line = data.tobytes().decode("utf-8")
        title, *lines = fasta_line.split("\n")
        title = title[1:].rstrip()
        seq = "".join(lines)
        seq_id, *description = title.split(None, 1)
        if description:
            description = description[0]
            jsonline = re.findall(r"{(.+?)}", description)
            try:
                description = json.loads("{" + jsonline[0] + "}")
            except:
                description = description
        else:
            description = {}

        data = {"seq": seq, "id": seq_id, "description": description}
        return data

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return self.to_fasta(data)


class MMapIndexedFastaDatasetBuilder(MMapIndexedDatasetBuilder):
    IndexClass = FastaIndex

    def __init__(self, out_file):
        super().__init__(out_file, dtype=np.uint8)

    def add_item(self,
                 seq,
                 id="<unknown id>",
                 description=""):
        if isinstance(seq, dict):
            data = seq
            seq = data["seq"]
            id = data["id"]
            description = data.get("description", "")

        if isinstance(description, dict):
            description = json.dumps(description)
        try:
            record = SeqIO.SeqRecord(Seq.Seq(seq),
                                     id=id,
                                     description=description)
        except:
            print(seq, file=sys.stderr, flush=True)
            raise f"can not write record, seq: {seq}, id: {id}, description: {description}"

        fasta_line = record.format("fasta")
        bytes = fasta_line.encode("utf-8")
        super().add_item(bytes)

    def add_record(self, record: SeqIO.SeqRecord):
        fasta_line = record.format("fasta")
        bytes = fasta_line.encode("utf-8")
        super().add_item(bytes)

    def merge_file(self, filename):
        index_file = filename + ".idx"
        if not os.path.isfile(index_file):
            FastaIndex.build_index(index_file)
        super().merge_file(filename)

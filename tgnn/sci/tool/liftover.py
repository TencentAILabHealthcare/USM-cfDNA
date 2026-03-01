# Copyright (c) 2024, Tencent Inc. All rights reserved.

import gzip
import os

from Bio import SeqIO
from liftover.chain_file import ChainFile
from liftover.download_file import download_file

# url to server with chain files. This allows for mirrors of
# the UCSC chain files, but they need to adhere to the UCSC url structure
# e.g. https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz
# or https://www.example.org/folder/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz
chain_server = 'https://hgdownload.soe.ucsc.edu'


class LiftOver:
    """create a converter to map between genome builds

    Args:
        src: genome build to convert from e.g. 'hg19'
        source: genome build to convert to e.g. 'hg38'
        cache: path to cache folder, defaults to ~/.liftover
    """

    def __init__(self,
                 src: str,
                 tgt: str,
                 cache: str = None,
                 one_based=False,
                 ):
        if cache is None:
            cache = os.path.expanduser('~/.liftover')

        if not os.path.exists(cache):
            os.mkdir(cache)

        assert src != tgt, f"{src} != {tgt}"
        self.chae_dir = cache
        self.src_version = src[0].lower() + src[1:]
        self.tgt_version = tgt[0].upper() + tgt[1:]
        basename = '{}To{}.over.chain.gz'.format(self.src_version, self.tgt_version)
        chain_path = os.path.join(self.chae_dir, basename)
        if not os.path.exists(chain_path):
            url = f'{chain_server}/goldenpath/{self.src_version}/liftOver/{basename}'
            download_file(url, chain_path)

        self.tgt_ref_seq = {}
        self.src_ref_seq = {}
        self.one_based = one_based
        self.chain_file = ChainFile(chain_path, one_based=self.one_based)

    def get_seq(self, version, contig):
        ref_url = f'{chain_server}/goldenpath/{version.lower()}/chromosomes/chr{contig}.fa.gz'
        if not os.path.exists(f"{self.chae_dir}/chr{contig}.fa.gz"):
            print(f"downloading reference sequence from {ref_url}")
            download_file(ref_url, f"{self.chae_dir}/chr{contig}.fa.gz")

        if not os.path.exists(f"{self.chae_dir}/chr{contig}.fa"):
            with open(f"{self.chae_dir}/chr{contig}.fa", "wb") as f:
                f.write(gzip.open(f"{self.chae_dir}/chr{contig}.fa.gz", 'rb').read())

        seq = next(SeqIO.parse(f"{self.chae_dir}/chr{contig}.fa", "fasta")).seq
        return seq

    def get_tgt_ref_sequence(self, contig):
        if contig not in self.tgt_ref_seq:
            self.tgt_ref_seq[contig] = self.get_seq(self.tgt_version, contig)
        return self.tgt_ref_seq[contig]

    def get_src_ref_sequence(self, contig):
        if contig not in self.src_ref_seq:
            self.src_ref_seq[contig] = self.get_seq(self.src_version, contig)
        return self.src_ref_seq[contig]

    def __call__(self, **kwargs):
        return self.query(**kwargs)

    def query(self, contig, start, end=None):
        if end is None:
            result = self.chain_file.query(contig, start)
            if not result:
                # reference base deleted in new version
                return None

            tgt_pos = result[0][1]
            seq = self.get_tgt_ref_sequence(contig)[tgt_pos - 1 if self.one_based else tgt_pos].upper()
            return [tgt_pos, ], seq

        assert end > start, f"end must be greater than start"
        seq = []
        tgt_positions = []
        for pos in range(start, end):
            result = self.query(contig, pos)
            if result is None:
                return None

            tgt_pos, base = result
            tgt_positions.extend(tgt_pos)
            seq.append(base)

        return tgt_positions, "".join(seq)

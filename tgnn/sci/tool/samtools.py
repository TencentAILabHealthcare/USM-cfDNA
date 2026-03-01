# Copyright (c) 2024, Tencent Inc. All rights reserved.

import os
import sys
from datetime import datetime
import pysam


def tabix_index(filename, force=False, preset="vcf", **kwargs):
    pysam.tabix_index(filename, force=force, preset=preset, **kwargs)


def index_vcf(filename, force=False):
    idx_file = filename + ".tbi"
    if os.path.exists(idx_file) and not force:
        print(f"index file: {filename}.tbi exist, skip index or set force=true to overwrite it.", file=sys.stderr)
        return idx_file

    return tabix_index(filename, force=force, preset="vcf")


class Samtools:

    def __init__(self,
                 binary="samtools",
                 tabix="tabix",
                 num_threads=0):
        self.samtools = binary
        self.tabix = tabix
        self.num_threads = int(num_threads)

    def index(self, filename):
        if filename.endswith((".bam", ".sam", ".cram")):
            return self.index_bam(filename)
        elif filename.endswith((".fa", "fasta", "fa.gz")):
            return self.index_fasta(filename)
        elif filename.endswith((".fq.gz", "fastq")):
            return self.index_fastq(filename)
        else:
            raise f"not support file format: {filename}"

    def index_bam(self, filename):
        now = datetime.now()
        os.system(f"{self.samtools} index {filename} -@ {self.num_threads}")
        print(f"samtools index finished, total time {(datetime.now() - now).seconds} s")

    def index_fasta(self, filename):
        print(f"index fasta: {filename}")
        os.system(f"{self.samtools} faidx {filename}")

    def index_fastq(self, filename):
        print(f"index fasta: {filename}")
        os.system(f"{self.samtools} fqidx {filename}")

    def index_vcf(self, filename):
        os.system(f"{self.tabix} index {filename}")

    def sort(self, filename, output=None):
        if output is None:
            output = filename[:3] + "sort.bam"

        os.system(f"{self.samtools} sort {filename} -o {output} -@ {self.num_threads}")

    def split_by_reference(self, filename):
        os.system(f"{self.samtools} split {filename} --reference -@ {self.num_threads}")

    def extract_regions(self, filename, output, regions):
        os.system(f"{self.samtools} view -b -@ {self.num_threads} {filename} " + " ".join(regions) + f" {output}")

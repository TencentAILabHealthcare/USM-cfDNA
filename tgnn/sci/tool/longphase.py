# Copyright (c) 2025, Tencent Inc. All rights reserved.

import os

from tgnn.sci.parser.sam_parsing import get_platform


class LongPhase:

    def __init__(self,
                 reference,
                 binary="longphase",
                 num_threads=None):
        self.binary = binary
        self.reference = reference
        self.num_threads = num_threads

    def phase(self, bam_file, vcf_file, output):
        platform = get_platform(bam_file)
        cmd = f"{self.binary} -s {vcf_file} -b {bam_file} -r {self.reference} -o {output} --{platform}"
        if self.num_threads is not None:
            cmd = cmd + f" -t {self.num_threads}"

        return os.system(cmd)
# Copyright (c) 2025, Tencent Inc. All rights reserved.

import os
import subprocess
from datetime import datetime
from tgnn.utils.io import is_tool


class BWA:
    def __init__(self,
                 binary="bwa",
                 num_threads=1,
                 samtools="samtools"):
        assert is_tool(binary), f"{binary} is not a valid tool"
        self.binary = binary
        self.num_threads = num_threads
        self.samtools = samtools

    def index(self, filename):
        cmd = [self.binary, "index", filename]
        print(" ".join(cmd))
        return subprocess.run(cmd)

    def mem(self, ref_file, in1, in2, out=None, sorted=False, options=None):
        now = datetime.now()

        cmd = [self.binary, "mem", "-t", f"{self.num_threads}", ref_file, in1, in2]
        if options is not None:
            for key, value in options.items():
                cmd.extend([key, value])

        if out is None:
            out = os.path.basename(in1).split(".")[0] + ".bam"

        assert out.endswith((".bam", "sam")), f"{out} is not a valid output file"
        if out.endswith(".bam"):
            assert is_tool(self.samtools), f"{self.samtools} is not a valid tool"
            print(" ".join(cmd))
            p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            if sorted:
                cmd2 = [self.samtools, "sort", "-@", f"{self.num_threads}", "-o", out]
            else:
                cmd2 = [self.samtools, "view", "-@", f"{self.num_threads}", "-o", out]
            print(" ".join(cmd2))
            out = subprocess.run(cmd2, stdin=p1.stdout, stdout=subprocess.PIPE)
        else:
            cmd.extend(["-o", out])
            print(" ".join(cmd))
            out = subprocess.run(cmd)

        print(f"bwa mem finished, total time {(datetime.now() - now).seconds} s")
        return out

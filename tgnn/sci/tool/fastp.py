# Copyright (c) 2025, Tencent Inc. All rights reserved.

import os
from datetime import datetime
import subprocess

class Fastp:
    def __init__(self, binary="fastp", num_threads=3):
        self.fastp = binary
        self.num_threads = num_threads

    @property
    def __version__(self):
        run = subprocess.run(["fastp", "--version"], capture_output=True)
        version = run.stderr.decode("utf-8").strip().split()[-1]
        return version

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def run(self, in1, in2, in1_out=None, in2_out=None, json_report=None, html_report=None, options=None):
        data_dir = os.path.dirname(os.path.abspath(in1_out))
        if in1_out is None:
            format = ".fq.gz" if in1.endswith(".gz") else ".fastq"
            in1_out = in1.split(".")[0] + format

        if in2_out is None:
            format = ".fq.gz" if in1.endswith(".gz") else ".fastq"
            in2_out = in2.split(".")[0] + format

        if json_report is None:
            json_report = f"{data_dir}/fastp.json"

        if html_report is None:
            html_report = f"{data_dir}/fastp.html"

        cmds = [self.fastp, "-i", in1, "-I", in2, "-o", in1_out, "-O", in2_out, "-w", str(self.num_threads),
                "-j", json_report, "-h", html_report]
        if options is not None:
            for key, value in options.items():
                assert key not in cmds
                cmds.extend([key, value])
        print(" ".join(cmds))
        now = datetime.now()
        out = subprocess.run(cmds)
        print(f"fastp finished, total time {(datetime.now() - now).seconds} s")
        return out
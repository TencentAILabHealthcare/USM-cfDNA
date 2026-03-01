# Copyright (c) 2025, Tencent Inc. All rights reserved.

import subprocess
from datetime import datetime
from tgnn.utils.io import is_tool


class GATK4:
    """
    conda config --add channels bioconda
    conda install gatk4=4.6.1
    conda install openjdk=17.0.14
    """
    def __init__(self, binary="gatk", num_threads=1):
        assert is_tool(binary), f"{binary} is not a valid GATK binary, please install gatk4 first!"
        self.binary = binary
        self.num_threads = num_threads

    def mark_duplicates_spark(self,
                              bam_file,
                              output=None,
                              options=None
                              ):
        now = datetime.now()
        if output is None:
            output =  bam_file[:-4] + ".markdup.bam"

        assert output.endswith(".bam"), f"{output} is not a .bam file!"
        cmd = [self.binary, "MarkDuplicatesSpark", "-I", bam_file, "-O", output,
               "-M", output[:-4] + ".metrics", "--spark-master", f"local[{self.num_threads}]"
               ]
        if options is not None:
            for key, value in options.items():
                cmd.extend([key, value])

        print(" ".join(cmd))
        out = subprocess.run(cmd)
        print(f"GATK4 MarkDuplicatesSpark finished, total time {(datetime.now() - now).seconds} s")
        return out

    def fix_mate_information(self, bam_file, output=None, create_index=True):
        now = datetime.now()
        if output is None:
            output =  bam_file[:-4] + ".fixmate.bam"

        cmd = [self.binary, "FixMateInformation", "-I", bam_file, "-O", output]
        if create_index:
            cmd.extend(["--CREATE_INDEX", "true"])

        print(" ".join(cmd))
        out = subprocess.run(cmd)
        print(f"GATK4 FixMateInformation finished, total time {(datetime.now() - now).seconds} s")
        return out

    def fix_tags(self, ref_file, bam_file, output=None, create_index=True):
        now = datetime.now()
        if output is None:
            output =  bam_file[:-4] + ".fixtags.bam"

        cmd = [self.binary, "SetNmMdAndUqTags", "-R", ref_file, "-I", bam_file, "-O", output]
        if create_index:
            cmd.extend(["--CREATE_INDEX", "true"])

        print(" ".join(cmd))
        out = subprocess.run(cmd)
        print(f"GATK4 FixMateInformation finished, total time {(datetime.now() - now).seconds} s")
        return out
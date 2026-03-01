# Copyright (c) 2024, Tencent Inc. All rights reserved.

import gzip
from functools import partial

CHRS = [f"chr{i}" for i in (list(range(1, 23)) + ["X", "Y"])]


def parse_maf(filename, chrs=None, max_len=None, sample_ids=None):
    sits = []
    chrs = chrs or CHRS
    open_fn = partial(gzip.open, mode="rt") if filename.endswith(".gz") else open
    with open_fn(filename) as f:
        header = None
        for line in f:
            if line.startswith("#"):
                continue
            else:
                header = line
                break

        fieldnames = header.split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = dict(zip(fieldnames, line.split("\t")))
            sample_id = item["Tumor_Sample_Barcode"]
            if sample_ids is not None and sample_id not in sample_ids:
                continue

            chr = item["Chromosome"]
            chr = chr if chr.startswith("chr") else f"chr{chr}"
            if chr not in chrs:
                continue

            ncbi_build = item["NCBI_Build"]
            assert ncbi_build in ("GRCh38", "GRCh37")
            ref_start = int(item["Start_Position"]) - 1  # zero-based
            ref_end = int(item["End_Position"])
            ref_seq = item["Reference_Allele"]
            var_seq = item["Tumor_Seq_Allele2"]
            variant_type = item["Variant_Type"]

            # in maf ins end - start = 1
            if variant_type in ("INS",):
                ref_end = ref_start + 1

            seq_len = ref_end - ref_start
            # avoid to large deletion and insertion
            if max_len is not None and max_len < seq_len:
                seq_len = max_len
                ref_end = min(ref_end, ref_start + seq_len)
                ref_seq = ref_seq[:seq_len]
                var_seq = var_seq[:seq_len]

            data = {
                "id": f"{sample_id}-{chr}-{ref_start}",
                "sample_id": sample_id,
                "ref_seq": ref_seq,
                "var_seq": var_seq,
                "ref_start": ref_start,
                "ref_end": ref_end,
                "chr": chr,
                "variant_type": variant_type,
                "variant_classification": item["Variant_Classification"],
                "hugo_symbol": item["Hugo_Symbol"],
                "entrez_gene_id": item["Entrez_Gene_Id"],
                "ncbi_build": ncbi_build
            }
            sits.append(data)

    return sits

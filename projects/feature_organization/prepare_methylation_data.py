# Copyright (c) 2025, Tencent Inc. All rights reserved.

import sys
sys.path.append(".")

from collections import defaultdict
import pandas as pd
from tgnn.utils.pack_files import open_resource_text

def load_sites(pack_path):

    with open_resource_text(pack_path, "methylation_sites", encoding="utf-8", errors="strict") as f:
        sites = pd.read_csv(f, sep="\t")

    groups = sites.pop("group").tolist()
    group_indices = defaultdict(list)
    for i, group in enumerate(groups):
        for g in group.split(","):
            group_indices[g].append(i)

    return sites, group_indices


def read_sample(sample_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(sample_path, usecols=["chr", "position", "probability"])
    except Exception:
        df = pd.read_csv(sample_path)
        cols = [c for c in ("chr", "position", "probability") if c in df.columns]
        return df[cols]


def get_methylation_feature(met_file, site_file, window=1_000_000, calibrate=True, fillna=-1):
    if met_file.endswith("csv"):
        sample_df = read_sample(met_file)
    else:
        sample_df = pd.read_json(met_file, lines=True)
    sample_df = sample_df.drop_duplicates(subset=["chr", "position"])
    sit_df, group_indices = load_sites(site_file)
    # filter methylation sit and merge sit probability and annotation
    df = pd.merge(sit_df, sample_df[["chr", "position", "probability"]], on=["chr", "position"], how="left")
    if calibrate:
        df["probability"] = df["probability"] - df["reference"]

    starts = (df["position"] // window) * window
    ends = starts + window
    df["bin_id"] = df["chr"] + ":" + starts.astype(str) + "-" + ends.astype(str)
    features = {}
    for name, indices in group_indices.items():
        bin_group = df.iloc[indices].groupby("bin_id")
        prob = bin_group["probability"].mean(numeric_only=True)
        prob.index = name + "_" + prob.index.astype(str)
        features.update(prob.to_dict())

    features = pd.Series(features).fillna(fillna).to_dict()
    return features

# Copyright (c) 2025, Tencent Inc. All rights reserved.

import sys
sys.path.append(".")

import numpy as np
import pandas as pd
import tqdm
from tgnn.utils import jload

end_motif_types = ['ACG', 'TCG', 'CCG', 'GCG', 'CGA', 'CGT', 'CGC', 'CGG']


def get_cnv_feature(path):
    features = jload(path)
    features = {**features["bin"], **features["arm"]}
    return features


def get_end_motif_feature(path, epsilon=1e-8):
    data = jload(path)
    feature_dict = {}
    total_count = np.zeros((len(end_motif_types),))
    for group in data:
        # cleavage profile feature
        watson_ends_array = np.array(data[group]['watson_ends'])
        crick_ends_array = np.array(data[group]['crick_ends'])[::-1]
        cleavage_profile = (watson_ends_array + crick_ends_array) / (np.array(data[group]['profiles']) + epsilon)
        feature_dict[f"cle_{group}"] = cleavage_profile

        # end motif feature
        end_motifs = np.array([data[group]['motifs'][item] for item in end_motif_types], dtype=np.int32)
        feature_dict[f"end_{group}"] = end_motifs

        # NCG/CGN
        NCG_count = np.sum(end_motifs[0:4])
        CGN_count = np.sum(end_motifs[4:8])
        feature_dict[f"ratio_{group}"] = CGN_count / (NCG_count + epsilon)
        # update total counts
        total_count = total_count + end_motifs

    total_count = total_count.sum()
    features = {}
    for key, values in feature_dict.items():
        if key.startswith("cle_"):
            for i, v in enumerate(values):
                features[f"{key}_{i}"] = v

        if key.startswith("end_"):
            for i, v in enumerate(values):
                features[f"{key}_{end_motif_types[i]}"] = v / (total_count + epsilon)

        if key.startswith("ratio_"):
            # ratio likely becomes a single column; flatten to 1D
            features[key] = values

    return features

# Copyright (c) 2025, Tencent Inc. All rights reserved.

import sys
sys.path.append(".")

import argparse
import pandas as pd
from tgnn.utils import jdump
from projects.feature_organization.prepare_mutation_data import get_mutation_feature
from projects.feature_organization.prepare_methylation_data import get_methylation_feature
from projects.feature_organization.prepare_other_features import get_end_motif_feature, get_cnv_feature

def get_age_sex(sample_id, csv_file):
    df = pd.read_csv(csv_file)
    row = df[df['sample_id'] == sample_id]
    if not row.empty:
        age = row['age'].values[0]
        sex = row['sex'].values[0]
        return age, sex
    return -1, 1

def main(args):
    meta_info = args.meta
    sample_id = args.name
    age, sex = get_age_sex(sample_id, meta_info)
    feats = {
        "sample_id": sample_id,
        "age": age,
        "sex": sex
    }
    met_feats = get_methylation_feature(args.met, args.site, )
    feats.update({f"met_{k}": v for k, v in met_feats.items()})
    mut_feats = get_mutation_feature(args.mut, args.ref)
    feats.update({f"mut_{k}": v for k, v in mut_feats.items()})
    em_feats = get_end_motif_feature(args.em)
    feats.update({f"end_{k}": v for k, v in em_feats.items()})
    em_all_cg_feats = get_end_motif_feature(args.em_a)
    feats.update({f"end_{k}": v for k, v in em_all_cg_feats.items()})
    cna_feats = get_cnv_feature(args.cna)
    feats.update({f"cna_{k}": v for k, v in cna_feats.items()})
    
    if args.output.endswith('.parquet'):
        pd.DataFrame([feats]).to_parquet(args.output, index=False)
    elif args.output.endswith('.json'):
        jdump(feats, args.output)
    else:
        pd.DataFrame([feats]).to_csv(args.output, index=False)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--meta", help="path to meta info", required=True)
    args.add_argument("--met", help="path to methylation file", required=True)
    args.add_argument("--mut", help="path to mutation file", required=True)
    args.add_argument("--em", help="path to end motif file", required=True)
    args.add_argument("--em-a", help="path to all cpg end motif file", required=True)
    args.add_argument("--cna", help="path to CNA file", required=True)
    args.add_argument("--site", help="path to cpg sit file", required=True)
    args.add_argument("--ref", help="path to reference file", required=True)
    args.add_argument("--name", default="sample0", help="sample name", required=True)
    args.add_argument("-o", "--output", help="path to output file", required=True)
    args = args.parse_args()
    main(args)
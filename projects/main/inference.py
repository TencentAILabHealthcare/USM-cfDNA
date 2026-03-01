# Copyright (c) 2025, Tencent Inc. All rights reserved.
import argparse
import sys

import pandas as pd

sys.path.append(".")

import os
import xgboost as xgb
from tgnn.utils import jdump, jload

class MECEDInferenceModel:
    def __init__(self):
        self.model = xgb.XGBClassifier()
        self.is_loaded = False
        self.feature_names = None
        self.num_class = 11

    def load_model(self, ckpt):
        print(f"Loading model from {ckpt}")
        assert os.path.exists(ckpt), f"{ckpt} does not exist"
        self.model.load_model(ckpt)
        self.is_loaded = True

        self.feature_names = self.model.get_booster().feature_names
        return self

    def get_feature_names(self):
        assert self.is_loaded, "model must be loaded"
        return self.feature_names

    def __call__(self, samples: pd.DataFrame):
        assert self.is_loaded, f"please load model first"

        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(samples.columns)
            if missing_features:
                for feature in missing_features:
                    samples[feature] = 0
            samples['age'] = samples['age'].astype(int)
            samples['sex'] = samples['sex'].astype(int)

            samples = samples[self.feature_names]
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(samples)
        return predictions

def load_samples(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".json"):
        df = pd.DataFrame([jload(path)])
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported file type: {path}")
    return df


def main(args):
    model = MECEDInferenceModel()
    model.load_model(args.ckpt)
    samples = load_samples(args.input)
    predictions = model(samples)
    sample_ids = samples["sample_id"].values
    for sample_id, prob in zip(sample_ids, predictions):
        p_healthy = (prob[0] + prob[1])
        p_cancer = 1 - p_healthy
        outputs_mced = {
            "sample_id": sample_id,
            "probability_healthy": p_healthy,
            "probability_cancer": p_cancer,
        }

        outputs_cso = {
            "sample_id": sample_id,
            "probability_healthy": p_healthy,
            "probability_lung_cancer": prob[2],
            "probability_gastric_cancer": prob[3],
            "probability_breast_cancer": prob[4],
            "probability_pancreatic_cancer": prob[5],
            "probability_ovarian_cancer": prob[6],
            "probability_bile_duct_cancer": prob[7],
            "probability_colorectal_cancer": prob[8],
            "probability_liver_cancer": prob[9],
            "probability_head_and_neck_cancer": prob[10],
        }

        print(f"Cancer probability of {sample_id}: {p_cancer:0.4f}")

        jdump(outputs_mced, f"{args.output_dir}/{sample_id}_mced.json")
        jdump(outputs_cso, f"{args.output_dir}/{sample_id}_cso.json")

    print(f"Saved mced and cso results to {args.output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Multi-Cancer Early Detection")
    parser.add_argument("-i", "--input",
                        required = True,
                        metavar="FILE", help="path to sample json or csv file")
    parser.add_argument("--ckpt", "-m",
                        required=True,
                        help="path to ensemble model directory")
    parser.add_argument("-o", "--output-dir", required=True, help="path to output dir")
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
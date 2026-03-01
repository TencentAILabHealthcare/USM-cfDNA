#!/usr/bin/env bash
set -euo pipefail

FEAT_FILE=""
XGB_CKPT=""
PRED_DIR=""

usage() {
  echo "Usage:"
  echo "  $0 --ckpt <MODEL_BIN> --pred-dir <PRED_DIR> --feat-file <FEATURE_TABLE>"
  echo ""
  echo "Required:"
  echo "  --ckpt <MODEL_BIN>               XGBoost model file (e.g., ckpt/model.bin)"
  echo "  --pred-dir <PRED_DIR>            Prediction output directory (e.g., output/prediction)"
  echo "  --feat-file <FEATURE_TABLE>      Precomputed feature table"
  echo ""
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      XGB_CKPT="${2:-}"; shift 2;;
    --pred-dir)
      PRED_DIR="${2:-}"; shift 2;;
    --feat-file)
      FEAT_FILE="${2:-}"; shift 2;;
    -h|--help)
      usage;;
    *)
      echo "Unknown arg: $1"
      usage;;
  esac
done

[[ -n "$XGB_CKPT" ]] || { echo "Missing --ckpt"; usage; }
[[ -n "$PRED_DIR"  ]] || { echo "Missing --pred-dir"; usage; }
[[ -n "$FEAT_FILE" ]] || { echo "Missing --feat-file"; usage; }

[[ -f "$FEAT_FILE" ]] || { echo "Error: missing feature table: $FEAT_FILE"; exit 1; }
[[ -f "$XGB_CKPT"  ]] || { echo "Error: missing model: $XGB_CKPT"; exit 1; }

mkdir -p "$PRED_DIR"


echo "FEAT_FILE=$FEAT_FILE"
echo "XGB_CKPT=$XGB_CKPT"
echo "PRED_DIR=$PRED_DIR"


echo "[INFO] infer MCED & CSO from precomputed feature table"
python projects/main/inference.py \
  -i "${FEAT_FILE}" \
  --ckpt "${XGB_CKPT}" \
  -o "${PRED_DIR}"
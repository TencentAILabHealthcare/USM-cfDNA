#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=""
SAMPLE_ID=""
RESOURCE_DIR=""
CKPT_DIR=""
OUTPUT_DIR=""
MET_HOST_GPU_NUM=""
MET_NUM_THREADS=""
MET_BATCH_SIZE=""
MUT_HOST_GPU_NUM=""
MUT_NUM_THREADS=""
MUT_BATCH_SIZE=""

usage() {
  echo "Usage: $0 --data-dir <DATA_DIR> --sample-id <SAMPLE_ID> --resource-dir <RESOURCE_DIR> --ckpt-dir <CKPT_DIR> --output-dir <OUTPUT_DIR> [OPTIONS]"
  echo "Required:"
  echo "  --data-dir <DATA_DIR>            Input data directory  (expects <DATA_DIR>/<SAMPLE_ID>.bam and <DATA_DIR>/meta.csv)"
  echo "  --sample-id <SAMPLE_ID>          Sample ID"
  echo "  --resource-dir <RESOURCE_DIR>    Resource directory"
  echo "  --ckpt-dir <CKPT_DIR>            Checkpoint directory"
  echo "  --output-dir <OUTPUT_DIR>        Output directory"
  echo "Options:"
  echo "  --met-host-gpu-num <NUM>         MET GPU number"
  echo "  --met-num-threads <NUM>          MET thread number"
  echo "  --met-batch-size <SIZE>          MET batch size"
  echo "  --mut-host-gpu-num <NUM>         MUT GPU number"
  echo "  --mut-num-threads <NUM>          MUT thread number"
  echo "  --mut-batch-size <SIZE>          MUT batch size"
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="${2:-}"; shift 2;;
    --sample-id)
      SAMPLE_ID="${2:-}"; shift 2;;
    --resource-dir)
      RESOURCE_DIR="${2:-}"; shift 2;;
    --ckpt-dir)
      CKPT_DIR="${2:-}"; shift 2;;
    --output-dir)
      OUTPUT_DIR="${2:-}"; shift 2;;
    --met-host-gpu-num)
      MET_HOST_GPU_NUM="${2:-}"; shift 2;;
    --met-num-threads)
      MET_NUM_THREADS="${2:-}"; shift 2;;
    --met-batch-size)
      MET_BATCH_SIZE="${2:-}"; shift 2;;
    --mut-host-gpu-num)
      MUT_HOST_GPU_NUM="${2:-}"; shift 2;;
    --mut-num-threads)
      MUT_NUM_THREADS="${2:-}"; shift 2;;
    --mut-batch-size)
      MUT_BATCH_SIZE="${2:-}"; shift 2;;
    -h|--help)
      usage;;
    *)
      echo "Unknown arg: $1"
      usage;;
  esac
done

[[ -n "$DATA_DIR" ]] || { echo "Missing --data-dir"; usage; }
[[ -n "$SAMPLE_ID" ]] || { echo "Missing --sample-id"; usage; }
[[ -n "$RESOURCE_DIR" ]] || { echo "Missing --resource-dir"; usage; }
[[ -n "$CKPT_DIR" ]]     || { echo "Missing --ckpt-dir"; usage; }
[[ -n "$OUTPUT_DIR" ]]   || { echo "Missing --output-dir"; usage; }

# Set default values if not provided
MET_HOST_GPU_NUM=${MET_HOST_GPU_NUM:-8}
MET_NUM_THREADS=${MET_NUM_THREADS:-307}
MET_BATCH_SIZE=${MET_BATCH_SIZE:-196}
MUT_HOST_GPU_NUM=${MUT_HOST_GPU_NUM:-1}
MUT_NUM_THREADS=${MUT_NUM_THREADS:-38}
MUT_BATCH_SIZE=${MUT_BATCH_SIZE:-256}

echo "DATA_DIR=$DATA_DIR"
echo "SAMPLE_ID=$SAMPLE_ID"
echo "RESOURCE_DIR=$RESOURCE_DIR"
echo "CKPT_DIR=$CKPT_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "MET_HOST_GPU_NUM=$MET_HOST_GPU_NUM"
echo "MET_NUM_THREADS=$MET_NUM_THREADS"
echo "MET_BATCH_SIZE=$MET_BATCH_SIZE"
echo "MUT_HOST_GPU_NUM=$MUT_HOST_GPU_NUM"
echo "MUT_NUM_THREADS=$MUT_NUM_THREADS"
echo "MUT_BATCH_SIZE=$MUT_BATCH_SIZE"

mkdir -p "${OUTPUT_DIR}/prediction"


# meta file
META_FILE=${DATA_DIR}/meta.csv
# bam path
BAM_FILE=${DATA_DIR}/${SAMPLE_ID}.bam

# checkpoint
MET_CKPT=${CKPT_DIR}/methylation_calling.pth
MUT_CKPT=${CKPT_DIR}/variant_calling.pth
XGB_CKPT=${CKPT_DIR}/model.bin

# reference
REF_FILE=${RESOURCE_DIR}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna
# sites
BED_FILE=${RESOURCE_DIR}/assets.pack
PON_FILE=${RESOURCE_DIR}/hg38_pon.json

# methylation and mutation inference script
MET_SCRIPT=projects/methylation_calling/infer_bam.py
MUT_SCRIPT=projects/variant_calling/infer_bam.py


# set feature dir
FEATURE_DIR=${OUTPUT_DIR}/features
# set methylation output path
MET_OUTPUT_DIR=${FEATURE_DIR}/methylation/${SAMPLE_ID}
MET_FILE=${MET_OUTPUT_DIR}/methylation_prediction.jsonl

# set mutation output path
MUT_OUTPUT_DIR=${FEATURE_DIR}/mutation/${SAMPLE_ID}
MUT_FILE=${MUT_OUTPUT_DIR}/variant_prediction.vcf

# set end motif output path
EM_FILE=${FEATURE_DIR}/end_motif/${SAMPLE_ID}_end_motif.json
EM_ALL_CG_FILE=${FEATURE_DIR}/end_motif_all_cg/${SAMPLE_ID}_end_motif.json

# set CNV output path
CNV_FILE=${FEATURE_DIR}/cna/${SAMPLE_ID}_cna.json

# set feature tabel output path
FEAT_FILE=${FEATURE_DIR}/combined_feature/${SAMPLE_ID}_feature.parquet
# Set prediction output path
PRED_DIR=${OUTPUT_DIR}/prediction

mkdir -p "${PRED_DIR}" \
         "${FEATURE_DIR}/combined_feature"

for f in "$META_FILE" "$BAM_FILE" "$MET_CKPT" "$MUT_CKPT" "$XGB_CKPT" "$REF_FILE" "$BED_FILE" "$PON_FILE"; do
  [[ -f "$f" ]] || { echo "Error: missing file: $f"; exit 1; }
done

echo "[STEP 1] inference methylation"
# set number of host and number of GPU in machine
HOST_NUM=1

if [ $HOST_NUM -eq 1 ]; then
  CHIEF_IP=127.0.0.1
fi

SEQ_LEN=256

deepspeed --num_nodes=$HOST_NUM --num_gpus=$MET_HOST_GPU_NUM --master_addr $CHIEF_IP $MET_SCRIPT \
    -r ${REF_FILE} \
    -b ${BAM_FILE} \
    --ckpt ${MET_CKPT} \
    -rs ${BED_FILE} \
    -o ${MET_OUTPUT_DIR} \
    -bs ${MET_BATCH_SIZE} \
    -sl ${SEQ_LEN} \
    -n ${MET_NUM_THREADS} \
    --overwrite

# check output exist or exit
if [ ! -f "$MET_FILE" ]; then
    echo "Error: File '$MET_FILE' is not exist"
    exit 1
fi

echo "[STEP 2] inference mutation"
# set number of host and number of GPU in machine
HOST_NUM=1

if [ $HOST_NUM -eq 1 ]; then
  CHIEF_IP=127.0.0.1
fi

MIN_DEPTH=4
CHRS=(chr{1..22})
CHRS+=(chrX chrY)

deepspeed --num_nodes=$HOST_NUM --num_gpus=$MUT_HOST_GPU_NUM  --master_addr $CHIEF_IP  $MUT_SCRIPT \
    -r ${REF_FILE} \
    -b ${BAM_FILE} \
    --ckpt ${MUT_CKPT} \
    --min_depth ${MIN_DEPTH}  \
    -c ${CHRS[@]} \
    -o ${MUT_OUTPUT_DIR} \
    -bs ${MUT_BATCH_SIZE} \
    -n ${MUT_NUM_THREADS}

# check output exist or exit
if [ ! -f "$MUT_FILE" ]; then
    echo "Error: File '$MUT_FILE' is not exist"
    exit 1
fi


echo "[STEP 3] analyse end motif"
NUM_THREADS=32

python projects/end_motif/compute_end_motif.py \
    -i ${BAM_FILE} \
    -b ${BED_FILE} \
    -r "end_motif" \
    -o ${EM_FILE} \
    -t ${NUM_THREADS}

python projects/end_motif/compute_end_motif.py \
    -i ${BAM_FILE} \
    -b ${BED_FILE} \
    -r "end_motif_all_cg" \
    -o ${EM_ALL_CG_FILE} \
    -t ${NUM_THREADS}

echo "[STEP 4] analyse copy number"
MAP_THRESHOLD=0.8
QUALITY=30

python projects/cnv_calling/compute_cna.py \
    --input ${BAM_FILE} \
    --window 1000kb \
    --mappability_threshold ${MAP_THRESHOLD} \
    --min_mapq ${QUALITY} \
    --pon_file ${PON_FILE} \
    --output ${CNV_FILE} \
    --visualize

echo "[STEP 5] computing features"
python projects/main/feature_transform.py \
    --met ${MET_FILE} \
    --mut ${MUT_FILE} \
    --em ${EM_FILE} \
    --em-a ${EM_ALL_CG_FILE} \
    --cna ${CNV_FILE} \
    --ref ${REF_FILE} \
    --site ${BED_FILE} \
    --name ${SAMPLE_ID} \
    --meta ${META_FILE} \
    -o ${FEAT_FILE}

echo "[STEP 6] computing probability"
python projects/main/inference.py \
    -i ${FEAT_FILE} \
    --ckpt ${XGB_CKPT} \
    -o ${PRED_DIR}
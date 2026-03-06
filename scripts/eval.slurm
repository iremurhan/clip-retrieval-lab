#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/%j/eval_slurm.out
#SBATCH --error=/users/beyza.urhan/experiments/results/%j/eval_slurm.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail

CONFIG_FILE="${1:-configs/config_coco.yaml}"
CHECKPOINT="${2}"
SPLIT="${3:-test}"
MODE="${4:-normal}"

# Çıktı klasörünü ayarla
JOB_DIR="/output/results/eval_${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Mode:          ${MODE}"
echo "Checkpoint:    ${CHECKPOINT}"
echo "Output Dir:    ${JOB_DIR}"
echo "=========================================="

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

# SADECE ECCV modu seçiliyse kütüphaneyi anında (runtime) kur
if [ "$MODE" == "eccv" ]; then
    echo "ECCV Mode detected. Installing eccv_caption library..."
    pip install eccv_caption
    ECCV_FLAG="--eccv"
else
    ECCV_FLAG=""
fi

echo "Running Evaluation..."

python3 -u tools/eval.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" \
    $ECCV_FLAG \
    > "$JOB_DIR/eval.log" 2>&1

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation failed with exit code: $EVAL_EXIT_CODE"
fi

exit $EVAL_EXIT_CODE

#!/bin/bash
# =============================================================================
# Fold a *.recovered.json (produced by retry_failed_rewrites.slurm) into the
# canonical caption_rewrites_<model>.json. Atomic dict update + atomic file
# replace. Also rewrites the *.failed.jsonl to drop now-recovered sentids.
#
# Pre-flight: the fold tool refuses to overwrite existing canonical entries.
#
# Positional args:
#   $1  DATASET     "coco" or "flickr30k"
#   $2  MODEL_SLUG  e.g. "meta-llama-3-8b-instruct" (no "/" — the slug used in
#                   the rewrites filename)
#
# Examples:
#   sbatch legacy/paraphrase_generation/fold_recovered_rewrites.slurm \
#       coco meta-llama-3-8b-instruct
#   sbatch legacy/paraphrase_generation/fold_recovered_rewrites.slurm \
#       flickr30k mistral-7b-instruct-v0.2
# =============================================================================
#SBATCH --time=00:15:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=48G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j_fold.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j_fold.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail

DATASET="${1:?Usage: sbatch <script> <coco|flickr30k> <model_slug>}"
MODEL_SLUG="${2:?Usage: sbatch <script> <dataset> <model_slug>}"

case "$DATASET" in
    coco|flickr30k) ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

DATASET_DIR="/workspace/datasets/${DATASET}"
CANONICAL="${DATASET_DIR}/caption_rewrites_${MODEL_SLUG}.json"
RECOVERED="${DATASET_DIR}/caption_rewrites_${MODEL_SLUG}.recovered.json"
FAILED="${DATASET_DIR}/caption_rewrites_${MODEL_SLUG}.failed.jsonl"

for f in "$CANONICAL" "$RECOVERED"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file missing: $f"
        exit 1
    fi
done

echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Canonical: ${CANONICAL}"
echo "Recovered: ${RECOVERED}"
echo "Failed:    ${FAILED}"
echo "=========================================="

cd /workspace

EXTRA=()
if [ -f "$FAILED" ]; then
    EXTRA+=(--failed "$FAILED")
fi

python3 -u tools/fold_recovered_rewrites.py \
    --canonical "$CANONICAL" \
    --recovered "$RECOVERED" \
    "${EXTRA[@]}"

echo ""
echo "=========================================="
echo "Fold complete. Verify:"
echo "  python3 -c \"import json; print(len(json.load(open('${CANONICAL}'))))\""
echo "Optional: delete the now-stale recovered.json:"
echo "  rm /users/beyza.urhan/experiments/datasets/${DATASET}/caption_rewrites_${MODEL_SLUG}.recovered.json"
echo "=========================================="

#!/bin/bash
# =============================================================================
# Retry K<num_rewrites paraphrase failures with elevated sampling on the SAME
# model. Reads <ds>/caption_rewrites_<model_slug>.failed.jsonl and writes
# .recovered.json + .still_failed.jsonl in the same dir. Use
# tools/fold_recovered_rewrites.py (or the fold slurm wrapper) to fold the
# recovered file back into the canonical.
#
# Positional args:
#   $1  DATASET   "coco" or "flickr30k"
#   $2  MODEL     HuggingFace model ID (must match the original generation)
#
# Examples:
#   sbatch legacy/paraphrase_generation/retry_failed_rewrites.slurm \
#       coco meta-llama/Meta-Llama-3-8B-Instruct
#   sbatch legacy/paraphrase_generation/retry_failed_rewrites.slurm \
#       flickr30k mistralai/Mistral-7B-Instruct-v0.2
#
# Override batch size: BATCH_SIZE=4 sbatch ...
# =============================================================================
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/.cache/huggingface:/root/.cache/huggingface
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j_retry.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j_retry.out
#SBATCH --container-env=HF_TOKEN,HUGGING_FACE_HUB_TOKEN,PYTHONPATH=/workspace

set -euo pipefail

DATASET="${1:?Usage: sbatch <script> <coco|flickr30k> <hf_model_id>}"
MODEL="${2:?Usage: sbatch <script> <dataset> <hf_model_id>}"
BATCH_SIZE="${BATCH_SIZE:-8}"

case "$DATASET" in
    coco)      DATASET_DIR="/workspace/datasets/coco" ;;
    flickr30k) DATASET_DIR="/workspace/datasets/flickr30k" ;;
    *) echo "Unknown dataset: $DATASET (expected coco or flickr30k)"; exit 1 ;;
esac

# Mirror generate_rewrites.slurm's slug derivation
MODEL_SLUG=$(echo "$MODEL" | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]')

FAILED_PATH="${DATASET_DIR}/caption_rewrites_${MODEL_SLUG}.failed.jsonl"
RECOVERED_PATH="${DATASET_DIR}/caption_rewrites_${MODEL_SLUG}.recovered.json"
STILL_FAILED_PATH="${DATASET_DIR}/caption_rewrites_${MODEL_SLUG}.still_failed.jsonl"

# Pre-flight: refuse to start if input is missing (avoids opening any model
# weights for nothing, and prevents the "Kill task failed" pyxis path).
if [ ! -f "$FAILED_PATH" ]; then
    echo "ERROR: failed.jsonl not found: $FAILED_PATH"
    echo "For COCO with newly-generated rewrites: it is produced by the main"
    echo "generation pass. For legacy Flickr: run tools/clean_canonical_rewrites.py"
    echo "first (via scripts/setup/clean_canonical_rewrites.slurm) to derive it."
    exit 1
fi

JOB_DIR="/output/results/rewrites/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Dataset:          ${DATASET}"
echo "Model:            ${MODEL}"
echo "Failed input:     ${FAILED_PATH}"
echo "Recovered output: ${RECOVERED_PATH}"
echo "Still failed:     ${STILL_FAILED_PATH}"
echo "Batch size:       ${BATCH_SIZE}"
echo "=========================================="

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

set +e
python3 -u tools/retry_failed_rewrites.py \
    --failed_path "$FAILED_PATH" \
    --recovered_path "$RECOVERED_PATH" \
    --still_failed_path "$STILL_FAILED_PATH" \
    --model "$MODEL" \
    --num_rewrites 2 \
    --batch_size "$BATCH_SIZE" 2>&1 | tee "$JOB_DIR/retry.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Retry complete. Next: fold recovered into canonical."
    echo "  sbatch legacy/paraphrase_generation/fold_recovered_rewrites.slurm \\"
    echo "      ${DATASET} ${MODEL_SLUG}"
else
    echo "Retry failed with exit code: ${EXIT_CODE}"
fi
echo "=========================================="

SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}_retry.out"
if [ -f "$SLURM_LOG" ]; then
    cp "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

exit $EXIT_CODE

#!/bin/bash
# =============================================================================
# Generate LLM caption rewrites using a local model via vllm (GPU required).
# =============================================================================
# Positional args:
#   $1  DATASET     "coco" or "flickr" (default: coco)
#   $2  MODEL       HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.2)
#
# Uses HuggingFace Transformers (AutoModelForCausalLM, bfloat16, device_map="auto").
#
# Usage:
#   sbatch scripts/generate_rewrites.slurm coco
#   sbatch scripts/generate_rewrites.slurm flickr
#   sbatch scripts/generate_rewrites.slurm coco meta-llama/Meta-Llama-3-8B-Instruct
#
# Smoke test (process only first N captions, e.g. 32):
#   SMOKE_TEST=32 sbatch scripts/generate_rewrites.slurm coco
#
# Override batch size (default: 16):
#   BATCH_SIZE=8 sbatch scripts/generate_rewrites.slurm coco
#
# Resume (safe to resubmit — always appends):
#   sbatch scripts/generate_rewrites.slurm coco
#
# Outputs:
#   datasets/{dataset}/caption_rewrites_{model_slug}.json
#   /output/results/rewrites/{job_id}/generate.log
# =============================================================================
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/.cache/huggingface:/root/.cache/huggingface
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --container-env=HF_TOKEN,HUGGING_FACE_HUB_TOKEN,PYTHONPATH=/workspace

set -euo pipefail

DATASET="${1:-coco}"
MODEL="${2:-mistralai/Mistral-7B-Instruct-v0.2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SMOKE_TEST="${SMOKE_TEST:-0}"

case "$DATASET" in
    coco)
        CAPTIONS_PATH="datasets/coco/caption_datasets/dataset_coco.json"
        OUTPUT_DIR="datasets/coco"
        ;;
    flickr)
        CAPTIONS_PATH="datasets/flickr30k/caption_datasets/dataset_flickr30k.json"
        OUTPUT_DIR="datasets/flickr30k"
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'. Use 'coco' or 'flickr'."
        exit 1
        ;;
esac

# Derive output filename from model name:
#   mistralai/Mistral-7B-Instruct-v0.2 -> caption_rewrites_mistral-7b-instruct-v0.2.json
MODEL_SLUG=$(echo "$MODEL" | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]')
if [ "$SMOKE_TEST" -gt 0 ]; then
    # Smoke test writes to a separate file so it cannot pollute the real run
    OUTPUT_PATH="${OUTPUT_DIR}/caption_rewrites_${MODEL_SLUG}_smoke${SMOKE_TEST}.json"
else
    OUTPUT_PATH="${OUTPUT_DIR}/caption_rewrites_${MODEL_SLUG}.json"
fi

JOB_DIR="/output/results/rewrites/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Dataset:     ${DATASET}"
echo "Model:       ${MODEL}"
echo "Captions:    ${CAPTIONS_PATH}"
echo "Output:      ${OUTPUT_PATH}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Smoke test:  ${SMOKE_TEST} (0 = full run)"
echo "Log:         ${JOB_DIR}/generate.log"
echo "=========================================="

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

echo "Starting rewrite generation..."

# Disable errexit around the python call so the failure banner runs even on
# non-zero exit. tee mirrors output to BOTH the slurm out file and generate.log
# so you can see progress in either place.
set +e
python3 -u scripts/generate_rewrites.py \
    --captions_path "$CAPTIONS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model "$MODEL" \
    --num_rewrites 3 \
    --batch_size "$BATCH_SIZE" \
    --smoke_test "$SMOKE_TEST" \
    --resume 2>&1 | tee "$JOB_DIR/generate.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Rewrite generation completed successfully!"
else
    echo "Rewrite generation failed with exit code: $EXIT_CODE"
fi
echo "Log: $JOB_DIR/generate.log"
echo "=========================================="

SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

exit $EXIT_CODE

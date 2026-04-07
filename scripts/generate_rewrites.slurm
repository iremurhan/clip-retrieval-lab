#!/bin/bash
# =============================================================================
# Generate LLM caption rewrites using a local model via vllm (GPU required).
# =============================================================================
# Positional args:
#   $1  DATASET     "coco" or "flickr" (default: coco)
#   $2  MODEL       HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.2)
#
# Requires Docker image v0.16+ (with vllm and gcc baked in).
#
# Usage:
#   sbatch scripts/generate_rewrites.slurm coco
#   sbatch scripts/generate_rewrites.slurm flickr
#   sbatch scripts/generate_rewrites.slurm coco meta-llama/Meta-Llama-3-8B-Instruct
#
# Resume (safe to resubmit — always appends):
#   sbatch scripts/generate_rewrites.slurm coco
#
# Outputs:
#   datasets/{dataset}/caption_rewrites_{model_slug}.json
#   /output/results/rewrites/{job_id}/generate.log
# =============================================================================
#SBATCH --time=24:00:00
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
OUTPUT_PATH="${OUTPUT_DIR}/caption_rewrites_${MODEL_SLUG}.json"

JOB_DIR="/output/results/rewrites/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Dataset:     ${DATASET}"
echo "Model:       ${MODEL}"
echo "Captions:    ${CAPTIONS_PATH}"
echo "Output:      ${OUTPUT_PATH}"
echo "Log:         ${JOB_DIR}/generate.log"
echo "=========================================="

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

echo "Starting rewrite generation..."

python3 -u scripts/generate_rewrites.py \
    --captions_path "$CAPTIONS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model "$MODEL" \
    --num_rewrites 3 \
    --batch_size 16 \
    --resume \
    > "$JOB_DIR/generate.log" 2>&1

EXIT_CODE=$?

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

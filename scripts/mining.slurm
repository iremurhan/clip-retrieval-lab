#!/bin/bash
#
# Unified Mining Script for Knowledge Distillation
#
# Usage:
#   sbatch --export=MINING_MODALITY=<modality>,MINING_CONFIG=<config> scripts/mining.slurm
#
# Prerequisite on the server (once): mkdir -p ~/experiments/results
#
# Output Structure:
#   experiments/results/{dataset}/{job_id}/
#     - slurm.out         (script/SLURM: container, setup, job messages)
#     - mining.log        (Python/mining output only)
#     - wandb/            (WandB files if enabled)
#
# Mining Output Files (in datasets/{dataset}/pairwise_similarities/):
#   visual    -> mining_image_visual.pt
#   consensus -> mining_image_consensus_{mean,min,max}.pt (3 files, computed in single run)
#   caption   -> mining_text_top{K}_indices.pt, mining_text_top{K}_values.pt
#
# Modalities:
#   visual    - Image-Image via CLIP vision embeddings
#   consensus - Image-Image via caption consensus
#   caption   - Text-Text direct similarity
#
# Config files:
#   configs/config_coco.yaml - COCO dataset (default)
#   configs/config_flickr.yaml - Flickr30k dataset
#
#SBATCH --job-name=mining
#SBATCH --time=07-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=beyza.urhan@bogazici.edu.tr
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
# Mounts: if NOT MOUNTED on cluster, replace ${HOME} with your home path (echo $HOME)
#SBATCH --container-mounts=${HOME}/tez_v2_clean:/workspace,${HOME}/experiments/datasets/coco:/workspace/datasets/coco,${HOME}/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,${HOME}/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,${HOME}/experiments/results:/output/results,${HOME}/experiments/env:/env
#SBATCH --output=${HOME}/experiments/results/%j/slurm.out
#SBATCH --error=${HOME}/experiments/results/%j/slurm.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,MINING_MODALITY,MINING_CONFIG,TOP_K,BATCH_SIZE,QUERY_CHUNK_SIZE,PYTHONPATH=/workspace

set -euo pipefail

# ------------------------------------------------------------------
# CRITICAL CHECK: MOUNT VERIFICATION
# ------------------------------------------------------------------
echo "Checking mounts..."
echo "  /workspace: $(ls /workspace 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /workspace/datasets/coco: $(ls /workspace/datasets/coco 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /workspace/datasets/flickr30k: $(ls /workspace/datasets/flickr30k 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /output: $(ls /output 2>/dev/null | head -3 || echo 'NOT MOUNTED')"

# Check that at least one dataset is mounted (actual dataset determined by config)
if [ ! -d "/workspace/datasets/coco" ] && [ ! -d "/workspace/datasets/flickr30k" ]; then
    echo "FATAL ERROR: No dataset directories found. Check container-mounts."
    echo "Expected at least one of: /workspace/datasets/coco or /workspace/datasets/flickr30k"
    exit 1
fi

# ------------------------------------------------------------------
# ARGUMENTS & DEFAULTS
# ------------------------------------------------------------------
MODALITY="${MINING_MODALITY:-caption}"
CONFIG_FILE="${MINING_CONFIG:-configs/config_coco.yaml}"
TOP_K="${TOP_K:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-100}"

# Validate modality
if [[ ! "$MODALITY" =~ ^(visual|consensus|caption)$ ]]; then
    echo "ERROR: Invalid MINING_MODALITY '$MODALITY'"
    echo "Valid options: visual, consensus, caption"
    exit 1
fi

# Validate config file
if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Determine dataset from config file path
if [[ "$CONFIG_FILE" == *"flickr"* ]]; then
    DATASET="flickr30k"
    WANDB_PROJECT_AUTO="mining-flickr30k"
else
    DATASET="coco"
    WANDB_PROJECT_AUTO="mining-coco"
fi

# ------------------------------------------------------------------
# ENVIRONMENT & DIRECTORIES
# ------------------------------------------------------------------
# Job Directory Structure: results/{dataset}/{job_id}/
JOB_DIR="/output/results/${DATASET}/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR" || echo "Warning: Could not create $JOB_DIR"

# Note: SLURM output/error will be written to default location
# All application logs go to JOB_DIR/ directory

# W&B Setup
export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

# WANDB_PROJECT_AUTO already set above based on dataset

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
    echo "W&B API Key not found. Running offline."
else
    export WANDB_MODE=online
    export WANDB_NAME="${SLURM_JOB_ID}_mining_${MODALITY}"
    export WANDB_PROJECT="$WANDB_PROJECT_AUTO"
    echo "W&B Project auto-set: ${WANDB_PROJECT}"
fi

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

echo "=========================================="
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Dataset:         ${DATASET}"
echo "Config:          ${CONFIG_FILE}"
echo "Modality:        ${MODALITY}"
echo "Top-K:           ${TOP_K}"
echo "Batch Size:      ${BATCH_SIZE}"
echo "Query Chunk:     ${QUERY_CHUNK_SIZE}"
echo "Job Dir:         ${JOB_DIR}"
echo "=========================================="

case $MODALITY in
    visual)
        echo "Mode: Image-Image via Vision Embeddings"
        echo "Output: datasets/${DATASET}/pairwise_similarities/mining_image_visual.pt"
        ;;
    consensus)
        echo "Mode: Image-Image via Caption Consensus"
        echo "Output: datasets/${DATASET}/pairwise_similarities/mining_image_consensus_{mean,min,max}.pt"
        echo "        (All 3 strategies computed in a single run)"
        ;;
    caption)
        echo "Mode: Text-Text Direct Similarity"
        echo "Output: datasets/${DATASET}/pairwise_similarities/mining_text.pt"
        ;;
esac

echo ""
echo "Starting mining process..."

# Run mining (Python output only -> mining.log; script output -> slurm.out)
python3 -u tools/mine_pairwise_sim.py \
    --modality "$MODALITY" \
    --config "$CONFIG_FILE" \
    --top_k "$TOP_K" \
    --batch_size "$BATCH_SIZE" \
    --query_chunk_size "$QUERY_CHUNK_SIZE" \
    > "$JOB_DIR/mining.log" 2>&1

MINING_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $MINING_EXIT_CODE -eq 0 ]; then
    echo "Mining completed successfully!"
else
    echo "Mining failed with exit code: $MINING_EXIT_CODE"
fi
echo "Results saved to: $JOB_DIR"
echo "=========================================="

exit $MINING_EXIT_CODE

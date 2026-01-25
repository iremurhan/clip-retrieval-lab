#!/bin/bash
#
# Unified Mining Script for Knowledge Distillation
#
# Usage:
#   sbatch --export=MINING_MODALITY=<modality>,MINING_CONFIG=<config> scripts/mining.slurm
#   Or via wrapper: ./scripts/run_mining.sh <modality> [config_file]
#
# Modalities:
#   visual    - Image-Image via CLIP vision embeddings
#   consensus - Image-Image via caption consensus
#   caption   - Text-Text direct similarity
#
# Config files:
#   config.yaml       - COCO dataset (default)
#   config_flickr.yaml - Flickr30k dataset
#
# Output files (in datasets/{dataset}/pairwise_similarities/):
#   visual    -> mining_image_visual.pt
#   consensus -> mining_image_consensus.pt
#   caption   -> mining_text.pt
#
#SBATCH --job-name=mining
#SBATCH --output=/users/beyza.urhan/experiments/runs/%j/mining.log
#SBATCH --error=/users/beyza.urhan/experiments/runs/%j/err.log
#SBATCH --time=07-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=beyza.urhan@bogazici.edu.tr
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/runs:/output,/users/beyza.urhan/experiments/env:/env
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,MINING_MODALITY,MINING_CONFIG,TOP_K,BATCH_SIZE,QUERY_CHUNK_SIZE

set -euo pipefail

# ------------------------------------------------------------------
# CRITICAL CHECK: MOUNT VERIFICATION
# ------------------------------------------------------------------
echo "Checking mounts..."
echo "  /workspace: $(ls /workspace 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /workspace/datasets/coco: $(ls /workspace/datasets/coco 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /output: $(ls /output 2>/dev/null | head -3 || echo 'NOT MOUNTED')"

if [ ! -d "/workspace/datasets/coco" ]; then
    echo "FATAL ERROR: '/workspace/datasets/coco' not mounted. Check container-mounts."
    exit 1
fi

# ------------------------------------------------------------------
# ARGUMENTS & DEFAULTS
# ------------------------------------------------------------------
MODALITY="${MINING_MODALITY:-caption}"
CONFIG_FILE="${MINING_CONFIG:-config.yaml}"
TOP_K="${TOP_K:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-100}"

# Validate modality
if [[ ! "$MODALITY" =~ ^(visual|consensus|caption)$ ]]; then
    echo "ERROR: Invalid MINING_MODALITY '$MODALITY'"
    echo "Valid options: visual, consensus, caption"
    exit 1
fi

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ------------------------------------------------------------------
# ENVIRONMENT & DIRECTORIES
# ------------------------------------------------------------------
# /output is mounted from /users/beyza.urhan/experiments/runs
JOB_DIR="/output/${SLURM_JOB_ID}_mining"
mkdir -p "$JOB_DIR" || echo "Warning: Could not create $JOB_DIR"

# W&B Setup
export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
    echo "W&B API Key not found. Running offline."
else
    export WANDB_MODE=online
    export WANDB_NAME="${SLURM_JOB_ID}_mining_${MODALITY}"
fi

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
cd /workspace

echo "=========================================="
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Config:          ${CONFIG_FILE}"
echo "Modality:        ${MODALITY}"
echo "Top-K:           ${TOP_K}"
echo "Batch Size:      ${BATCH_SIZE}"
echo "Query Chunk:     ${QUERY_CHUNK_SIZE}"
echo "Job Dir:         ${JOB_DIR}"
echo "=========================================="

# Extract dataset name from config file for display
if [[ "$CONFIG_FILE" == *"flickr"* ]]; then
    DATASET_NAME="flickr30k"
else
    DATASET_NAME="coco"
fi

case $MODALITY in
    visual)
        echo "Mode: Image-Image via Vision Embeddings"
        echo "Output: datasets/${DATASET_NAME}/pairwise_similarities/mining_image_visual.pt"
        ;;
    consensus)
        echo "Mode: Image-Image via Caption Consensus"
        echo "Output: datasets/${DATASET_NAME}/pairwise_similarities/mining_image_consensus.pt"
        ;;
    caption)
        echo "Mode: Text-Text Direct Similarity"
        echo "Output: datasets/${DATASET_NAME}/pairwise_similarities/mining_text.pt"
        ;;
esac

echo ""
echo "Starting mining process..."

# Run unified mining script
python3 -u tools/mine_pairwise_sim.py \
    --modality "$MODALITY" \
    --config "$CONFIG_FILE" \
    --top_k "$TOP_K" \
    --batch_size "$BATCH_SIZE" \
    --query_chunk_size "$QUERY_CHUNK_SIZE"

echo ""
echo "=========================================="
echo "Mining complete!"
echo "=========================================="

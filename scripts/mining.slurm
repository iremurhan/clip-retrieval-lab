#!/bin/bash
# Mining. Prefer: ./scripts/submit_mining.sh <modality> [config]
# Direct: sbatch --export=MINING_MODALITY=...,MINING_CONFIG=... scripts/mining.slurm
# Logs: results/{coco|flickr30k}/{job_id}/ (slurm.out, mining.log). W&B: mining.
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/{DATASET}/%j/slurm.out
#SBATCH --error=/users/beyza.urhan/experiments/results/{DATASET}/%j/slurm.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace,MINING_MODALITY,MINING_CONFIG,TOP_K,BATCH_SIZE,QUERY_CHUNK_SIZE

set -euo pipefail

echo "Checking mounts..."
echo "  /workspace: $(ls /workspace 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /output: $(ls /output 2>/dev/null | head -3 || echo 'NOT MOUNTED')"

if [ ! -d "/workspace" ] || [ ! -d "/output" ]; then
    echo "FATAL ERROR: Required mounts not found. Check container-mounts."
    exit 1
fi

MODALITY="${MINING_MODALITY:-caption}"
CONFIG_FILE="${MINING_CONFIG:-configs/config_coco.yaml}"
TOP_K="${TOP_K:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-100}"

if [[ ! "$MODALITY" =~ ^(visual|consensus|caption)$ ]]; then
    echo "ERROR: Invalid MINING_MODALITY '$MODALITY'. Valid: visual, consensus, caption"
    exit 1
fi

if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [[ "$CONFIG_FILE" == *"flickr"* ]]; then
    DATASET="flickr30k"
else
    DATASET="coco"
fi
WANDB_PROJECT_AUTO="mining"

JOB_DIR="/output/results/${DATASET}/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Modality:      ${MODALITY}"
echo "Dataset:       ${DATASET}"
echo "Job dir:       ${JOB_DIR} (slurm.out, mining.log, wandb/)"
echo "=========================================="

export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
    export WANDB_NAME="${SLURM_JOB_ID}_mining_${MODALITY}"
    export WANDB_PROJECT="$WANDB_PROJECT_AUTO"
    echo "W&B Project auto-set: $WANDB_PROJECT"
else
    echo "W&B API Key not found. Switching to OFFLINE mode."
    export WANDB_MODE=offline
fi

echo "Starting mining..."
cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

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

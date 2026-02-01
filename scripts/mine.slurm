#!/bin/bash
# To run COCO: sbatch --export=TARGET_DATASET=coco scripts/mine.slurm or sbatch scripts/mine.slurm
# To run Flickr: sbatch --export=TARGET_DATASET=flickr scripts/mine.slurm or sbatch --export=ALL,TARGET_DATASET=flickr scripts/mine.slurm
#SBATCH --job-name=mining
#SBATCH --output=/users/beyza.urhan/experiments/results/{DATASET}/%j/mining_log.out
#SBATCH --error=/users/beyza.urhan/experiments/results/{DATASET}/%j/mining_log.out
#SBATCH --time=07:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4

# Mount settings
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env

# Pass WandB variables to container
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,TARGET_DATASET

set -euo pipefail

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
# Default to 'coco' if TARGET_DATASET is not set
DATASET="${TARGET_DATASET:-coco}"

if [ "$DATASET" == "flickr" ]; then
    CONFIG_FILE="configs/config_flickr.yaml"
    echo ">>> MODE: FLICKR30K"
elif [ "$DATASET" == "coco" ]; then
    CONFIG_FILE="configs/config_coco.yaml"
    echo ">>> MODE: MS-COCO"
else
    echo "ERROR: Unknown dataset '$DATASET'. Use 'coco' or 'flickr'."
    exit 1
fi

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Date:          $(date)"
echo "Target dataset: ${TARGET_DATASET}"
echo "Results directory: /users/beyza.urhan/experiments/results/${TARGET_DATASET}/${SLURM_JOB_ID}"
echo "=========================================="

# ------------------------------------------------------------------
# WANDB SETUP
# ------------------------------------------------------------------
# Load wandb.env from /env if present
if [ -f /env/wandb.env ]; then
    echo "Loading WandB environment file..."
    set -a
    source /env/wandb.env
    set +a
else
    echo "WARNING: /env/wandb.env not found. API key may be missing."
fi

# Set project name here (overrides file). Set your project name in quotes:
export WANDB_PROJECT="mining"

# Set run name dynamically (include Job ID)
export WANDB_NAME="mining_${DATASET}_top1000_${SLURM_JOB_ID}"

echo "Active WandB project: $WANDB_PROJECT"

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
cd /workspace

echo "Starting mining... for $DATASET"

python3 -u tools/mine_pairwise_sim.py \
    --modality caption \
    --config "$CONFIG_FILE" \
    --top_k 1000 \
    --batch_size 512 \
    --query_chunk_size 5000

echo "Mining completed... for $DATASET"

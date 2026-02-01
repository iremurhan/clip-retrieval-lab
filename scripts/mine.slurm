#!/bin/bash
#SBATCH --job-name=mining_coco
#SBATCH --output=mining_log_%j.out
#SBATCH --error=mining_log_%j.out
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=beyza.urhan@bogazici.edu.tr

# Mount settings
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env

# Pass WandB variables to container
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT

set -euo pipefail

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Date:          $(date)"
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
export WANDB_NAME="mining_top1000_${SLURM_JOB_ID}"

echo "Active WandB project: $WANDB_PROJECT"

# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------
cd /workspace

echo "Starting mining..."

python3 -u tools/mine_pairwise_sim.py \
    --modality caption \
    --config configs/config_coco.yaml \
    --top_k 1000 \
    --batch_size 512 \
    --query_chunk_size 5000

echo "Mining completed."

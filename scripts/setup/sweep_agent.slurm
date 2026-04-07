#!/bin/bash
#SBATCH --job-name=sweep_worker
#SBATCH --output=/users/beyza.urhan/experiments/results/sweep_logs/%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/sweep_logs/%j.out
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

# ------------------------------------------------------------------
# W&B Sweep Agent (runs a single worker that consumes runs from a sweep)
# Usage:
#   sbatch scripts/sweep_agent.slurm <SWEEP_ID> [PROJECT_NAME] [COUNT]
#     SWEEP_ID      – ID returned by `wandb sweep ...`
#     PROJECT_NAME  – W&B project name (default: retrieval)
#     COUNT         – number of runs this agent should execute (default: 10)
#
# Example:
#   sbatch scripts/sweep_agent.slurm abc123def my_project 20
# ------------------------------------------------------------------

SWEEP_ID=$1
PROJECT_NAME=${2:-retrieval}
COUNT=${3:-10}

mkdir -p /users/beyza.urhan/experiments/results/sweep_logs

# Environment Setup
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi
export WANDB_PROJECT="$PROJECT_NAME"

cd /workspace

echo "========================================================"
echo "Starting Sweep Agent"
echo "Job ID: $SLURM_JOB_ID"
echo "Sweep ID: $SWEEP_ID"
echo "Project: $WANDB_PROJECT"
echo "========================================================"

wandb agent --count $COUNT $SWEEP_ID
#!/bin/bash
# =============================================================================
# WANDB SYNC SCRIPT
# =============================================================================
# Syncs a crashed/offline W&B run directory to the server using the same
# Pyxis container setup as the training jobs.
#
# Positional args:
#   $1  JOB_DIR   Absolute path on the host containing the `wandb/` folder
#                 (e.g. /users/beyza.urhan/experiments/results/coco/24918)
#
# Example usage:
#   sbatch scripts/wandb_sync.slurm /users/beyza.urhan/experiments/results/coco/24918
# =============================================================================
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.14
#SBATCH --container-mounts=/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/wandb_sync_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/wandb_sync_%j.out
#SBATCH --container-env=WANDB_API_KEY

set -euo pipefail

HOST_JOB_DIR="${1:?Usage: sbatch scripts/wandb_sync.slurm <host_job_dir>}"

# Map host path -> container path under /output/results
HOST_PREFIX="/users/beyza.urhan/experiments/results/"
CONTAINER_JOB_DIR="/output/results/${HOST_JOB_DIR#$HOST_PREFIX}"

if [ ! -d "$CONTAINER_JOB_DIR/wandb" ]; then
    echo "ERROR: $CONTAINER_JOB_DIR/wandb not found inside container."
    exit 1
fi

# Load WANDB_API_KEY from the mounted env file.
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY not set."
    exit 1
fi

export WANDB_MODE=online

echo "Syncing $CONTAINER_JOB_DIR/wandb ..."
python3 -m wandb sync --sync-all "$CONTAINER_JOB_DIR/wandb"
SYNC_EXIT=$?

echo "Sync exit code: $SYNC_EXIT"
exit $SYNC_EXIT

#!/bin/bash
# =============================================================================
# Export W&B run summaries from the project container.
# =============================================================================
# Usage:
#   sbatch scripts/util/export_wandb.slurm [out_dir]
# =============================================================================
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env,/tmp:/tmp
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j_wandb_export.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j_wandb_export.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

OUT_DIR="${1:-/output/results/wandb_export/${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR"

if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY is not available."
    exit 1
fi

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export WANDB_MODE=online

python3 -u scripts/util/export_wandb.py --out-dir "$OUT_DIR"
echo "Export directory: $OUT_DIR"

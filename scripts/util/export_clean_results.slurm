#!/bin/bash
# =============================================================================
# Export clean W&B result tables and per-run evaluation manifest.
# =============================================================================
# Usage:
#   sbatch scripts/util/export_clean_results.slurm [out_dir]
# =============================================================================
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env,/tmp:/tmp
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j_clean_results.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j_clean_results.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace,CLIP_RETRIEVAL_ARTIFACT_ROOT=/output/results/artifacts/clip-retrieval-lab

set -euo pipefail

OUT_DIR="${1:-/output/results/artifacts/clip-retrieval-lab/clean}"
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

python3 -u scripts/util/export_clean_results.py \
    --out-dir "$OUT_DIR" \
    --results-root /output/results

echo "Clean result export directory: $OUT_DIR"

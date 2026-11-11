#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/datasets/sugarcrepe:/workspace/datasets/sugarcrepe,/users/beyza.urhan/experiments/datasets/mmvp_vlm:/workspace/datasets/mmvp_vlm,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/paper_smoke_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/paper_smoke_%j.out
#SBATCH --container-env=WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

OUT_DIR="/output/results/paper/baseline-calibration/smoke/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export WANDB_MODE=disabled

python3 -u scripts/paper/smoke_clip_b_calibration.py 2>&1 | tee "$OUT_DIR/smoke.log"
cp "/users/beyza.urhan/experiments/results/paper_smoke_${SLURM_JOB_ID}.out" "$OUT_DIR/slurm.out" 2>/dev/null || true

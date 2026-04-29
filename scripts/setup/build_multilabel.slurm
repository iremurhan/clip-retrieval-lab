#!/bin/bash
# =============================================================================
# Build COCO multi-label map (one-time preprocessing)
#
# Usage:
#   sbatch scripts/build_multilabel.slurm
# =============================================================================
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --container-image=biremurhan/image-text-contrast:b5
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j_multilabel.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j_multilabel.out

set -euo pipefail
cd /workspace

python3 tools/build_coco_multilabel.py \
    --ann_dir /workspace/datasets/coco/annotations \
    --output /workspace/datasets/coco/coco_multilabel_map.pt

echo "Done. Output: datasets/coco/coco_multilabel_map.pt"

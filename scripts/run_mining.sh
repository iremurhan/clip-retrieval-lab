#!/bin/bash
#
# Run Mining Jobs on SLURM
#
# Usage:
#   ./scripts/run_mining.sh <modality>
#
# Modalities:
#   visual    - Image-Image similarity via CLIP vision embeddings
#   consensus - Image-Image similarity via caption consensus (5x5 matrix mean)
#   caption   - Text-Text direct cosine similarity
#
# Output files (in datasets/coco/pairwise_similarities/):
#   visual    -> mining_image_visual.pt    (ID-based mapping)
#   consensus -> mining_image_consensus.pt (ID-based mapping)
#   caption   -> mining_text.pt            (index-based mapping)
#

set -e

MODALITY="${1:-}"

if [ -z "$MODALITY" ]; then
    echo "============================================"
    echo "Unified Mining Script"
    echo "============================================"
    echo ""
    echo "Usage: ./scripts/run_mining.sh <modality>"
    echo ""
    echo "Available modalities:"
    echo "  visual    - Image-Image via vision features"
    echo "  consensus - Image-Image via caption agreement"
    echo "  caption   - Text-Text direct similarity"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_mining.sh visual"
    echo "  ./scripts/run_mining.sh consensus"
    echo "  ./scripts/run_mining.sh caption"
    echo ""
    exit 1
fi

# Validate modality
if [[ ! "$MODALITY" =~ ^(visual|consensus|caption)$ ]]; then
    echo "ERROR: Invalid modality '$MODALITY'"
    echo "Valid options: visual, consensus, caption"
    exit 1
fi

echo "============================================"
echo "Mining: $MODALITY"
echo "============================================"

# Export modality for SLURM script
export MINING_MODALITY="$MODALITY"

echo "Submitting job for modality: $MODALITY"
sbatch --export=ALL,MINING_MODALITY="$MODALITY" scripts/mining.slurm

echo ""
echo "Monitor with: squeue -u \$USER"
echo "============================================"

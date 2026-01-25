#!/bin/bash
#
# Run Mining Jobs on SLURM
#
# Usage:
#   ./scripts/run_mining.sh <modality> [config_file]
#
# Modalities:
#   visual    - Image-Image similarity via CLIP vision embeddings
#   consensus - Image-Image similarity via caption consensus (5x5 matrix mean)
#   caption   - Text-Text direct cosine similarity
#
# Config files:
#   config.yaml       - COCO dataset (default)
#   config_flickr.yaml - Flickr30k dataset
#
# Output files (in datasets/{dataset}/pairwise_similarities/):
#   visual    -> mining_image_visual.pt    (ID-based mapping)
#   consensus -> mining_image_consensus.pt (ID-based mapping)
#   caption   -> mining_text.pt            (index-based mapping)
#

set -e

MODALITY="${1:-}"
CONFIG_FILE="${2:-config.yaml}"

if [ -z "$MODALITY" ]; then
    echo "============================================"
    echo "Unified Mining Script"
    echo "============================================"
    echo ""
    echo "Usage: ./scripts/run_mining.sh <modality> [config_file]"
    echo ""
    echo "Available modalities:"
    echo "  visual    - Image-Image via vision features"
    echo "  consensus - Image-Image via caption agreement"
    echo "  caption   - Text-Text direct similarity"
    echo ""
    echo "Config files:"
    echo "  config.yaml       - COCO dataset (default)"
    echo "  config_flickr.yaml - Flickr30k dataset"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_mining.sh visual"
    echo "  ./scripts/run_mining.sh consensus config.yaml"
    echo "  ./scripts/run_mining.sh caption config_flickr.yaml"
    echo ""
    exit 1
fi

# Validate modality
if [[ ! "$MODALITY" =~ ^(visual|consensus|caption)$ ]]; then
    echo "ERROR: Invalid modality '$MODALITY'"
    echo "Valid options: visual, consensus, caption"
    exit 1
fi

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "============================================"
echo "Mining: $MODALITY"
echo "Config: $CONFIG_FILE"
echo "============================================"

# Export variables for SLURM script
export MINING_MODALITY="$MODALITY"
export MINING_CONFIG="$CONFIG_FILE"

echo "Submitting job for modality: $MODALITY, config: $CONFIG_FILE"
sbatch --export=ALL,MINING_MODALITY="$MODALITY",MINING_CONFIG="$CONFIG_FILE" scripts/mining.slurm

echo ""
echo "Monitor with: squeue -u \$USER"
echo "============================================"

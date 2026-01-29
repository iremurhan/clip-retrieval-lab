#!/bin/bash
# Submit mining job with $HOME expanded for Pyxis (sbatch does not expand variables in #SBATCH).
# Usage: ./scripts/submit_mining.sh <modality> [config]
# Example: ./scripts/submit_mining.sh caption configs/config_flickr.yaml

set -e

MODALITY="${1:-caption}"
CONFIG="${2:-configs/config_coco.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME="${HOME:-$HOME}"

TMP_SCRIPT=$(mktemp)
sed "s|\${HOME}|$HOME|g" "$SCRIPT_DIR/mining.slurm" > "$TMP_SCRIPT"

sbatch --export=MINING_MODALITY="$MODALITY",MINING_CONFIG="$CONFIG" "$TMP_SCRIPT"
rm -f "$TMP_SCRIPT"

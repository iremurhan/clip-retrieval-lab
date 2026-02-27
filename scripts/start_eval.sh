#!/bin/bash
# =============================================================================
# EVALUATION LAUNCHER — Wrapper for scripts/eval.slurm
# =============================================================================
# Usage:
#   ./scripts/start_eval.sh <config_path> <checkpoint_path> <split> <mode>
#
# Arguments:
#   config_path      Config file relative to workspace (e.g. configs/config_coco.yaml)
#   checkpoint_path  Absolute path to .pth checkpoint, or "none" for zero-shot
#   split            val | test  (default: test)
#   mode             normal | eccv  (default: normal)
#
# Examples:
#   ./scripts/start_eval.sh configs/config_coco.yaml /output/results/coco/12345/best_model.pth test normal
#   ./scripts/start_eval.sh configs/config_coco.yaml /output/results/coco/12345/best_model.pth test eccv
#   ./scripts/start_eval.sh configs/config_flickr.yaml /output/results/flickr30k/67890/best_model.pth test normal
#   ./scripts/start_eval.sh configs/config_coco.yaml none test normal   # zero-shot
# =============================================================================

set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }
warn()  { echo -e "${YELLOW}WARN:${NC} $*" >&2; }
info()  { echo -e "${GREEN}INFO:${NC} $*"; }

# =============================================================================
# Argument parsing with defaults
# =============================================================================
CONFIG_PATH="${1:-}"
CHECKPOINT_PATH="${2:-none}"
SPLIT="${3:-test}"
EVAL_MODE="${4:-normal}"

# =============================================================================
# Input validation
# =============================================================================

# 1. Config path is required
if [ -z "$CONFIG_PATH" ]; then
    error "Config path is required.\n  Usage: $0 <config_path> <checkpoint_path> <split> <mode>"
fi

# 2. Config file must exist locally
if [ ! -f "$CONFIG_PATH" ]; then
    error "Config file not found: $CONFIG_PATH"
fi

# 3. Split must be val or test
if [[ "$SPLIT" != "val" && "$SPLIT" != "test" ]]; then
    error "Split must be 'val' or 'test'. Got: $SPLIT"
fi

# 4. Mode must be normal or eccv
if [[ "$EVAL_MODE" != "normal" && "$EVAL_MODE" != "eccv" ]]; then
    error "Mode must be 'normal' or 'eccv'. Got: $EVAL_MODE"
fi

# 5. ECCV mode requires COCO config
if [[ "$EVAL_MODE" == "eccv" ]]; then
    if [[ "$CONFIG_PATH" == *"flickr"* ]]; then
        error "ECCV mode requires a COCO config. Flickr30k is not supported by the ECCV Caption protocol.\n  Got: $CONFIG_PATH"
    fi
    info "ECCV mode: will use COCO Karpathy test split with eccv_caption protocol."
fi

# 6. Checkpoint validation
if [[ "$CHECKPOINT_PATH" == "none" ]]; then
    CHECKPOINT_PATH=""
    info "No checkpoint — zero-shot evaluation."
else
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        error "Checkpoint file not found: $CHECKPOINT_PATH\n  Provide an absolute path to a .pth file on the HPC filesystem, or 'none' for zero-shot."
    fi
    info "Checkpoint: $CHECKPOINT_PATH"
fi

# =============================================================================
# Derive job name
# =============================================================================
if [[ "$CONFIG_PATH" == *"flickr"* ]]; then
    DATASET="flickr30k"
else
    DATASET="coco"
fi

JOB_NAME="${DATASET}_${EVAL_MODE}_eval"

# =============================================================================
# Submit
# =============================================================================
echo ""
echo "=========================================="
echo "  Submitting Evaluation Job"
echo "=========================================="
echo "  Config:       $CONFIG_PATH"
echo "  Checkpoint:   ${CHECKPOINT_PATH:-(zero-shot)}"
echo "  Split:        $SPLIT"
echo "  Mode:         $EVAL_MODE"
echo "  Job name:     $JOB_NAME"
echo "=========================================="
echo ""

sbatch \
    --job-name="$JOB_NAME" \
    scripts/eval.slurm \
    "$CONFIG_PATH" \
    "$CHECKPOINT_PATH" \
    "$SPLIT" \
    "$EVAL_MODE"

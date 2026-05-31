#!/bin/bash
# =============================================================================
# Paired baseline-vs-intervention retrieval comparison (fixed / broken cases).
# =============================================================================
# Positional args:
#   $1  BASELINE_CKPT       Path to the baseline best_model.pth
#   $2  INTERVENTION_CKPT   Path to the intervention best_model.pth
#   $3  BASELINE_LABEL      Display label for the baseline (e.g. Base-min)
#   $4  INTERVENTION_LABEL  Display label for the intervention (e.g. HN-Syntactic)
#   $5  CONFIG_PATH         Dataset config (default: configs/config_coco.yaml)
#
# Optional env:
#   TOP_K                   Retrieved items shown per model (default: 10)
#   MAX_CASES               Max cases per direction/kind (default: 40)
#
# Usage:
#   sbatch scripts/eval/compare_retrieval_pairs.slurm \
#       /output/results/coco/B0_coco_s42/best_model.pth \
#       /output/results/coco/B2_coco_s123/best_model.pth \
#       Base-min HN-Syntactic \
#       configs/config_coco.yaml
#
# Outputs:
#   /output/results/retrieval_pairs/{job_id}/flips.html
#   /output/results/retrieval_pairs/{job_id}/flips.json
#   /output/results/retrieval_pairs/{job_id}/eval.log
# =============================================================================
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.16
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail

echo "Checking mounts..."
echo "  /workspace: $(ls /workspace 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /output:    $(ls /output 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
if [ ! -d "/workspace" ] || [ ! -d "/output" ]; then
    echo "FATAL ERROR: Required mounts not found. Check container-mounts."
    exit 1
fi

BASELINE_CKPT="${1:?ERROR: baseline checkpoint required as \$1}"
INTERVENTION_CKPT="${2:?ERROR: intervention checkpoint required as \$2}"
BASELINE_LABEL="${3:-Baseline}"
INTERVENTION_LABEL="${4:-Intervention}"
CONFIG_FILE="${5:-configs/config_coco.yaml}"
TOP_K="${TOP_K:-10}"
MAX_CASES="${MAX_CASES:-40}"

for ckpt in "$BASELINE_CKPT" "$INTERVENTION_CKPT"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: Checkpoint not found: $ckpt"; exit 1
    fi
done
if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: /workspace/$CONFIG_FILE"; exit 1
fi

JOB_DIR="/output/results/retrieval_pairs/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Baseline:      ${BASELINE_LABEL}  (${BASELINE_CKPT})"
echo "Intervention:  ${INTERVENTION_LABEL}  (${INTERVENTION_CKPT})"
echo "Config:        ${CONFIG_FILE}"
echo "Output dir:    ${JOB_DIR}"
echo "Top-k:         ${TOP_K}"
echo "Max cases:     ${MAX_CASES}"
echo "=========================================="

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

# Datasets are mounted at the config's default paths in-container, so no --data-root.
python3 -u scripts/eval/compare_retrieval_pairs.py \
    --baseline-ckpt "$BASELINE_CKPT" \
    --intervention-ckpt "$INTERVENTION_CKPT" \
    --baseline-label "$BASELINE_LABEL" \
    --intervention-label "$INTERVENTION_LABEL" \
    --config "$CONFIG_FILE" \
    --output-dir "$JOB_DIR" \
    --top-k "$TOP_K" --max-cases "$MAX_CASES" \
    > "$JOB_DIR/eval.log" 2>&1

EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Paired comparison complete."
    echo "HTML report: $JOB_DIR/flips.html"
    echo "JSON data:   $JOB_DIR/flips.json"
else
    echo "Paired comparison failed with exit code: $EXIT_CODE"
    tail -30 "$JOB_DIR/eval.log" || true
fi
echo "=========================================="

SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

exit $EXIT_CODE

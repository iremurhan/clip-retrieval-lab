#!/bin/bash
#SBATCH --job-name=mining
#SBATCH --partition=gpu
#SBATCH --time=07-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/coco/%j/slurm.out
#SBATCH --error=/users/beyza.urhan/experiments/results/coco/%j/slurm.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,MINING_MODALITY,MINING_CONFIG,TOP_K,BATCH_SIZE,QUERY_CHUNK_SIZE,PYTHONPATH=/workspace

set -euo pipefail

echo "=========================================="
echo "MOUNT CHECK (container required; host execution not allowed)"
echo "=========================================="
WS_OK=0
COCO_OK=0
FLICKR_OK=0
OUT_OK=0
[ -d "/workspace" ] && WS_OK=1
[ -d "/workspace/datasets/coco" ] && COCO_OK=1
[ -d "/workspace/datasets/flickr30k" ] && FLICKR_OK=1
[ -d "/output" ] && [ -d "/output/results" ] && OUT_OK=1

echo "  /workspace                    -> $( [ $WS_OK -eq 1 ] && echo "OK ($(ls /workspace 2>/dev/null | wc -l) items)" || echo "MISSING" )"
echo "  /workspace/datasets/coco      -> $( [ $COCO_OK -eq 1 ] && echo "OK" || echo "MISSING" )"
echo "  /workspace/datasets/flickr30k -> $( [ $FLICKR_OK -eq 1 ] && echo "OK" || echo "MISSING" )"
echo "  /output/results               -> $( [ $OUT_OK -eq 1 ] && echo "OK" || echo "MISSING" )"

if [ $WS_OK -eq 0 ] || [ $COCO_OK -eq 0 ] && [ $FLICKR_OK -eq 0 ]; then
    IN_CONTAINER=0
    [ -f /.dockerenv ] || [ -f /run/.containerenv ] && IN_CONTAINER=1
    HN=$(hostname)
    echo ""
    echo "--- DIAGNOSIS ---"
    echo "  SLURM_JOB_ID:    ${SLURM_JOB_ID:- (not set)}"
    echo "  PWD:             $(pwd)"
    echo "  Hostname:        $HN"
    echo "  In container:    $([ $IN_CONTAINER -eq 1 ] && echo 'yes' || echo 'NO (running on host)')"
    echo ""
    if [ "$HN" = "ultramarine" ] || [ "$HN" = "ultramarine.ulakbim.gov.tr" ]; then
        echo "  NOTE: Hostname is the login/submit node. Container jobs usually run on COMPUTE nodes."
        echo "        If the job ran on the login node, containers are often disabled there."
        echo "        Check: squeue -u \$USER (job NODE); scontrol show job $SLURM_JOB_ID (NodeList)."
        echo ""
    fi
    echo "FATAL: This script MUST run inside a container. Running on the host is not allowed."
    echo "       /workspace and /output are not visible -> container was not started."
    echo "       #SBATCH --container-image and --container-mounts were ignored."
    echo ""
    echo "NEXT STEPS:"
    echo "  1. Ask HPC support: which partition/account enables container jobs (Pyxis/Enroot)."
    echo "  2. Submit with a container-enabled partition: sbatch --partition=<...> ..."
    echo "  3. Ensure job runs on a compute node, not the login node (ultramarine)."
    echo "  4. Check: scontrol show partition; cluster docs for 'container', 'Pyxis', 'Enroot'."
    exit 1
fi
echo "=========================================="

MODALITY="${MINING_MODALITY:-caption}"
CONFIG_FILE="${MINING_CONFIG:-configs/config_coco.yaml}"
TOP_K="${TOP_K:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-100}"

if [[ ! "$MODALITY" =~ ^(visual|consensus|caption)$ ]]; then
    echo "ERROR: Invalid MINING_MODALITY '$MODALITY'"
    echo "Valid options: visual, consensus, caption"
    exit 1
fi

if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [[ "$CONFIG_FILE" == *"flickr"* ]]; then
    DATASET="flickr30k"
else
    DATASET="coco"
fi
WANDB_PROJECT_AUTO="mining"

JOB_DIR="/output/results/${DATASET}/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR" || echo "Warning: Could not create $JOB_DIR"

echo "Job dir:      $JOB_DIR (slurm.out, mining.log, wandb/)"

export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
    echo "W&B API Key not found. Running offline."
else
    export WANDB_MODE=online
    export WANDB_NAME="${SLURM_JOB_ID}_mining_${MODALITY}"
    export WANDB_PROJECT="$WANDB_PROJECT_AUTO"
fi

cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

echo "=========================================="
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Dataset:         ${DATASET}"
echo "Config:          ${CONFIG_FILE}"
echo "Modality:        ${MODALITY}"
echo "Top-K:           ${TOP_K}"
echo "Batch Size:      ${BATCH_SIZE}"
echo "Query Chunk:     ${QUERY_CHUNK_SIZE}"
echo "Job Dir:         ${JOB_DIR}"
echo "=========================================="

case $MODALITY in
    visual)
        echo "Mode: Image-Image via Vision Embeddings"
        echo "Output: datasets/${DATASET}/pairwise_similarities/mining_image_visual.pt"
        ;;
    consensus)
        echo "Mode: Image-Image via Caption Consensus"
        echo "Output: datasets/${DATASET}/pairwise_similarities/mining_image_consensus_{mean,min,max}.pt"
        echo "        (All 3 strategies computed in a single run)"
        ;;
    caption)
        echo "Mode: Text-Text Direct Similarity"
        echo "Output: datasets/${DATASET}/pairwise_similarities/mining_text.pt"
        ;;
esac

echo ""
echo "Starting mining process..."

python3 -u tools/mine_pairwise_sim.py \
    --modality "$MODALITY" \
    --config "$CONFIG_FILE" \
    --top_k "$TOP_K" \
    --batch_size "$BATCH_SIZE" \
    --query_chunk_size "$QUERY_CHUNK_SIZE" \
    > "$JOB_DIR/mining.log" 2>&1

MINING_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $MINING_EXIT_CODE -eq 0 ]; then
    echo "Mining completed successfully!"
else
    echo "Mining failed with exit code: $MINING_EXIT_CODE"
fi
echo "Job dir: $JOB_DIR"
echo "=========================================="

exit $MINING_EXIT_CODE

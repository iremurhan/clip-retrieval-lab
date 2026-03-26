#!/bin/bash
# Launch a W&B hyperparameter sweep and submit multiple Slurm agents.
# Dataset-specific sweep configs:
#   - configs/sweep_flickr.yaml  (uses configs/config_flickr30k.yaml)
#   - configs/sweep_coco.yaml    (uses configs/config_coco.yaml)
#
# Usage:
#   bash scripts/launch_sweep.sh <DATASET> [NUM_AGENTS]
#     DATASET     – flickr or coco
#     NUM_AGENTS  – number of parallel sweep workers (default: 3)
#
# Examples:
#   # Flickr30k sweep with 3 workers
#   bash scripts/launch_sweep.sh flickr
#
#   # COCO sweep with 5 workers
#   bash scripts/launch_sweep.sh coco 5

DATASET=${1:-flickr}
NUM_AGENTS=${2:-3} # Number of parallel sweep workers (default: 3)

if [ "$DATASET" = "flickr" ] || [ "$DATASET" = "flickr30k" ]; then
    CONFIG_PATH="configs/sweep_flickr.yaml"
elif [ "$DATASET" = "coco" ]; then
    CONFIG_PATH="configs/sweep_coco.yaml"
else
    echo "ERROR: Unknown dataset '$DATASET'. Use 'flickr' or 'coco'."
    exit 1
fi
PROJECT_NAME="retrieval"

# Load WANDB_API_KEY from wandb.env file if it exists
if [ -f ~/experiments/env/wandb.env ]; then
    source ~/experiments/env/wandb.env
    export WANDB_API_KEY
fi

# Verify WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set. Please run:"
    echo "  source ~/experiments/env/wandb.env"
    echo "  export WANDB_API_KEY"
    exit 1
fi

# 1. Launch the W&B sweep inside the container and capture the SWEEP_ID
echo ">>> Launching W&B sweep inside container and capturing SWEEP_ID..."
SWEEP_OUTPUT=$(srun --container-image=biremurhan/image-text-contrast:v0.8 \
     --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/env:/env \
     --container-env=WANDB_API_KEY \
     bash -c "source /env/wandb.env 2>/dev/null || true && cd /workspace && wandb sweep $CONFIG_PATH 2>&1")

SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep 'wandb agent' | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to obtain SWEEP_ID. Check sweep config and WANDB credentials."
    echo "WandB sweep output:"
    echo "$SWEEP_OUTPUT"
    exit 1
fi

echo "=========================================="
echo "SWEEP CREATED"
echo "SWEEP ID: $SWEEP_ID"
echo "PROJECT:  $PROJECT_NAME"
echo "AGENTS:   $NUM_AGENTS workers will be submitted"
echo "=========================================="

# 2. Submit sweep agents via Slurm
for i in $(seq 1 $NUM_AGENTS); do
    echo "Submitting sweep agent #$i..."
    sbatch scripts/sweep_agent.slurm "$SWEEP_ID" "$PROJECT_NAME" 10
done

echo "All sweep agents submitted. Monitor with:  squeue -u $USER"
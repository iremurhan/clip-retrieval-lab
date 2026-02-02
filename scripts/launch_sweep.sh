#!/bin/bash
# Launch a W&B hyperparameter sweep and submit multiple Slurm agents.
# Usage:
#   bash scripts/launch_sweep.sh [NUM_AGENTS]
#     NUM_AGENTS (optional) – number of parallel sweep workers (default: 3)
#
# Examples:
#   # Run 3 workers (default)
#   bash scripts/launch_sweep.sh
#
#   # Run 5 workers
#   bash scripts/launch_sweep.sh 5

NUM_AGENTS=${1:-3} # Number of parallel sweep workers (default: 3)
CONFIG_PATH="configs/sweep.yaml"
PROJECT_NAME="retrieval"

# 1. Launch the W&B sweep inside the container and capture the SWEEP_ID
echo ">>> Launching W&B sweep inside container and capturing SWEEP_ID..."
SWEEP_ID=$(srun --container-image=biremurhan/image-text-contrast:v0.4 \
     --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/env:/env \
     --container-env=WANDB_API_KEY \
     bash -c "source /env/wandb.env && cd /workspace && wandb sweep $CONFIG_PATH 2>&1 | grep 'wandb agent' | awk '{print \$NF}'")

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to obtain SWEEP_ID. Check sweep config and WANDB credentials."
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
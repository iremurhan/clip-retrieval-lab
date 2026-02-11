#!/bin/bash
#SBATCH -p palamut-cuda                # TRUBA partition
#SBATCH -A burhan                     # Account name
#SBATCH -J training_job
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16            # 8-10 is ideal for data loading
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00               # Time limit required

# --- 1. Preparation ---
# module purge and load lines removed as requested

# --- Parsing Arguments ---
EXP_NAME="${1:-default_run}"           # Arg 1: Experiment Name
CONFIG_FILE="${2:-configs/config_coco.yaml}" # Arg 2: Config Path

# --- 2. Path Settings ---

# 1. Location of the code (HOME - Host)
HOST_REPO_DIR="/arf/home/burhan/clip-retrieval-lab"

# 2. Location of datasets (SCRATCH - Host)
HOST_DATA_DIR="/arf/scratch/burhan/experiments/datasets"

# 3. Location for results (SCRATCH - Host)
HOST_RESULTS_DIR="/arf/scratch/burhan/experiments/results"

# 4. Location of the env folder (HOME - Host)
HOST_ENV_FILE="/arf/home/burhan/experiments/env/wandb.env"

# Container Image File (Created in scratch)
SIF_IMAGE="/arf/scratch/burhan/image-text-contrast_v0.4.sif"

# --- BIND MOUNTS ---
# Maps Host paths to Container paths
# We map the directory containing the env file to /env
BINDS="$HOST_REPO_DIR:/workspace,$HOST_DATA_DIR:/workspace/datasets,$HOST_RESULTS_DIR:/output/results,$(dirname $HOST_ENV_FILE):/env"

# --- WandB Setup ---
# Source the env file on the HOST to get the API key
if [ -f "$HOST_ENV_FILE" ]; then
    set -a
    source "$HOST_ENV_FILE"
    set +a
else
    echo "WARNING: WandB env file ($HOST_ENV_FILE) not found!"
fi

export WANDB_PROJECT="retrieval"
export WANDB_NAME="${SLURM_JOB_ID}_${EXP_NAME}"

# CRITICAL FIX: Use CONTAINER INTERNAL PATH for WANDB_DIR
CONTAINER_JOB_DIR="/output/results/${SLURM_JOB_ID}"
# WandB needs a writable directory inside the container
export WANDB_DIR="${CONTAINER_JOB_DIR}/wandb" 

# --- 3. Execution ---
echo "Starting... Job: $SLURM_JOB_ID, Exp: $EXP_NAME, Config: $CONFIG_FILE"

# Change directory to repo root (Host side, but Apptainer handles mapping)
cd "$HOST_REPO_DIR"

# apptainer exec --nv (Required for Nvidia GPU support)
# --pwd /workspace ensures we start inside the container's mapped workspace
apptainer exec --nv -B $BINDS --pwd /workspace $SIF_IMAGE python3 -u run.py \
    --config "$CONFIG_FILE" \
    --override \
        debug.debug_mode=false \
        mining.consensus_path="/workspace/datasets/coco/pairwise_similarities/mining_image_consensus_mean.pt" \
        logging.checkpoint_dir="$CONTAINER_JOB_DIR" \
        logging.wandb_project="$WANDB_PROJECT"

echo "Process Completed."
#!/bin/bash
#SBATCH -p barbun-cuda                # TRUBA partition
#SBATCH -A burhan                     # Account name
#SBATCH -J training_job
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10            # 8-10 is ideal for data loading
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00               # Time limit required

# --- 1. Preparation ---
module purge
module load centos7.3/app/apptainer/1.1.8   # Load Apptainer module

# WandB API Key (Can also be read from .env if available)
export WANDB_API_KEY="YOUR_API_KEY_HERE"
export WANDB_MODE=online
export WANDB_PROJECT="retrieval"

# --- 2. Path Settings ---
# --- UPDATED PATHS ---

# 1. Location of the code (HOME)
HOST_REPO_DIR="/arf/home/burhan/clip-retrieval-lab"

# 2. Location of datasets (SCRATCH - Necessary for speed)
HOST_DATA_DIR="/arf/scratch/burhan/experiments/datasets"

# 3. Location for results (SCRATCH - To avoid quota issues)
HOST_RESULTS_DIR="/arf/scratch/burhan/experiments/results"

# 4. Location of the env folder (HOME)
HOST_ENV_DIR="/arf/home/burhan/experiments/env"

# Container Image File (Created in scratch)
SIF_IMAGE="/arf/scratch/burhan/image-text-contrast_v0.4.sif"

# --- BIND MOUNTS ---
# Left side TRUBA (Host), Right side Container (SAME as Docker)
# Added /env folder.
BINDS="$HOST_REPO_DIR:/workspace,$HOST_DATA_DIR:/workspace/datasets,$HOST_RESULTS_DIR:/output/results,$HOST_ENV_DIR:/env"

# Change directory to repo root
cd "$HOST_REPO_DIR"

# --- 3. Execution ---
echo "Starting..."

# apptainer exec --nv (Required for Nvidia GPU support)
apptainer exec --nv -B $BINDS --pwd /workspace $SIF_IMAGE python3 -u run.py \
    --config configs/config_coco.yaml \
    --override \
        debug.debug_mode=false \
        mining.consensus_path="/workspace/datasets/coco/pairwise_similarities/mining_image_consensus_mean.pt" \
        logging.checkpoint_dir="/output/results/${SLURM_JOB_ID}" \
        logging.wandb_project="$WANDB_PROJECT"

echo "Process Completed."
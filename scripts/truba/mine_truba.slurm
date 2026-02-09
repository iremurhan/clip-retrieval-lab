#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A burhan
#SBATCH -J mining_job
#SBATCH -o logs/%j_mining.out
#SBATCH -e logs/%j_mining.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module purge
module load centos7.3/app/apptainer/1.1.8

# Dataset selection (from script argument or manual)
DATASET="${TARGET_DATASET:-coco}" 
if [ "$DATASET" == "flickr30k" ]; then
    CONFIG_FILE="configs/config_flickr.yaml"
else
    CONFIG_FILE="configs/config_coco.yaml"
fi

# Paths
HOST_REPO_DIR="/arf/home/burhan/clip-retrieval-lab"
HOST_DATA_DIR="/arf/scratch/burhan/experiments/datasets"
HOST_RESULTS_DIR="/arf/scratch/burhan/experiments/results"
SIF_IMAGE="/arf/scratch/burhan/image-text-contrast_v0.4.sif"

# Bind Mounts (Keep container paths same as Docker)
BINDS="$HOST_REPO_DIR:/workspace,$HOST_DATA_DIR:/workspace/datasets,$HOST_RESULTS_DIR:/output/results"

# Change directory to repo root to ensure relative paths work
cd "$HOST_REPO_DIR"

echo "Mining starting: $DATASET"

apptainer exec --nv -B $BINDS --pwd /workspace $SIF_IMAGE python3 -u tools/mine_pairwise_sim.py \
    --modality caption \
    --config "$CONFIG_FILE" \
    --top_k 1000 \
    --batch_size 512 \
    --query_chunk_size 5000

echo "Done."
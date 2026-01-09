#!/bin/bash
#SBATCH -p normal
#SBATCH --time=24:00:00
#SBATCH --account=root
#SBATCH --job-name=pairwise_mining
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pairwise_out-%j.out
#SBATCH --error=logs/pairwise_err-%j.err

# 1. Environment Setup (taken from slurm.sh)
module load Python/3.12.0 cuda/10.1

# Exit on error
set -e

# 2. Parameters
MODE="${1:-mean}"        # Default mode: mean
BATCH_SIZE="${2:-32}"    # Reduced to 32 due to OOM error (reduce to 16 if needed)

# 3. Prepare Directories
mkdir -p datasets/coco
mkdir -p logs

# 4. Output File Path
OUTPUT_PATH="datasets/coco/pairwise_mining_${MODE}.pt"
LOG_FILE="logs/pairwise_mining_${MODE}.log"

echo "=========================================="
echo "Task:        Pairwise Image Similarity Mining"
echo "Mode:        ${MODE}"
echo "Batch Size:  ${BATCH_SIZE}"
echo "Python:      $(which python3.12)"
echo "Output:      ${OUTPUT_PATH}"
echo "=========================================="

# 5. Wandb Configuration (sets offline if key is missing)
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=offline
fi

# 6. Run Python Code
# Note: using python3.12 as in slurm.sh
python3.12 -u tools/calculate_pairwise_similarity.py \
    --config config.yaml \
    --mode "$MODE" \
    --batch_size "$BATCH_SIZE" \
    --output "$OUTPUT_PATH" \
    2>&1 | tee "$LOG_FILE"

# 7. Result
if [ $? -eq 0 ]; then
    echo "Task completed: $OUTPUT_PATH"
else
    echo "Error occurred! Check the log: $LOG_FILE"
    exit 1
fi
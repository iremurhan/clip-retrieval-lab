#!/bin/bash

# Usage: ./scripts/start_mining.sh [coco | flickr30k]
# Example: ./scripts/start_mining.sh flickr30k

# 1. Select dataset
DATASET=${1:-coco}  # Default 'coco' if no argument given

if [ "$DATASET" == "flickr30k" ] || [ "$DATASET" == "flickr" ]; then
    # Parameter passed to Slurm script
    EXPORT_CMD="ALL,TARGET_DATASET=flickr30k"
    FOLDER_NAME="flickr30k"
elif [ "$DATASET" == "coco" ]; then
    EXPORT_CMD="ALL,TARGET_DATASET=coco"
    FOLDER_NAME="coco"
else
    echo "ERROR: Invalid dataset. Use 'coco' or 'flickr30k'."
    exit 1
fi

# 2. Submit job (sbatch) and capture output
echo "Submitting job ($DATASET)..."
OUTPUT=$(sbatch --export=$EXPORT_CMD scripts/mine.slurm)

# Extract Job ID from output (e.g. "Submitted batch job 21060")
JOB_ID=$(echo "$OUTPUT" | awk '{print $4}')

# 3. Info
if [ -z "$JOB_ID" ]; then
    echo "ERROR: Job submission failed!"
    echo "$OUTPUT"
else
    echo "------------------------------------------------------"
    echo "Job submitted successfully: $JOB_ID"
    echo "Dataset mode:               $FOLDER_NAME"
    echo "Log file (when job finishes):"
    echo "   /users/beyza.urhan/experiments/results/$FOLDER_NAME/$JOB_ID/mining_log.out"
    echo "------------------------------------------------------"
fi
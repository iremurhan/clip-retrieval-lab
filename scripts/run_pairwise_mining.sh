#!/bin/bash
#
# Run Pairwise Image Similarity Mining on SLURM
#
# Usage:
#   sbatch scripts/run_pairwise_mining.sh           # Default: mode=mean
#   sbatch scripts/run_pairwise_mining.sh mean      # Explicit mode
#   sbatch scripts/run_pairwise_mining.sh max       # Optimistic matching
#   sbatch scripts/run_pairwise_mining.sh min       # Pessimistic matching
#   sbatch scripts/run_pairwise_mining.sh all       # Run all three modes
#

MODE="${1:-mean}"

echo "============================================"
echo "Pairwise Image Similarity Mining"
echo "============================================"
echo "Mode: $MODE"
echo ""

if [ "$MODE" == "all" ]; then
    echo "Submitting 3 jobs (mean, max, min)..."
    echo ""
    
    echo "Submitting mode=mean..."
    sbatch --export=MODE=mean scripts/pairwise_mining.slurm
    
    echo "Submitting mode=max..."
    sbatch --export=MODE=max scripts/pairwise_mining.slurm
    
    echo "Submitting mode=min..."
    sbatch --export=MODE=min scripts/pairwise_mining.slurm
else
    echo "Submitting mode=$MODE..."
    sbatch --export=MODE="$MODE" scripts/pairwise_mining.slurm
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "============================================"

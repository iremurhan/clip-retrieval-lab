#!/bin/bash
#
# Run Pairwise Image Similarity Mining on SLURM
#
# Usage:
#   ./scripts/run_pairwise_mining.sh           # Default: mode=mean
#   ./scripts/run_pairwise_mining.sh mean      # Explicit mode
#   ./scripts/run_pairwise_mining.sh max       # Optimistic matching
#   ./scripts/run_pairwise_mining.sh min       # Pessimistic matching
#   ./scripts/run_pairwise_mining.sh all       # Run all three modes
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/pairwise_mining.slurm"

MODE="${1:-mean}"

echo "============================================"
echo "Pairwise Image Similarity Mining"
echo "============================================"
echo "SLURM Script: $SLURM_SCRIPT"

if [ "$MODE" == "all" ]; then
    echo "Mode: ALL (submitting 3 jobs)"
    echo ""
    
    echo "Submitting mode=mean..."
    JOB_MEAN=$(sbatch --export=MODE=mean "$SLURM_SCRIPT" | awk '{print $4}')
    echo "  Job ID: $JOB_MEAN"
    
    echo "Submitting mode=max..."
    JOB_MAX=$(sbatch --export=MODE=max "$SLURM_SCRIPT" | awk '{print $4}')
    echo "  Job ID: $JOB_MAX"
    
    echo "Submitting mode=min..."
    JOB_MIN=$(sbatch --export=MODE=min "$SLURM_SCRIPT" | awk '{print $4}')
    echo "  Job ID: $JOB_MIN"
    
    echo ""
    echo "All jobs submitted:"
    echo "  mean: $JOB_MEAN"
    echo "  max:  $JOB_MAX"
    echo "  min:  $JOB_MIN"
else
    echo "Mode: $MODE"
    echo ""
    
    sbatch --export=MODE="$MODE" "$SLURM_SCRIPT"
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "============================================"

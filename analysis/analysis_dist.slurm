#!/bin/bash
#SBATCH --job-name=dist_analysis
#SBATCH --output=logs/%j_analysis.log
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# COCO Analizi
python3 tools/analyze_distributions.py --dataset_name coco --output_dir analysis_results

# Flickr Analizi (Opsiyonel - Dataset varsa)
# python3 tools/analyze_distributions.py --dataset_name flickr30k --output_dir analysis_results
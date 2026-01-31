#!/bin/bash
#SBATCH --job-name=mining_coco
#SBATCH --output=mining_log_%j.out  # Log dosyası buraya oluşacak
#SBATCH --error=mining_log_%j.out
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=beyza.urhan@bogazici.edu.tr

# ------------------------------------------------------------------
# MOUNT AYARLARI (Sunucu Yolları -> Container Yolları)
# ------------------------------------------------------------------
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env

# WandB ve diğer değişkenleri içeri aktar
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE

set -euo pipefail

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Date:          $(date)"
echo "=========================================="

# ------------------------------------------------------------------
# KONTROLLER
# ------------------------------------------------------------------
# Container içinde mountlar çalışmış mı?
if [ ! -d "/workspace/datasets/coco" ]; then
    echo "HATA: /workspace/datasets/coco klasörü bulunamadı."
    echo "Lütfen host makinede '/users/beyza.urhan/experiments/datasets/coco' yolunu kontrol edin."
    exit 1
fi

if [ ! -d "/output/results" ]; then
    echo "HATA: /output/results klasörü bulunamadı."
    exit 1
fi

# ------------------------------------------------------------------
# ÇALIŞTIRMA
# ------------------------------------------------------------------
cd /workspace

# Text-Text Mining Başlat (COCO için)
echo "Mining Başlıyor..."

python3 -u tools/mine_pairwise_sim.py \
    --modality caption \
    --config configs/config_coco.yaml \
    --top_k 1000 \
    --batch_size 512 \
    --query_chunk_size 5000

echo "Mining Tamamlandı."
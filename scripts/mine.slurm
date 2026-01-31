#!/bin/bash
#SBATCH --job-name=mining_coco
#SBATCH --output=mining_log_%j.out
#SBATCH --error=mining_log_%j.out
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=beyza.urhan@bogazici.edu.tr

# Mount Ayarları
#SBATCH --container-mounts=/users/beyza.urhan/tez_v2_clean:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env

# WandB değişkenlerini aktar
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT

set -euo pipefail

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Date:          $(date)"
echo "=========================================="

# ------------------------------------------------------------------
# WANDB KURULUMU (Eksik olan kısım burasıydı)
# ------------------------------------------------------------------
# Eğer env klasöründe wandb.env varsa onu yükle
if [ -f /env/wandb.env ]; then
    echo "WandB environment dosyası yükleniyor..."
    set -a
    source /env/wandb.env
    set +a
else
    echo "UYARI: /env/wandb.env bulunamadı. API Key eksik olabilir."
fi

# 2. PROJE ADINI BURADA BELİRLİYORUZ (Dosyadakini Ezer)
#    İstediğiniz proje adını tırnak içine yazın:
export WANDB_PROJECT="mining-coco"

# Run adını da dinamik yapalım (Job ID eklensin)
export WANDB_NAME="mining_top1000_${SLURM_JOB_ID}"

echo "Aktif WandB Projesi: $WANDB_PROJECT"

# ------------------------------------------------------------------
# ÇALIŞTIRMA
# ------------------------------------------------------------------
cd /workspace

echo "Mining Başlıyor..."

python3 -u tools/mine_pairwise_sim.py \
    --modality caption \
    --config configs/config_coco.yaml \
    --top_k 1000 \
    --batch_size 512 \
    --query_chunk_size 5000

echo "Mining Tamamlandı."

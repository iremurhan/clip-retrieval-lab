#!/bin/bash
#SBATCH --job-name=mining_sim
#SBATCH --output=logs/%x_%j.out   # Log dosyası (logs klasörü olmalı)
#SBATCH --error=logs/%x_%j.err    # Hata log dosyası
#SBATCH --nodes=1                 # Tek node
#SBATCH --ntasks=1                # Tek görev
#SBATCH --gres=gpu:1              # 1 adet GPU
#SBATCH --cpus-per-task=8         # Veri yükleme için CPU çekirdekleri
#SBATCH --mem=32G                 # RAM miktarı (Dataset boyutuna göre artırılabilir)
#SBATCH --time=04:00:00           # Zaman sınırı (Saat:Dakika:Saniye)
#SBATCH --partition=gpu           # Partition adı (Cluster'ınıza göre değiştirin)

# Hata durumunda durdur ve komutları ekrana bas
set -e
set -x

# -----------------------------------------------------------------------
# 1. Ortam Değişkenleri ve Sanal Ortam
# -----------------------------------------------------------------------
# Cluster'a özgü modülleri yükleyin (Gerekirse)
# module load cuda/11.8
# module load python/3.10

# Conda veya venv ortamınızı aktif edin
source activate my_env_name  # Kendi ortam adınızı buraya yazın

# Python path ayarı (Proje ana dizinini görmesi için)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# WandB Ayarları (Kodunuzda wandb kullanıldığı için)
export WANDB_PROJECT="knowledge-distillation-mining"
# export WANDB_MODE="offline" # İnternet yoksa offline mod
# export WANDB_API_KEY="YOUR_KEY_HERE"

# -----------------------------------------------------------------------
# 2. Parametreler
# -----------------------------------------------------------------------
# mine_pairwise_sim.py içindeki argparse seçeneklerine göre ayarlar
# Seçenekler: "visual", "consensus", "caption"
MODALITY="caption" 

CONFIG_PATH="config.yaml"
TOP_K=1000
BATCH_SIZE=512
CHUNK_SIZE=5000

echo "Başlangıç Zamanı: $(date)"
echo "Çalıştırılan Modality: $MODALITY"
echo "Config Dosyası: $CONFIG_PATH"

# -----------------------------------------------------------------------
# 3. Python Scriptini Çalıştırma
# -----------------------------------------------------------------------
python tools/mine_pairwise_sim.py \
    --modality $MODALITY \
    --config $CONFIG_PATH \
    --top_k $TOP_K \
    --batch_size $BATCH_SIZE \
    --query_chunk_size $CHUNK_SIZE \
    --output_dir "./datasets/pairwise_similarities"

echo "Bitiş Zamanı: $(date)"
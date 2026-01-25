#!/bin/bash
#
# scripts/setup/download_flickr.sh
#
# Description:
#   Automates the downloading and setup of the Flickr30k dataset.
#   Uses 'python3 -m kaggle' to bypass 'noexec' permission issues.
#
# Usage:
#   bash scripts/setup/download_flickr.sh
#

set -e

TARGET_DIR="datasets/flickr30k"
KAGGLE_DATASET="hsankesara/flickr-image-dataset"

echo "========================================================"
echo "  Setting up Flickr30k Dataset"
echo "========================================================"
echo "Target Directory: $TARGET_DIR"

mkdir -p "$TARGET_DIR"

# --- 1. Check Prerequisites ---
# Check if we can import kaggle module in python
if ! python3 -c "import kaggle" &> /dev/null; then
    echo "[ERROR] 'kaggle' python module not found."
    echo "Please install it using: pip install kaggle"
    exit 1
fi

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "[ERROR] Kaggle API key not found at ~/.kaggle/kaggle.json"
    echo "Please download your API key from Kaggle Settings and place it there."
    echo "Ensure permissions are correct: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# --- 2. Download Images via Kaggle ---
echo ""
echo "[1/3] Downloading images via Kaggle API..."

# Check if images already exist to avoid re-downloading
if [ -d "$TARGET_DIR/images" ] && [ $(find "$TARGET_DIR/images" -name "*.jpg" | wc -l) -gt 30000 ]; then
    echo "      Images found. Skipping download."
else
    # Download and unzip directly to target using python3 -m kaggle
    python3 -m kaggle datasets download -d "$KAGGLE_DATASET" -p "$TARGET_DIR" --unzip

    echo "      Organizing directory structure..."
    
    # The Kaggle dataset usually extracts into a nested folder like 'flickr30k_images/flickr30k_images/...'
    # We consolidate everything into $TARGET_DIR/images
    
    mkdir -p "$TARGET_DIR/images"
    
    # Find and move all jpg files to the target images folder
    find "$TARGET_DIR" -type f -name "*.jpg" -exec mv {} "$TARGET_DIR/images/" \;
    
    # Clean up empty folders left behind by the unzip
    find "$TARGET_DIR" -type d -name "flickr30k_images" -exec rm -rf {} +
    
    echo "      Cleanup complete."
fi

# Verify image count
IMG_COUNT=$(find "$TARGET_DIR/images" -name "*.jpg" | wc -l)
echo "      Total images: $IMG_COUNT"

if [ "$IMG_COUNT" -lt 31000 ]; then
    echo "[WARNING] Expected ~31,783 images, found only $IMG_COUNT."
fi

# --- 3. Download Karpathy Splits ---
echo ""
echo "[2/3] Downloading Karpathy Splits (JSON)..."

if [ ! -f "$TARGET_DIR/dataset_flickr30k.json" ]; then
    wget -q https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip -O caption_datasets.zip
    unzip -q -o caption_datasets.zip
    mv dataset_flickr30k.json "$TARGET_DIR/"
    
    # Remove unnecessary files
    rm caption_datasets.zip dataset_coco.json dataset_flickr8k.json
    echo "      Downloaded dataset_flickr30k.json"
else
    echo "      dataset_flickr30k.json already exists."
fi

echo ""
echo "========================================================"
echo "  Setup Complete!"
echo "  Images: $(pwd)/$TARGET_DIR/images/"
echo "  Splits: $(pwd)/$TARGET_DIR/dataset_flickr30k.json"
echo "========================================================"
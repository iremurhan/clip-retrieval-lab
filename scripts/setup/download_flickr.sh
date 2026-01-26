#!/bin/bash
#
# scripts/setup/download_flickr.sh
#
# Description:
#   Automates the downloading and setup of the Flickr30k dataset.
#   Uses HuggingFace Hub for reliable download (no Kaggle API required).
#
# Usage:
#   bash scripts/setup/download_flickr.sh
#

set -e

TARGET_DIR="datasets/flickr30k"
HF_DATASET="nlphuji/flickr30k"

echo "========================================================"
echo "  Setting up Flickr30k Dataset"
echo "========================================================"
echo "Target Directory: $TARGET_DIR"

mkdir -p "$TARGET_DIR"

# --- 1. Check Prerequisites ---
if ! python3 -c "import huggingface_hub" &> /dev/null; then
    echo "[ERROR] 'huggingface_hub' python module not found."
    echo "Please install it using: pip install huggingface_hub"
    exit 1
fi

# --- 2. Download Images via HuggingFace Hub ---
echo ""
echo "[1/3] Downloading images via HuggingFace Hub..."

# Check if images already exist to avoid re-downloading
if [ -d "$TARGET_DIR/flickr30k_images" ] && [ $(find "$TARGET_DIR/flickr30k_images" -name "*.jpg" | wc -l) -gt 30000 ]; then
    echo "      Images found. Skipping download."
else
    # Download using huggingface_hub
    python3 << EOF
from huggingface_hub import snapshot_download
import os

target_dir = "$TARGET_DIR"
repo_id = "$HF_DATASET"

print("      Downloading from HuggingFace Hub...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=target_dir,
    local_dir_use_symlinks=False
)
print("      Download complete.")
EOF

    echo "      Organizing directory structure..."
    
    # HuggingFace usually downloads to flickr30k_images/ or images/
    if [ -d "$TARGET_DIR/flickr30k_images" ]; then
        echo "      Found flickr30k_images/ directory"
    elif [ -d "$TARGET_DIR/images" ]; then
        echo "      Found images/ directory"
        # Rename to flickr30k_images for consistency
        mv "$TARGET_DIR/images" "$TARGET_DIR/flickr30k_images"
    else
        echo "      Searching for image directory..."
        # Find the directory with most jpg files
        IMG_DIR=$(find "$TARGET_DIR" -type d -exec sh -c 'echo "$(find "$1" -name "*.jpg" | wc -l) $1"' _ {} \; | sort -rn | head -1 | cut -d' ' -f2-)
        if [ -n "$IMG_DIR" ] && [ "$IMG_DIR" != "$TARGET_DIR" ]; then
            mv "$IMG_DIR" "$TARGET_DIR/flickr30k_images"
            echo "      Moved to flickr30k_images/"
        else
            echo "[WARNING] Could not find image directory. Please check manually."
        fi
    fi
fi

# Verify image count
if [ -d "$TARGET_DIR/flickr30k_images" ]; then
    IMG_COUNT=$(find "$TARGET_DIR/flickr30k_images" -name "*.jpg" | wc -l)
    echo "      Total images: $IMG_COUNT"
    
    if [ "$IMG_COUNT" -lt 31000 ]; then
        echo "[WARNING] Expected ~31,783 images, found only $IMG_COUNT."
    fi
else
    echo "[WARNING] flickr30k_images directory not found."
fi

# --- 3. Download Karpathy Splits (JSON) ---
echo ""
echo "[2/3] Downloading Karpathy Splits (JSON)..."

# Create caption_datasets directory if needed
mkdir -p "$TARGET_DIR/caption_datasets"

JSON_PATH="$TARGET_DIR/caption_datasets/dataset_flickr30k.json"

if [ ! -f "$JSON_PATH" ]; then
    # Download from Stanford
    cd "$TARGET_DIR"
    wget -q https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip -O caption_datasets.zip
    unzip -q -o caption_datasets.zip
    
    # Move to caption_datasets subdirectory
    mv dataset_flickr30k.json caption_datasets/
    
    # Remove unnecessary files
    rm -f caption_datasets.zip dataset_coco.json dataset_flickr8k.json
    echo "      Downloaded dataset_flickr30k.json"
else
    echo "      dataset_flickr30k.json already exists."
fi

# --- 4. Final Verification ---
echo ""
echo "[3/3] Verifying setup..."

if [ -d "$TARGET_DIR/flickr30k_images" ]; then
    IMG_COUNT=$(find "$TARGET_DIR/flickr30k_images" -name "*.jpg" | wc -l)
    echo "      Images: $IMG_COUNT found in flickr30k_images/"
else
    echo "      [WARNING] flickr30k_images/ directory not found"
fi

if [ -f "$JSON_PATH" ]; then
    echo "      JSON: dataset_flickr30k.json found"
else
    echo "      [WARNING] dataset_flickr30k.json not found"
fi

echo ""
echo "========================================================"
echo "  Setup Complete!"
echo "  Images: $TARGET_DIR/flickr30k_images/"
echo "  JSON:   $JSON_PATH"
echo "========================================================"
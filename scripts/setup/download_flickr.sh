#!/bin/bash
# scripts/setup/download_flickr.sh
#
# Downloads Flickr30k dataset from HuggingFace Hub.
# Uses huggingface-cli for reliable download.
#
# Dataset: nlphuji/flickr30k
# Contents: images.tar.gz + dataset.json (Karpathy format)
#
# Usage:
#   ./scripts/setup/download_flickr.sh
#   DATA_DIR=/custom/path ./scripts/setup/download_flickr.sh

set -e  # Exit on error

# Target directory (can be overridden via environment variable)
DATA_DIR="${DATA_DIR:-$HOME/experiments/datasets/flickr30k}"

echo "=============================================="
echo "Downloading Flickr30k Dataset"
echo "=============================================="
echo "Target directory: $DATA_DIR"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "[ERROR] huggingface-cli not found."
    echo "Install with: pip install huggingface_hub"
    exit 1
fi

# Create directory
mkdir -p "$DATA_DIR"

# Check if already downloaded
if [ -d "$DATA_DIR/images" ] && [ -f "$DATA_DIR/dataset_flickr30k.json" ]; then
    echo "[OK] Flickr30k already downloaded and extracted."
    echo "Structure:"
    ls -lh "$DATA_DIR"
    exit 0
fi

# Download from HuggingFace Hub
echo "[1/3] Downloading from HuggingFace Hub..."
echo "      This may take a while depending on your connection speed."
echo ""

huggingface-cli download nlphuji/flickr30k \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "[2/3] Download complete. Checking contents..."

# List downloaded files
ls -lh "$DATA_DIR"

# Extract images if tar.gz exists
if [ -f "$DATA_DIR/images.tar.gz" ]; then
    echo ""
    echo "[3/3] Extracting images.tar.gz..."
    
    cd "$DATA_DIR"
    tar -xzf images.tar.gz
    
    # Verify extraction
    if [ -d "images" ] || [ -d "flickr30k-images" ]; then
        echo "[OK] Images extracted successfully."
        
        # Rename if needed (some versions use flickr30k-images)
        if [ -d "flickr30k-images" ] && [ ! -d "images" ]; then
            mv flickr30k-images images
            echo "     Renamed flickr30k-images -> images"
        fi
        
        # Optionally remove tar.gz to save space
        read -p "Remove images.tar.gz to save space? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm images.tar.gz
            echo "[OK] Removed images.tar.gz"
        fi
    else
        echo "[WARN] Extraction may have failed. Check contents manually."
    fi
elif [ -d "$DATA_DIR/images" ]; then
    echo "[3/3] Images already extracted. Skipping."
else
    echo "[WARN] images.tar.gz not found. Dataset may be incomplete."
    echo "       Check HuggingFace repo for the correct structure."
fi

# Rename/copy dataset JSON if needed
if [ -f "$DATA_DIR/dataset.json" ] && [ ! -f "$DATA_DIR/dataset_flickr30k.json" ]; then
    cp "$DATA_DIR/dataset.json" "$DATA_DIR/dataset_flickr30k.json"
    echo "[OK] Copied dataset.json -> dataset_flickr30k.json"
fi

# Final verification
echo ""
echo "=============================================="
echo "Download Complete!"
echo "=============================================="
echo "Directory: $DATA_DIR"
echo ""
echo "Expected structure:"
echo "  flickr30k/"
echo "    images/          (31,783 images)"
echo "    dataset_flickr30k.json  (Karpathy splits)"
echo ""
echo "Current contents:"
ls -lh "$DATA_DIR"

# Count images if directory exists
if [ -d "$DATA_DIR/images" ]; then
    IMG_COUNT=$(find "$DATA_DIR/images" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
    echo ""
    echo "Image count: $IMG_COUNT"
fi

echo "=============================================="

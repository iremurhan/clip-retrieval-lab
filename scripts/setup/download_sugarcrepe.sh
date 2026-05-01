#!/usr/bin/env bash
# =============================================================================
# download_sugarcrepe.sh — Download SugarCrepe dataset for compositional eval
# =============================================================================
# Downloads the SugarCrepe (NeurIPS'23) compositional understanding benchmark.
# Also downloads the COCO 2017 validation images used by SugarCrepe.
#
# Output structure:
#   /users/beyza.urhan/experiments/datasets/sugarcrepe/
#       add_att.json
#       add_obj.json
#       replace_att.json
#       replace_obj.json
#       replace_rel.json
#       swap_att.json
#       swap_obj.json
#   /users/beyza.urhan/experiments/datasets/coco/val2017/
#       000000000139.jpg
#       ...
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# SugarCrepe downloads to shared experiments directory
SUGARCREPE_DIR="/users/beyza.urhan/experiments/datasets/sugarcrepe"
COCO_DIR="/users/beyza.urhan/experiments/datasets/coco"
COCO_VAL_DIR="${COCO_DIR}/val2017"
COCO_VAL_ZIP="${COCO_DIR}/val2017.zip"
COCO_VAL_URL="http://images.cocodataset.org/zips/val2017.zip"

# Create target directories
mkdir -p "$SUGARCREPE_DIR"
mkdir -p "$COCO_DIR"

echo "Downloading SugarCrepe dataset to: $SUGARCREPE_DIR"

# SugarCrepe official release URL (RAIVNLab - official source)
SUGARCREPE_REPO="https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/main/data"

# Subcategories to download
SUBCATEGORIES=(
    "add_att"
    "add_obj"
    "replace_att"
    "replace_obj"
    "replace_rel"
    "swap_att"
    "swap_obj"
)

for subcat in "${SUBCATEGORIES[@]}"; do
    JSON_FILE="${SUGARCREPE_DIR}/${subcat}.json"
    
    if [ -f "$JSON_FILE" ]; then
        echo "  ✓ $subcat.json already exists, skipping"
        continue
    fi
    
    URL="${SUGARCREPE_REPO}/${subcat}.json"
    echo "  Downloading: $subcat.json"
    
    if ! curl -f -L -o "$JSON_FILE" "$URL"; then
        echo "  ERROR: Failed to download $subcat.json from $URL"
        exit 1
    fi
    
    # Verify JSON is valid
    if ! python3 -c "import json; json.load(open('$JSON_FILE'))" 2>/dev/null; then
        echo "  ERROR: Downloaded file is not valid JSON: $JSON_FILE"
        rm "$JSON_FILE"
        exit 1
    fi
done

echo ""
if find "$COCO_VAL_DIR" -maxdepth 1 -name '*.jpg' -print -quit 2>/dev/null | grep -q .; then
    echo "COCO val2017 images already exist at: $COCO_VAL_DIR"
else
    echo "Downloading COCO val2017 images to: $COCO_VAL_DIR"
    echo "  URL: $COCO_VAL_URL"
    curl -f -L -C - -o "$COCO_VAL_ZIP" "$COCO_VAL_URL"

    echo "Unzipping COCO val2017..."
    unzip -q "$COCO_VAL_ZIP" -d "$COCO_DIR"
fi

if ! find "$COCO_VAL_DIR" -maxdepth 1 -name '*.jpg' -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: COCO val2017 images were not found after setup: $COCO_VAL_DIR"
    exit 1
fi

if [ -f "$COCO_VAL_ZIP" ]; then
    echo "Removing downloaded archive: $COCO_VAL_ZIP"
    rm "$COCO_VAL_ZIP"
fi

echo ""
echo "========================================================"
echo "SugarCrepe dataset downloaded successfully!"
echo "Location: $SUGARCREPE_DIR"
echo "COCO val2017 images: $COCO_VAL_DIR"
echo "========================================================"
echo ""
echo "Verify by running:"
echo "  ls -la $SUGARCREPE_DIR"
echo "  ls -la $COCO_VAL_DIR | head"
echo ""
echo "SugarCrepe evaluation will now be enabled at end-of-training."

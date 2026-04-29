#!/usr/bin/env bash
# =============================================================================
# download_sugarcrepe.sh — Download SugarCrepe dataset for compositional eval
# =============================================================================
# Downloads the SugarCrepe (NeurIPS'23) compositional understanding benchmark.
# Expects COCO val2014 images to already exist at datasets/coco/val2014/
#
# Output structure:
#   datasets/sugarcrepe/
#       add_att.json
#       add_obj.json
#       replace_att.json
#       replace_obj.json
#       replace_rel.json
#       swap_att.json
#       swap_obj.json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# SugarCrepe downloads to shared experiments directory
SUGARCREPE_DIR="/users/beyza.urhan/experiments/datasets/sugarcrepe"
COCO_VAL_DIR="/users/beyza.urhan/experiments/datasets/coco/val2014"

# Create sugarcrepe directory
mkdir -p "$SUGARCREPE_DIR"

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
echo "========================================================"
echo "SugarCrepe dataset downloaded successfully!"
echo "Location: $SUGARCREPE_DIR"
echo "========================================================"
echo ""
echo "Verify by running:"
echo "  ls -la $SUGARCREPE_DIR"
echo ""
echo "SugarCrepe evaluation will now be enabled at end-of-training."

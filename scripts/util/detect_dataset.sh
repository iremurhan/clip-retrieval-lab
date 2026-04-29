#!/usr/bin/env bash
# =============================================================================
# detect_dataset.sh — Shared helper to extract dataset name from config YAML.
# =============================================================================
# Sources: scripts that previously had this logic inline (train.slurm,
# resume.slurm, start_training.sh).
#
# Usage (source, then call):
#   source "$(dirname "$0")/../util/detect_dataset.sh"
#   DATASET=$(detect_dataset "$CONFIG_FILE") || exit 1
# =============================================================================

detect_dataset() {
    local config_file="$1"
    python3 -c "
import sys, yaml
base = yaml.safe_load(open('configs/config_base.yaml'))
override = yaml.safe_load(open('${config_file}'))
base.setdefault('data', {}).update(override.get('data', {}))
d = base['data'].get('dataset')
if not d:
    sys.exit('ERROR: data.dataset not set in config')
print(d)
"
}

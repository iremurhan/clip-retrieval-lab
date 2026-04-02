#!/bin/bash
# =============================================================================
# Quick smoke test: verify eccv_caption metric keys match extraction code.
# No training, no checkpoint — just random similarities through the pipeline.
#
# Usage:
#   sbatch scripts/test_eccv_metrics.slurm
# =============================================================================
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --container-image=biremurhan/image-text-contrast:v0.14
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j_eccv_test.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j_eccv_test.out
#SBATCH --container-env=PYTHONPATH=/workspace

set -euo pipefail
cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

python3 -u - <<'PYEOF'
import numpy as np
from eccv_caption import Metrics
from src.metrics import build_ranked_dicts, compute_eccv_metrics

# --- Step 1: Verify raw eccv_caption return format ---
print("=" * 60)
print("Step 1: Raw eccv_caption library output format")
print("=" * 60)

m = Metrics()
# Use real COCO image/caption IDs from the library's ground truth
image_ids = sorted(m.coco_gts['i2t'].keys())
caption_ids = sorted(m.coco_gts['t2i'].keys())

print(f"  COCO test: {len(image_ids)} images, {len(caption_ids)} captions")

# Random similarity matrix [N_captions, N_images]
rng = np.random.default_rng(42)
sim = rng.standard_normal((len(caption_ids), len(image_ids)), dtype=np.float32)

i2t, t2i = build_ranked_dicts(sim, image_ids, caption_ids)

scores = m.compute_all_metrics(
    i2t, t2i,
    target_metrics=('eccv_r1', 'eccv_map_at_r', 'eccv_rprecision',
                    'coco_1k_recalls', 'coco_5k_recalls', 'cxc_recalls'),
    Ks=(1, 5, 10),
    verbose=False,
)

print(f"\n  Returned keys ({len(scores)}):")
for k in sorted(scores.keys()):
    v = scores[k]
    print(f"    {k}: {v}")

# --- Step 2: Validate our extraction code expects exactly these keys ---
print("\n" + "=" * 60)
print("Step 2: Validate extraction keys")
print("=" * 60)

EXPECTED_KEYS = [
    'coco_5k_r1', 'coco_5k_r5', 'coco_5k_r10',
    'coco_1k_r1', 'coco_1k_r5', 'coco_1k_r10',
    'cxc_r1', 'cxc_r5', 'cxc_r10',
    'eccv_map_at_r', 'eccv_rprecision',
]

missing = [k for k in EXPECTED_KEYS if k not in scores]
unexpected = [k for k in scores if k not in EXPECTED_KEYS]

if missing:
    print(f"  FAIL: Missing keys: {missing}")
else:
    print(f"  OK: All {len(EXPECTED_KEYS)} expected keys present.")

if unexpected:
    print(f"  INFO: Extra keys (not extracted): {unexpected}")

# --- Step 3: Verify each value is {i2t: float, t2i: float} ---
print("\n" + "=" * 60)
print("Step 3: Validate value format")
print("=" * 60)

bad = []
for k in EXPECTED_KEYS:
    if k not in scores:
        continue
    v = scores[k]
    if not isinstance(v, dict) or 'i2t' not in v or 't2i' not in v:
        bad.append((k, type(v).__name__, v))

if bad:
    print(f"  FAIL: Unexpected value format:")
    for k, t, v in bad:
        print(f"    {k} -> type={t}, value={v}")
else:
    print(f"  OK: All values are dict with 'i2t' and 't2i' float entries.")

# --- Step 4: Verify through compute_eccv_metrics wrapper ---
print("\n" + "=" * 60)
print("Step 4: End-to-end through compute_eccv_metrics()")
print("=" * 60)

scores2 = compute_eccv_metrics(sim, image_ids, caption_ids, dataset='coco')
assert scores2 == scores, "Wrapper returned different results than direct call!"
print("  OK: compute_eccv_metrics() wrapper matches direct call.")

# --- Final verdict ---
print("\n" + "=" * 60)
if missing or bad:
    print("RESULT: FAIL — fix extraction code before training.")
    exit(1)
else:
    print("RESULT: PASS — metric extraction is correct.")
    exit(0)
PYEOF

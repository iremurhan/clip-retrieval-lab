#!/bin/sh
set -eu

# Backfill post-hoc evaluation metrics into existing WandB runs.
#
# Usage:
#   sh scripts/eval/run_missing_wandb_evals.sh
#
# Optional overrides:
#   WANDB_PROJECT=clip-retrieval
#   WANDB_ENTITY=iremurhan-bogazici-university
#   RESULTS_ROOT=/Volumes/T7/Research/experiments/results
#   DATA_ROOT=/Volumes/T7/Research/experiments/datasets
#   WANDB_OUT_DIR=/Volumes/T7/Research/wandb
#   ARTIFACT_ROOT=/Volumes/T7/Research/artifacts/clip-retrieval-lab
#   DEVICE=auto
#   RUN_SUGARCREPE=1 RUN_MMVP=1 RUN_OOD_FLICKR_TO_COCO=1

PYTHON="${PYTHON:-python}"
EXPORT_WANDB_SCRIPT="scripts/util/export_wandb.py"
WANDB_PROJECT="${WANDB_PROJECT:-clip-retrieval}"
RESULTS_ROOT="${RESULTS_ROOT:-/Volumes/T7/Research/experiments/results}"
DATA_ROOT="${DATA_ROOT:-/Volumes/T7/Research/experiments/datasets}"
WANDB_OUT_DIR="${WANDB_OUT_DIR:-/Volumes/T7/Research/wandb}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/Volumes/T7/Research/artifacts/clip-retrieval-lab}"
DEVICE="${DEVICE:-auto}"

RUN_SUGARCREPE="${RUN_SUGARCREPE:-1}"
RUN_MMVP="${RUN_MMVP:-1}"
RUN_OOD_FLICKR_TO_COCO="${RUN_OOD_FLICKR_TO_COCO:-1}"

RUNS_CSV="${WANDB_OUT_DIR}/runs_summary.csv"
BACKFILL_DIR="${BACKFILL_DIR:-${ARTIFACT_ROOT}/wandb_eval_backfill}"
LOG_DIR="${LOG_DIR:-${BACKFILL_DIR}/logs}"
MANIFEST_DIR="${MANIFEST_DIR:-${BACKFILL_DIR}/manifests}"
CACHE_DIR="${CACHE_DIR:-${BACKFILL_DIR}/cache}"
WANDB_DIR="${WANDB_DIR:-${BACKFILL_DIR}/wandb}"

SUGARCREPE_MANIFEST="${SUGARCREPE_MANIFEST:-${MANIFEST_DIR}/sugarcrepe_manifest.csv}"
MMVP_MANIFEST="${MMVP_MANIFEST:-${MANIFEST_DIR}/mmvp_manifest.csv}"
OOD_FLICKR_TO_COCO_MANIFEST="${OOD_FLICKR_TO_COCO_MANIFEST:-${MANIFEST_DIR}/ood_flickr_to_coco_eccv_manifest.csv}"

mkdir -p "$LOG_DIR" "$MANIFEST_DIR" "$CACHE_DIR" "$WANDB_DIR"
export WANDB_DIR

echo "Artifacts:"
echo "  logs:      ${LOG_DIR}"
echo "  manifests: ${MANIFEST_DIR}"
echo "  cache:     ${CACHE_DIR}"
echo "  wandb:     ${WANDB_DIR}"

echo "Exporting WandB summaries..."
if [ -n "${WANDB_ENTITY:-}" ]; then
  "$PYTHON" "$EXPORT_WANDB_SCRIPT" \
    --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" \
    --out-dir "$WANDB_OUT_DIR"
else
  "$PYTHON" "$EXPORT_WANDB_SCRIPT" \
    --project "$WANDB_PROJECT" \
    --out-dir "$WANDB_OUT_DIR"
fi

echo "Building pending manifests from ${RUNS_CSV}..."
RUNS_CSV="$RUNS_CSV" \
RESULTS_ROOT="$RESULTS_ROOT" \
SUGARCREPE_MANIFEST="$SUGARCREPE_MANIFEST" \
MMVP_MANIFEST="$MMVP_MANIFEST" \
OOD_FLICKR_TO_COCO_MANIFEST="$OOD_FLICKR_TO_COCO_MANIFEST" \
"$PYTHON" - <<'PY'
import csv
import os
from pathlib import Path

from scripts.eval.discover_pending_sugarcrepe import discover_checkpoints

runs_csv = Path(os.environ["RUNS_CSV"])
results_root = Path(os.environ["RESULTS_ROOT"])

with runs_csv.open(encoding="utf-8") as f:
    runs = {row["id"]: row for row in csv.DictReader(f) if row.get("id")}

entries = [
    entry for entry in discover_checkpoints(str(results_root))
    if entry.get("wandb_run_id", "").strip()
]

def has_metric(row: dict, columns: list[str]) -> bool:
    for col in columns:
        value = str(row.get(col, "")).strip()
        if not value:
            continue
        try:
            float(value)
        except ValueError:
            continue
        return True
    return False

def write_manifest(path: str, rows: list[dict]) -> None:
    fieldnames = ["checkpoint_path", "dataset", "wandb_run_name", "wandb_run_id"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path}: {len(rows)} pending")

write_manifest(
    os.environ["SUGARCREPE_MANIFEST"],
    [
        entry for entry in entries
        if not has_metric(runs.get(entry["wandb_run_id"], {}), [
            "summary/sugarcrepe/macro_avg",
            "summary/sugarcrepe/overall",
        ])
    ],
)

write_manifest(
    os.environ["MMVP_MANIFEST"],
    [
        entry for entry in entries
        if not has_metric(runs.get(entry["wandb_run_id"], {}), [
            "summary/mmvp_vlm/overall",
        ])
    ],
)

# ECCV/CxC metrics are defined for COCO in this repo. For Flickr-trained runs,
# backfill them by running the cross-dataset evaluator on COCO and logging under
# summary/ood/flickr30k_to_coco/*.
write_manifest(
    os.environ["OOD_FLICKR_TO_COCO_MANIFEST"],
    [
        entry for entry in entries
        if entry["dataset"] == "flickr30k"
        and not has_metric(runs.get(entry["wandb_run_id"], {}), [
            "summary/ood/flickr30k_to_coco/eccv_map_at_r_i2t",
            "summary/ood/flickr30k_to_coco/eccv_map_at_r_t2i",
        ])
    ],
)
PY

if [ "$RUN_SUGARCREPE" = "1" ]; then
  echo "Running SugarCrepe backfill. Log: ${LOG_DIR}/sugarcrepe.log"
  "$PYTHON" scripts/eval/run_sugarcrepe_local.py \
    --manifest "$SUGARCREPE_MANIFEST" \
    --data-dir "${DATA_ROOT}/sugarcrepe" \
    --images-dir "${DATA_ROOT}/coco/val2017" \
    --device "$DEVICE" \
    --continue-on-error \
    < /dev/null \
    > "${LOG_DIR}/sugarcrepe.log" 2>&1
fi

if [ "$RUN_MMVP" = "1" ]; then
  echo "Running MMVP-VLM backfill. Log: ${LOG_DIR}/mmvp_vlm.log"
  "$PYTHON" scripts/eval/run_mmvp_local.py \
    --manifest "$MMVP_MANIFEST" \
    --data-dir "${DATA_ROOT}/mmvp_vlm" \
    --cache-dir "${CACHE_DIR}/mmvp_vlm" \
    --device "$DEVICE" \
    --force \
    --continue-on-error \
    --log-wandb \
    --wandb_project "$WANDB_PROJECT" \
    < /dev/null \
    > "${LOG_DIR}/mmvp_vlm.log" 2>&1
fi

if [ "$RUN_OOD_FLICKR_TO_COCO" = "1" ]; then
  echo "Running Flickr-to-COCO OOD/ECCV backfill. Log: ${LOG_DIR}/ood_flickr_to_coco_eccv.log"
  "$PYTHON" scripts/eval/run_ood_eval_local.py \
    --manifest "$OOD_FLICKR_TO_COCO_MANIFEST" \
    --data-root "$DATA_ROOT" \
    --cache-dir "${CACHE_DIR}/ood_eval" \
    --device "$DEVICE" \
    --force \
    --continue-on-error \
    --log-wandb \
    --wandb_project "$WANDB_PROJECT" \
    < /dev/null \
    > "${LOG_DIR}/ood_flickr_to_coco_eccv.log" 2>&1
fi

echo "Done. Re-export WandB later to verify:"
echo "  ${PYTHON} ${EXPORT_WANDB_SCRIPT} --project ${WANDB_PROJECT} --out-dir ${WANDB_OUT_DIR}"

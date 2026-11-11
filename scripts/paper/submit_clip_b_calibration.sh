#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PAPER_WANDB_PROJECT="cross-modal-retrieval-paper"
REGISTRY_FILE="configs/registry_paper_clip_b.yaml"
CONFIG_FILE="configs/config_coco.yaml"
EXPERIMENT_FAMILY="baseline_calibration"
WANDB_GROUP="baseline-calibration"
BACKBONE_MODEL_ID="openai/clip-vit-base-patch16"
DATASET="coco"
SEED="42"
NATIVE_IMAGE_RESOLUTION="224"
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
GIT_COMMIT="$(git rev-parse HEAD)"
SUBMIT_SLEEP_SECONDS="${SUBMIT_SLEEP_SECONDS:-45}"

backbone_slug() {
    case "$1" in
        openai/clip-vit-base-patch16) echo "clipb16" ;;
        *) echo "$1" | tr '/_' '--' ;;
    esac
}

BACKBONE_SLUG="$(backbone_slug "$BACKBONE_MODEL_ID")"
MANIFEST_DIR="results/manifests"
MANIFEST_PATH="${MANIFEST_DIR}/paper_clip_b_baseline_calibration_${TIMESTAMP}.jsonl"
mkdir -p "$MANIFEST_DIR"

append_manifest() {
    local job_id="$1"
    local registry_id="$2"
    local image_resolution="$3"
    local positional_interpolation="$4"
    local wandb_run_name="$5"
    local output_dir="$6"
    python3 - "$MANIFEST_PATH" <<PY
import json
import sys
record = {
    "scheduler_job_id": "${job_id}",
    "registry_file": "${REGISTRY_FILE}",
    "registry_id": "${registry_id}",
    "experiment_family": "${EXPERIMENT_FAMILY}",
    "backbone_model_id": "${BACKBONE_MODEL_ID}",
    "backbone_slug": "${BACKBONE_SLUG}",
    "image_resolution": int("${image_resolution}"),
    "native_image_resolution": int("${NATIVE_IMAGE_RESOLUTION}"),
    "positional_interpolation": "${positional_interpolation}".lower() == "true",
    "dataset": "${DATASET}",
    "seed": int("${SEED}"),
    "wandb_project": "${PAPER_WANDB_PROJECT}",
    "wandb_group": "${WANDB_GROUP}",
    "wandb_run_name": "${wandb_run_name}",
    "output_directory": "${output_dir}",
    "git_commit": "${GIT_COMMIT}",
    "submission_timestamp": "${TIMESTAMP}",
}
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

submit_zero_shot() {
    local registry_id="$1"
    local image_resolution="$2"
    local positional_interpolation="$3"
    local run_suffix="$4"
    local wandb_run_name="${BACKBONE_SLUG}_${run_suffix}"
    local job_id
    job_id="$(sbatch --parsable \
        --job-name="${wandb_run_name}" \
        scripts/paper/eval_zero_shot_clip_b.slurm \
        "$registry_id" "$image_resolution" "$positional_interpolation" "$SEED")"
    local output_dir="/output/results/paper/baseline-calibration/${BACKBONE_SLUG}/${registry_id}/r${image_resolution}/${DATASET}/s${SEED}/${job_id}"
    append_manifest "$job_id" "$registry_id" "$image_resolution" "$positional_interpolation" "$wandb_run_name" "$output_dir"
    printf "%-22s %s\n" "$registry_id" "$job_id"
}

submit_trained() {
    local registry_id="$1"
    local unfreeze_blocks="$2"
    local run_suffix="$3"
    local wandb_run_name="${BACKBONE_SLUG}_${run_suffix}"
    local job_id
    job_id="$(sbatch --parsable \
        --job-name="${wandb_run_name}" \
        scripts/paper/train_clip_b_calibration.slurm \
        "$registry_id" "$unfreeze_blocks" "$SEED")"
    local output_dir="/output/results/paper/baseline-calibration/${BACKBONE_SLUG}/${registry_id}/r336/${DATASET}/s${SEED}/${job_id}"
    append_manifest "$job_id" "$registry_id" "336" "true" "$wandb_run_name" "$output_dir"
    printf "%-22s %s\n" "$registry_id" "$job_id"
}

echo "Submitting first-stage CLIP-B baseline calibration runs"
echo "Manifest: ${MANIFEST_PATH}"
printf "%-22s %s\n" "registry_id" "job_id"
printf "%-22s %s\n" "----------------------" "------"

submit_zero_shot "zero_shot_336" "336" "true" "zero-shot-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"

submit_trained "minimal_projonly" "0" "minimal-projonly-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_trained "minimal_uf1" "1" "minimal-uf1-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_trained "minimal_uf2" "2" "minimal-uf2-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_trained "minimal_uf3" "3" "minimal-uf3-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_trained "minimal_uf4" "4" "minimal-uf4-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_trained "minimal_uf5" "5" "minimal-uf5-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_trained "minimal_uf6" "6" "minimal-uf6-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_trained "minimal_uf7" "7" "minimal-uf7-r336_coco_s42"

echo "Done. Manifest: ${MANIFEST_PATH}"

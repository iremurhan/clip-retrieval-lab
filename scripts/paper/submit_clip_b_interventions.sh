#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PAPER_WANDB_PROJECT="cross-modal-retrieval-paper"
REGISTRY_FILE="configs/registry_paper_clip_b.yaml"
CONFIG_FILE="configs/config_coco.yaml"
EXPERIMENT_FAMILY="paper_intervention"
WANDB_GROUP="paper-interventions"
BACKBONE_MODEL_ID="openai/clip-vit-base-patch16"
BACKBONE_SLUG="clipb16"
DATASET="coco"
SEED="42"
IMAGE_RESOLUTION="336"
NATIVE_IMAGE_RESOLUTION="224"
POSITIONAL_INTERPOLATION="true"
UNFREEZE_BLOCKS="1"
TIMESTAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
GIT_COMMIT="$(git rev-parse HEAD)"
SUBMIT_SLEEP_SECONDS="${SUBMIT_SLEEP_SECONDS:-45}"
CHAIN_JOBS="${CHAIN_JOBS:-false}"
DEPENDENCY_AFTERANY="${INITIAL_DEPENDENCY_AFTERANY:-}"

MANIFEST_DIR="results/manifests"
MANIFEST_PATH="${MANIFEST_DIR}/paper_clip_b_interventions_${TIMESTAMP}.jsonl"
mkdir -p "$MANIFEST_DIR"

append_manifest() {
    local job_id="$1"
    local registry_id="$2"
    local wandb_run_name="$3"
    local output_dir="$4"
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
    "image_resolution": int("${IMAGE_RESOLUTION}"),
    "native_image_resolution": int("${NATIVE_IMAGE_RESOLUTION}"),
    "positional_interpolation": "${POSITIONAL_INTERPOLATION}".lower() == "true",
    "dataset": "${DATASET}",
    "seed": int("${SEED}"),
    "selected_unfreeze_depth": int("${UNFREEZE_BLOCKS}"),
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

submit_intervention() {
    local registry_id="$1"
    local run_suffix="$2"
    local wandb_run_name="${BACKBONE_SLUG}_${run_suffix}"
    local job_id
    local dependency_args=()
    if [ -n "$DEPENDENCY_AFTERANY" ]; then
        dependency_args=(--dependency="afterany:${DEPENDENCY_AFTERANY}")
    fi
    job_id="$(sbatch --parsable \
        "${dependency_args[@]}" \
        --job-name="${wandb_run_name}" \
        scripts/paper/train_clip_b_intervention.slurm \
        "$registry_id" "$SEED")"
    local output_dir="/output/results/paper/interventions/${BACKBONE_SLUG}/${registry_id}/r${IMAGE_RESOLUTION}/${DATASET}/s${SEED}/${job_id}"
    append_manifest "$job_id" "$registry_id" "$wandb_run_name" "$output_dir"
    printf "%-30s %s\n" "$registry_id" "$job_id"
    if [ "$CHAIN_JOBS" = "true" ]; then
        DEPENDENCY_AFTERANY="$job_id"
    fi
}

echo "Submitting CLIP-B paper intervention runs"
echo "Manifest: ${MANIFEST_PATH}"
printf "%-30s %s\n" "registry_id" "job_id"
printf "%-30s %s\n" "------------------------------" "------"

submit_intervention "B1" "B1-siglip-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_intervention "B2" "B2-hardneg-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_intervention "B4" "B4-aux-objcls-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_intervention "B5a_seg_spatial" "B5a-seg-spatial-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_intervention "B5b_seg_semantic" "B5b-seg-semantic-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_intervention "B5c_seg_continuous" "B5c-seg-continuous-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_intervention "B5d_multistream_gate" "B5d-gate-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_intervention "B5d_multistream_crossattn" "B5d-crossattn-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_intervention "B5d_multistream_concat" "B5d-concat-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_intervention "B5e_sam_skip" "B5e-sam-skip-r336_coco_s42"
sleep "$SUBMIT_SLEEP_SECONDS"
submit_intervention "BLIP_TEXT" "BLIP-TEXT-r336_coco_s42"

echo "Done. Manifest: ${MANIFEST_PATH}"

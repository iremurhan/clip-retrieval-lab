#!/bin/bash
# Resume training from an existing checkpoint and W&B run.
# Usage:
#   sbatch scripts/resume.slurm <RUN_NAME> <CONFIG> <WANDB_ID> <CKPT_PATH>
#     RUN_NAME   – used for Slurm job name / W&B run name suffix
#     CONFIG     – path to dataset config (e.g. configs/config_coco.yaml)
#     WANDB_ID   – existing WandB run id to resume
#     CKPT_PATH  – path to last_model.pth checkpoint to resume from
#
# Example:
#   sbatch scripts/resume.slurm \
#     resume_deneme \
#     configs/config_coco.yaml \
#     a1b2c3d4 \
#     /users/beyza.urhan/experiments/results/coco/21074/last_model.pth
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --container-image=biremurhan/image-text-contrast:b5
#SBATCH --container-mounts=/users/beyza.urhan/clip-retrieval-lab:/workspace,/users/beyza.urhan/experiments/datasets/coco:/workspace/datasets/coco,/users/beyza.urhan/experiments/datasets/coco/caption_datasets:/workspace/datasets/coco/caption_datasets,/users/beyza.urhan/experiments/datasets/flickr30k:/workspace/datasets/flickr30k,/users/beyza.urhan/experiments/results:/output/results,/users/beyza.urhan/experiments/env:/env
#SBATCH --output=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --error=/users/beyza.urhan/experiments/results/slurm_%j.out
#SBATCH --container-env=WANDB_API_KEY,WANDB_MODE,PYTHONPATH=/workspace

set -euo pipefail

echo "Checking mounts..."
echo "  /workspace: $(ls /workspace 2>/dev/null | head -3 || echo 'NOT MOUNTED')"
echo "  /output: $(ls /output 2>/dev/null | head -3 || echo 'NOT MOUNTED')"

if [ ! -d "/workspace" ] || [ ! -d "/output" ]; then
    echo "FATAL ERROR: Required mounts not found. Check container-mounts."
    exit 1
fi

EXP_NAME="${1:-}"
CONFIG_FILE="${2:-configs/config_coco.yaml}"
WANDB_ID="${3:-}"
RESUME_PATH="${4:-}"

if [ -z "$WANDB_ID" ] || [ -z "$RESUME_PATH" ]; then
    echo "ERROR: WANDB_ID and CKPT_PATH must be provided."
    echo "Usage: sbatch scripts/resume.slurm <RUN_NAME> <CONFIG> <WANDB_ID> <CKPT_PATH>"
    exit 1
fi

if [ ! -f "/workspace/$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$RESUME_PATH" ]; then
    echo "ERROR: Checkpoint file not found: $RESUME_PATH"
    exit 1
fi

DATASET=$(python3 -c "
import sys, yaml
base = yaml.safe_load(open('configs/config_base.yaml'))
override = yaml.safe_load(open('${CONFIG_FILE}'))
base.setdefault('data', {}).update(override.get('data', {}))
d = base['data'].get('dataset')
if not d:
    sys.exit('ERROR: data.dataset not set in config')
print(d)
") || exit 1
WANDB_PROJECT_AUTO="retrieval"

JOB_DIR="/output/results/${DATASET}/${SLURM_JOB_ID}"
CKPT_DIR="$JOB_DIR"
mkdir -p "$JOB_DIR"

echo "=========================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Experiment:    ${EXP_NAME:- (Unnamed)}"
echo "Dataset:       ${DATASET}"
echo "Job dir:       ${JOB_DIR} (slurm.out, resumed_training.log, last_model.pth, best_model.pth, wandb/)"
echo "Resuming from: $RESUME_PATH"
echo "W&B run id:    $WANDB_ID"
echo "=========================================="

export WANDB_DIR="$JOB_DIR"
if [ -f /env/wandb.env ]; then
    set -a; source /env/wandb.env; set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=online
    if [ -n "$EXP_NAME" ]; then
        export WANDB_NAME="${SLURM_JOB_ID}_${EXP_NAME}"
    else
        export WANDB_NAME="${SLURM_JOB_ID}"
    fi

    export WANDB_PROJECT="$WANDB_PROJECT_AUTO"
    echo "W&B Project auto-set: $WANDB_PROJECT"
else
    echo "W&B API Key not found. Switching to OFFLINE mode."
    export WANDB_MODE=offline
fi

echo "Starting resumed training..."
cd /workspace
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

python3 -u run.py \
    --config "$CONFIG_FILE" \
    --resume "$RESUME_PATH" \
    --override \
        debug.debug_mode=false \
        logging.checkpoint_dir="$CKPT_DIR" \
        logging.wandb_project="$WANDB_PROJECT_AUTO" \
        logging.wandb_id="$WANDB_ID" \
    > "$JOB_DIR/resumed_training.log" 2>&1

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Resumed training completed successfully!"
else
    echo "Resumed training failed with exit code: $TRAIN_EXIT_CODE"
fi
echo "Results saved to: $JOB_DIR"
echo "=========================================="

# Attempt to sync WandB logs before any cleanup.
echo "Syncing WandB logs before cleanup..."
SYNC_EXIT=1
if [ -d "$JOB_DIR/wandb" ]; then
    python3 -m wandb sync --sync-all "$JOB_DIR/wandb" > "$JOB_DIR/wandb_sync.log" 2>&1
    SYNC_EXIT=$?
fi

# Verify the run actually exists on the W&B server with history before
# trusting the sync exit code (it sometimes returns 0 with no data uploaded).
VERIFY_EXIT=1
if [ $SYNC_EXIT -eq 0 ] && [ -d "$JOB_DIR/wandb" ]; then
    python3 - "$JOB_DIR/wandb" "$WANDB_PROJECT" <<'PYEOF'
import glob, json, os, sys
wandb_dir, project = sys.argv[1], sys.argv[2]
metas = sorted(glob.glob(os.path.join(wandb_dir, "run-*", "files", "wandb-metadata.json")))
if not metas:
    print(f"VERIFY: no wandb-metadata.json found under {wandb_dir}", flush=True)
    sys.exit(2)
import wandb
api = wandb.Api()
ok = True
for meta_path in metas:
    run_dir = os.path.dirname(os.path.dirname(meta_path))
    run_id = os.path.basename(run_dir).split("-")[-1]
    entity = json.load(open(meta_path)).get("entity") or os.environ.get("WANDB_ENTITY")
    path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    try:
        run = api.run(path)
        steps = run.lastHistoryStep
        state = run.state
        if steps is None or steps < 0:
            print(f"VERIFY FAIL: {path} has no history (state={state})", flush=True)
            ok = False
        else:
            print(f"VERIFY OK:   {path} state={state} lastStep={steps}", flush=True)
    except Exception as e:
        print(f"VERIFY FAIL: cannot fetch {path}: {e}", flush=True)
        ok = False
sys.exit(0 if ok else 3)
PYEOF
    VERIFY_EXIT=$?
fi

if [ $VERIFY_EXIT -eq 0 ]; then
    echo "WandB server-side verification passed. Local cache can be safely removed."
    # If training succeeded, remove last_model.pth (keep only best_model.pth to save disk)
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        if [ -f "$JOB_DIR/last_model.pth" ]; then
            rm -f "$JOB_DIR/last_model.pth"
        fi
    fi
    # Remove local WandB cache
    if [ -d "$JOB_DIR/wandb" ]; then
        rm -rf "$JOB_DIR/wandb"
    fi
else
    echo "WARNING: WandB sync exit=$SYNC_EXIT verify exit=$VERIFY_EXIT. Keeping '$JOB_DIR/wandb' and 'last_model.pth' for manual recovery."
fi

SLURM_LOG="/users/beyza.urhan/experiments/results/slurm_${SLURM_JOB_ID}.out"
if [ -f "$SLURM_LOG" ]; then
    mv "$SLURM_LOG" "$JOB_DIR/slurm.out"
fi

exit $TRAIN_EXIT_CODE


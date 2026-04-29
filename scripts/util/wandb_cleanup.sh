#!/usr/bin/env bash
# =============================================================================
# wandb_cleanup.sh — Shared WandB sync, server-side verify, and cleanup.
# =============================================================================
# Extracted from the ~55-line block duplicated in train.slurm, resume.slurm,
# and ablation_train.slurm.
#
# Usage (source, then call):
#   source "$(dirname "$0")/../util/wandb_cleanup.sh"
#   wandb_sync_and_cleanup "$JOB_DIR" "$WANDB_PROJECT" "$TRAIN_EXIT_CODE"
# =============================================================================

wandb_sync_and_cleanup() {
    local JOB_DIR="$1"
    local WANDB_PROJECT="$2"
    local TRAIN_EXIT_CODE="$3"

    # Attempt to sync WandB logs before any cleanup.
    echo "Syncing WandB logs before cleanup..."
    local SYNC_EXIT=1
    if [ -d "$JOB_DIR/wandb" ]; then
        python3 -m wandb sync --sync-all "$JOB_DIR/wandb" > "$JOB_DIR/wandb_sync.log" 2>&1
        SYNC_EXIT=$?
    fi

    # Verify the run actually exists on the W&B server with history before
    # trusting the sync exit code (it sometimes returns 0 with no data uploaded).
    local VERIFY_EXIT=1
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
}

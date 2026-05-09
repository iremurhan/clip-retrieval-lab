"""
scripts/eval/discover_pending_sugarcrepe.py
--------------------------------------------
Scan a results tree for checkpoints that still need SugarCrepe evaluation.

Cross-references with a WandB runs_summary.csv (exported by export_wandb.py)
to determine:
  1. Which checkpoints have already been evaluated (summary/sugarcrepe/macro_avg
     is populated).
  2. The WandB run ID for each checkpoint (parsed from training.log in the
     same directory).

Outputs a CSV manifest consumed by run_sugarcrepe_local.py.

Results directory layout (on T7 external drive):
    {results_root}/{run_id}/{run_name}/best_model.pth
                                      /training.log

The WandB run ID (short hash like "3ofk2x76") is extracted from training.log
by matching the line: "wandb: setting up run <run_id>".

Usage:
    python scripts/eval/discover_pending_sugarcrepe.py \
        --results-root /Volumes/T7/Research/experiments/results \
        --runs-csv /Volumes/T7/Research/wandb/runs_summary.csv \
        --output sugarcrepe_manifest.csv
"""

import argparse
import csv
import logging
import os
import re
import sys

logger = logging.getLogger(__name__)

# Columns in the runs_summary.csv that indicate SugarCrepe was already done.
# Old eval logged "overall"; new eval logs "macro_avg".  Either suffices.
SUGARCREPE_DONE_COLUMNS = [
    "summary/sugarcrepe/macro_avg",
    "summary/sugarcrepe/overall",
]


def parse_wandb_run_id_from_log(training_log_path):
    """Extract the WandB run ID from a training.log file.

    Looks for the pattern: 'wandb: setting up run <run_id>'
    Falls back to: 'run-YYYYMMDD_HHMMSS-<run_id>' in the log path line.

    Returns the run_id string or None if not found.
    """
    if not os.path.isfile(training_log_path):
        return None

    # Only read the first 100 lines — WandB setup is always at the top
    patterns = [
        re.compile(r"wandb:\s+setting up run\s+(\S+)"),
        re.compile(r"run-\d{8}_\d{6}-(\w+)"),
    ]

    try:
        with open(training_log_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 100:
                    break
                for pattern in patterns:
                    m = pattern.search(line)
                    if m:
                        return m.group(1)
    except OSError as e:
        logger.warning(f"Could not read {training_log_path}: {e}")
    return None


def parse_wandb_run_name_from_log(training_log_path):
    """Extract the WandB run name from a training.log file.

    Looks for the pattern: 'wandb: Syncing run <run_name>'

    Returns the run_name string or None if not found.
    """
    if not os.path.isfile(training_log_path):
        return None

    pattern = re.compile(r"wandb:\s+Syncing run\s+(\S+)")

    try:
        with open(training_log_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 100:
                    break
                m = pattern.search(line)
                if m:
                    return m.group(1)
    except OSError as e:
        logger.warning(f"Could not read {training_log_path}: {e}")
    return None


def infer_dataset_from_run_name(run_name):
    """Infer dataset name from WandB run name like 'B0_coco_s456' → 'coco'."""
    if not run_name:
        return "unknown"
    parts = run_name.split("_")
    for part in parts:
        if part in ("coco", "flickr30k", "flickr"):
            return part
    return "unknown"


def load_completed_run_ids(runs_csv_path):
    """Load the set of WandB run IDs that already have SugarCrepe results.

    A run is considered completed if *either* ``summary/sugarcrepe/macro_avg``
    or ``summary/sugarcrepe/overall`` contains a numeric value.  This handles
    the column rename between old and new eval versions.

    Returns:
        completed: set of run IDs (short hashes) with SugarCrepe done.
        all_runs: dict mapping run_id → row dict.
    """
    completed = set()
    all_runs = {}

    if not os.path.isfile(runs_csv_path):
        logger.warning(f"Runs CSV not found: {runs_csv_path}")
        return completed, all_runs

    with open(runs_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = row.get("id", "").strip()
            if not run_id:
                continue
            all_runs[run_id] = row

            for col in SUGARCREPE_DONE_COLUMNS:
                value = row.get(col, "").strip()
                if value:
                    try:
                        float(value)
                        completed.add(run_id)
                        break  # one valid column is enough
                    except ValueError:
                        pass

    return completed, all_runs


def discover_checkpoints(results_root):
    """Walk the results tree and find all best_model.pth checkpoints.

    Expected layout: {results_root}/**/best_model.pth
    Each checkpoint dir may also contain training.log.

    Returns a list of dicts with:
        checkpoint_path, run_dir, wandb_run_id, wandb_run_name, dataset
    """
    entries = []

    for dirpath, dirnames, filenames in os.walk(results_root):
        if "best_model.pth" not in filenames:
            continue

        checkpoint_path = os.path.join(dirpath, "best_model.pth")
        training_log = os.path.join(dirpath, "training.log")

        wandb_run_id = parse_wandb_run_id_from_log(training_log)
        wandb_run_name = parse_wandb_run_name_from_log(training_log)
        dataset = infer_dataset_from_run_name(wandb_run_name)

        entries.append({
            "checkpoint_path": checkpoint_path,
            "run_dir": dirpath,
            "wandb_run_id": wandb_run_id or "",
            "wandb_run_name": wandb_run_name or os.path.basename(dirpath),
            "dataset": dataset,
        })

    # Sort by path for deterministic ordering
    entries.sort(key=lambda e: e["checkpoint_path"])
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Discover checkpoints pending SugarCrepe evaluation"
    )
    parser.add_argument(
        "--results-root", type=str, required=True,
        help="Root of the results directory tree"
    )
    parser.add_argument(
        "--runs-csv", type=str, required=True,
        help="Path to runs_summary.csv from export_wandb.py"
    )
    parser.add_argument(
        "--output", type=str, default="sugarcrepe_manifest.csv",
        help="Output manifest CSV path"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load already-completed WandB runs
    completed, all_runs = load_completed_run_ids(args.runs_csv)
    logger.info(
        f"Loaded {len(all_runs)} WandB runs from CSV, "
        f"{len(completed)} already have SugarCrepe results."
    )

    # Discover all checkpoints
    all_entries = discover_checkpoints(args.results_root)
    logger.info(f"Found {len(all_entries)} checkpoints in {args.results_root}")

    # Filter: keep only pending (not in completed set)
    pending = []
    skipped_done = 0
    for entry in all_entries:
        if entry["wandb_run_id"] in completed:
            skipped_done += 1
            continue
        pending.append(entry)

    logger.info(
        f"Skipped {skipped_done} already-evaluated checkpoints. "
        f"{len(pending)} pending."
    )

    # Write manifest
    fieldnames = [
        "checkpoint_path", "dataset", "wandb_run_name", "wandb_run_id",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pending)

    logger.info(f"Manifest written to: {args.output}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  Total pending entries:  {len(pending)}")

    # Breakdown by dataset
    datasets = {}
    for e in pending:
        ds = e["dataset"]
        datasets[ds] = datasets.get(ds, 0) + 1
    print(f"  Breakdown by dataset:")
    for ds, count in sorted(datasets.items()):
        print(f"    {ds:20s} {count}")

    # Breakdown by config/run pattern
    configs = {}
    for e in pending:
        # Extract the config part: e.g., B0 from B0_coco_s456
        name = e["wandb_run_name"]
        parts = name.split("_")
        config_key = parts[0] if parts else name
        configs[config_key] = configs.get(config_key, 0) + 1
    print(f"  Breakdown by config:")
    for cfg, count in sorted(configs.items()):
        print(f"    {cfg:20s} {count}")

    # Flag entries missing wandb_run_id
    missing_wandb = [e for e in pending if not e["wandb_run_id"]]
    if missing_wandb:
        print(f"\n  ⚠️  {len(missing_wandb)} entries MISSING wandb_run_id:")
        for e in missing_wandb:
            print(f"    - {e['wandb_run_name']}: {e['checkpoint_path']}")
        print("  These will be evaluated but results won't auto-log to WandB.")
    else:
        print(f"\n  ✓ All entries have a wandb_run_id.")

    print("=" * 60)


if __name__ == "__main__":
    main()

"""
scripts/eval/run_sugarcrepe_local.py
--------------------------------------
Batch-run SugarCrepe evaluations locally from a manifest CSV.

Reads the manifest produced by discover_pending_sugarcrepe.py and sequentially
invokes scripts/eval/eval_sugarcrepe.py for each entry via subprocess.

Designed for Apple Silicon Macs: auto-selects MPS via --device auto.

Usage:
    # Dry run (print commands without executing)
    python scripts/eval/run_sugarcrepe_local.py \
        --manifest sugarcrepe_manifest.csv \
        --data-dir /Volumes/T7/Research/experiments/datasets/sugarcrepe \
        --images-dir /Volumes/T7/Research/experiments/datasets/coco/val2017 \
        --limit 1 --dry-run

    # Run first entry only
    python scripts/eval/run_sugarcrepe_local.py \
        --manifest sugarcrepe_manifest.csv \
        --data-dir /Volumes/T7/Research/experiments/datasets/sugarcrepe \
        --images-dir /Volumes/T7/Research/experiments/datasets/coco/val2017 \
        --limit 1

    # Run all remaining (skip first entry, continue on errors, background)
    nohup python scripts/eval/run_sugarcrepe_local.py \
        --manifest sugarcrepe_manifest.csv \
        --data-dir /Volumes/T7/Research/experiments/datasets/sugarcrepe \
        --images-dir /Volumes/T7/Research/experiments/datasets/coco/val2017 \
        --start 1 --continue-on-error \
        > sugarcrepe_run.log 2>&1 &
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

# Resolve the eval script path relative to this file
EVAL_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "eval_sugarcrepe.py",
)


def load_manifest(manifest_path):
    """Load manifest CSV rows as list of dicts."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_command(entry, data_dir, images_dir, device):
    """Build the eval_sugarcrepe.py command for a manifest entry."""
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--checkpoint", entry["checkpoint_path"],
        "--data_dir", data_dir,
        "--images_dir", images_dir,
        "--device", device,
    ]

    wandb_run_id = entry.get("wandb_run_id", "").strip()
    if wandb_run_id:
        cmd.extend(["--wandb_run_id", wandb_run_id])

    return cmd


def format_duration(seconds):
    """Format seconds as HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run SugarCrepe evaluations from a manifest CSV"
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to sugarcrepe_manifest.csv"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to SugarCrepe JSON files"
    )
    parser.add_argument(
        "--images-dir", type=str, required=True,
        help="Path to COCO val2017 images"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["cuda", "mps", "cpu", "auto"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process first N entries (after --start offset)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Skip first N entries (0-indexed)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue to next entry if one fails"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    entries = load_manifest(args.manifest)
    total = len(entries)
    logger.info(f"Loaded {total} entries from {args.manifest}")

    # Apply start/limit
    entries = entries[args.start:]
    if args.limit is not None:
        entries = entries[:args.limit]

    logger.info(
        f"Processing {len(entries)} entries "
        f"(start={args.start}, limit={args.limit})"
    )

    if not entries:
        logger.info("No entries to process.")
        return

    succeeded = 0
    failed = 0
    skipped = 0
    batch_start = time.time()

    for i, entry in enumerate(entries):
        idx = args.start + i
        run_name = entry.get("wandb_run_name", "?")
        wandb_id = entry.get("wandb_run_id", "").strip()
        checkpoint = entry["checkpoint_path"]

        print(f"\n{'=' * 70}")
        print(f"  [{idx + 1}/{total}] {run_name}")
        print(f"  Checkpoint: {checkpoint}")
        print(f"  WandB ID:   {wandb_id or '(none — results will not log to WandB)'}")
        print(f"{'=' * 70}")

        if not os.path.isfile(checkpoint):
            logger.error(f"Checkpoint not found: {checkpoint}")
            skipped += 1
            if args.continue_on_error:
                continue
            else:
                sys.exit(1)

        cmd = build_command(entry, args.data_dir, args.images_dir, args.device)

        if args.dry_run:
            print(f"  [DRY RUN] {' '.join(cmd)}")
            continue

        entry_start = time.time()
        try:
            result = subprocess.run(
                cmd,
                check=True,
                # Stream output to parent stdout/stderr in real time
            )
            elapsed = time.time() - entry_start
            logger.info(
                f"✓ [{idx + 1}/{total}] {run_name} completed in "
                f"{format_duration(elapsed)}"
            )
            succeeded += 1
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - entry_start
            logger.error(
                f"✗ [{idx + 1}/{total}] {run_name} FAILED (exit code {e.returncode}) "
                f"after {format_duration(elapsed)}"
            )
            failed += 1
            if not args.continue_on_error:
                logger.error("Aborting. Use --continue-on-error to keep going.")
                sys.exit(1)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            break

    batch_elapsed = time.time() - batch_start

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  BATCH COMPLETE")
    print(f"  Total time:   {format_duration(batch_elapsed)}")
    print(f"  Succeeded:    {succeeded}")
    print(f"  Failed:       {failed}")
    print(f"  Skipped:      {skipped}")
    if args.dry_run:
        print(f"  (dry-run mode — no commands were executed)")
    print(f"{'=' * 70}")

    if not args.dry_run:
        print(
            "\n  ➡️  Next step: Re-export runs_summary.csv from WandB:\n"
            "     python export_wandb.py\n"
        )


if __name__ == "__main__":
    main()

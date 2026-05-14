"""
Batch-run cross-dataset OOD retrieval evaluation from sugarcrepe_manifest.csv.

Usage:
    python scripts/eval/run_ood_eval_local.py \
        --manifest sugarcrepe_manifest.csv \
        --device auto

    nohup python scripts/eval/run_ood_eval_local.py \
        --manifest sugarcrepe_manifest.csv \
        --device auto \
        --continue-on-error \
        > ood_eval.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = Path(__file__).resolve().with_name("eval_cross_dataset.py")
ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
DATASET_ALIASES = {"coco": "coco", "flickr": "flickr30k", "flickr30k": "flickr30k"}
OPPOSITE_DATASET = {"coco": "flickr30k", "flickr30k": "coco"}


def load_manifest(manifest_path: str) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_identity(entry: dict) -> tuple[str, str, int]:
    candidates = [
        entry.get("wandb_run_name", ""),
        Path(entry.get("checkpoint_path", "")).parent.name,
    ]
    pattern = re.compile(r"(?P<run>.+?)_(?P<dataset>coco|flickr30k|flickr)_s?(?P<seed>\d+)$")
    for candidate in candidates:
        match = pattern.match(candidate)
        if match:
            return (
                match.group("run"),
                DATASET_ALIASES[match.group("dataset")],
                int(match.group("seed")),
            )
    dataset = DATASET_ALIASES.get(entry.get("dataset", "").strip(), entry.get("dataset", "").strip())
    return Path(entry["checkpoint_path"]).parent.parent.name, dataset, -1


def output_json_path(cache_dir: Path, run_id: str, train_dataset: str, seed: int) -> Path:
    eval_dataset = OPPOSITE_DATASET[train_dataset]
    return cache_dir / f"{run_id.replace('/', '_')}_{train_dataset}_to_{eval_dataset}_s{seed}.json"


def build_command(entry: dict, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--checkpoint",
        entry["checkpoint_path"],
        "--data-root",
        args.data_root,
        "--cache-dir",
        args.cache_dir,
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.force:
        cmd.append("--force")
    run_name = entry.get("wandb_run_name", "").strip()
    if run_name:
        cmd.extend(["--wandb_run_name", run_name])
    train_dataset = entry.get("dataset", "").strip()
    if train_dataset:
        cmd.extend(["--train-dataset", train_dataset])
    wandb_run_id = entry.get("wandb_run_id", "").strip()
    if args.log_wandb and wandb_run_id:
        cmd.extend(["--log-wandb", "--wandb_run_id", wandb_run_id])
    if args.wandb_project:
        cmd.extend(["--wandb_project", args.wandb_project])
    return cmd


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run cross-dataset OOD retrieval evaluation")
    parser.add_argument("--manifest", type=str, default=str(PROJECT_ROOT / "sugarcrepe_manifest.csv"))
    parser.add_argument("--data-root", type=str, default="/Volumes/T7/Research/experiments/datasets")
    parser.add_argument("--cache-dir", type=str, default=str(ARTIFACT_ROOT / "cache" / "ood_eval"))
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    entries = load_manifest(args.manifest)
    total = len(entries)
    entries_to_run = entries[args.start :]
    if args.limit is not None:
        entries_to_run = entries_to_run[: args.limit]
    logger.info("Loaded %d entries; processing %d", total, len(entries_to_run))

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    skipped = 0
    batch_start = time.time()

    for offset, entry in enumerate(entries_to_run):
        idx = args.start + offset
        run_id, train_dataset, seed = parse_identity(entry)
        run_name = entry.get("wandb_run_name", "?")
        checkpoint = entry["checkpoint_path"]
        target = OPPOSITE_DATASET.get(train_dataset)

        print(f"\n{'=' * 72}")
        print(f"  [{idx + 1}/{total}] {run_name}")
        print(f"  Train:      {train_dataset}")
        print(f"  OOD eval:   {target}")
        print(f"  Checkpoint: {checkpoint}")
        print(f"{'=' * 72}")

        if target is None:
            logger.error("Unknown training dataset for %s: %s", run_name, train_dataset)
            failed += 1
            if args.continue_on_error:
                continue
            sys.exit(1)

        if not os.path.isfile(checkpoint):
            logger.error("Checkpoint not found: %s", checkpoint)
            skipped += 1
            if args.continue_on_error:
                continue
            sys.exit(1)

        cache_path = output_json_path(cache_dir, run_id, train_dataset, seed)
        if cache_path.exists() and not args.force:
            logger.info("Cache exists, skipping: %s", cache_path)
            skipped += 1
            continue

        cmd = build_command(entry, args)
        if args.dry_run:
            print("  [DRY RUN] " + " ".join(cmd))
            continue

        entry_start = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - entry_start
            logger.info("[%d/%d] %s completed in %s", idx + 1, total, run_name, format_duration(elapsed))
            succeeded += 1
        except subprocess.CalledProcessError as exc:
            elapsed = time.time() - entry_start
            logger.error(
                "[%d/%d] %s failed with exit code %d after %s",
                idx + 1,
                total,
                run_name,
                exc.returncode,
                format_duration(elapsed),
            )
            failed += 1
            if not args.continue_on_error:
                sys.exit(1)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            break

    elapsed = time.time() - batch_start
    print(f"\n{'=' * 72}")
    print("  BATCH COMPLETE")
    print(f"  Total time: {format_duration(elapsed)}")
    print(f"  Succeeded:  {succeeded}")
    print(f"  Failed:     {failed}")
    print(f"  Skipped:    {skipped}")
    if args.dry_run:
        print("  (dry-run mode, no commands executed)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

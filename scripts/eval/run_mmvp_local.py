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
EVAL_SCRIPT = Path(__file__).resolve().with_name("eval_mmvp_vlm.py")
DATASET_ALIASES = {"coco": "coco", "flickr": "flickr30k", "flickr30k": "flickr30k"}


def load_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_identity(entry: dict) -> tuple[str, str, int]:
    pattern = re.compile(r"(?P<run>.+?)_(?P<dataset>coco|flickr30k|flickr)_s?(?P<seed>\d+)$")
    for candidate in [entry.get("wandb_run_name", ""), Path(entry.get("checkpoint_path", "")).parent.name]:
        match = pattern.match(candidate)
        if match:
            return match.group("run"), DATASET_ALIASES[match.group("dataset")], int(match.group("seed"))
    dataset = DATASET_ALIASES.get(entry.get("dataset", "").strip(), entry.get("dataset", "").strip())
    return Path(entry["checkpoint_path"]).parent.parent.name, dataset, -1


def output_json_path(cache_dir: Path, run_id: str, dataset: str, seed: int) -> Path:
    return cache_dir / f"{run_id.replace('/', '_')}_{dataset}_s{seed}.json"


def build_command(entry: dict, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--checkpoint",
        entry["checkpoint_path"],
        "--data-dir",
        args.data_dir,
        "--cache-dir",
        args.cache_dir,
        "--device",
        args.device,
    ]
    if args.force:
        cmd.append("--force")
    run_name = entry.get("wandb_run_name", "").strip()
    if run_name:
        cmd.extend(["--wandb_run_name", run_name])
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
    parser = argparse.ArgumentParser(description="Batch-run MMVP-VLM evaluation")
    parser.add_argument("--manifest", type=str, default=str(PROJECT_ROOT / "sugarcrepe_manifest.csv"))
    parser.add_argument("--data-dir", type=str, default="/Volumes/T7/Research/experiments/datasets/mmvp_vlm")
    parser.add_argument("--cache-dir", type=str, default=str(PROJECT_ROOT / "results" / "cache" / "mmvp_vlm"))
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
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
    selected = entries[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded %d entries; processing %d", total, len(selected))

    succeeded = failed = skipped = 0
    batch_start = time.time()
    for offset, entry in enumerate(selected):
        idx = args.start + offset
        run_id, dataset, seed = parse_identity(entry)
        run_name = entry.get("wandb_run_name", "?")
        checkpoint = entry["checkpoint_path"]
        print(f"\n{'=' * 72}")
        print(f"  [{idx + 1}/{total}] {run_name}")
        print(f"  Checkpoint: {checkpoint}")
        print(f"{'=' * 72}")

        if not os.path.isfile(checkpoint):
            logger.error("Checkpoint not found: %s", checkpoint)
            skipped += 1
            if args.continue_on_error:
                continue
            sys.exit(1)
        target = output_json_path(cache_dir, run_id, dataset, seed)
        if target.exists() and not args.force:
            logger.info("Cache exists, skipping: %s", target)
            skipped += 1
            continue

        cmd = build_command(entry, args)
        if args.dry_run:
            print("  [DRY RUN] " + " ".join(cmd))
            continue
        start = time.time()
        try:
            subprocess.run(cmd, check=True)
            succeeded += 1
            logger.info("[%d/%d] %s completed in %s", idx + 1, total, run_name, format_duration(time.time() - start))
        except subprocess.CalledProcessError as exc:
            failed += 1
            logger.error("[%d/%d] %s failed with exit code %d", idx + 1, total, run_name, exc.returncode)
            if not args.continue_on_error:
                sys.exit(1)
    print(f"\n{'=' * 72}")
    print("  BATCH COMPLETE")
    print(f"  Total time: {format_duration(time.time() - batch_start)}")
    print(f"  Succeeded:  {succeeded}")
    print(f"  Failed:     {failed}")
    print(f"  Skipped:    {skipped}")
    if args.dry_run:
        print("  (dry-run mode, no commands executed)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

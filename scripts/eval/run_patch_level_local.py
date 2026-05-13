"""
Batch-run patch-level SugarCrepe diagnostics from sugarcrepe_manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = Path(__file__).resolve().with_name("diagnostic_patch_level.py")
DEFAULT_CACHE_DIR = PROJECT_ROOT / "results" / "cache" / "patch_level"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results" / "cache" / "patch_level_results.csv"
EXCLUDE = {"B0v2", "B0plus_fixed"}


def load_manifest(manifest_path: str) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_identity(entry: dict) -> tuple[str, str, int]:
    pattern = re.compile(r"(?P<run>.+?)_(?P<dataset>coco|flickr30k|flickr)_s?(?P<seed>\d+)$")
    for candidate in [entry.get("wandb_run_name", ""), Path(entry.get("checkpoint_path", "")).parent.name]:
        match = pattern.match(candidate)
        if match:
            dataset = "flickr30k" if match.group("dataset") == "flickr" else match.group("dataset")
            return match.group("run"), dataset, int(match.group("seed"))
    return Path(entry["checkpoint_path"]).parent.parent.name, entry.get("dataset", "").strip(), -1


def output_json_path(cache_dir: Path, run_id: str, dataset: str, seed: int) -> Path:
    return cache_dir / f"{dataset}_{run_id.replace('/', '_')}_s{seed}.json"


def build_command(entry: dict, data_dir: str, images_dir: str, output: Path, device: str, log_wandb: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--checkpoint",
        entry["checkpoint_path"],
        "--data_dir",
        data_dir,
        "--images_dir",
        images_dir,
        "--output",
        str(output),
        "--device",
        device,
    ]
    wandb_run_id = entry.get("wandb_run_id", "").strip()
    if log_wandb and wandb_run_id:
        cmd.extend(["--wandb_run_id", wandb_run_id])
    return cmd


def aggregate_jsons(cache_dir: Path, results_csv: Path) -> list[dict]:
    rows = []
    for path in sorted(cache_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for subcat, metrics in payload["results"].items():
            if subcat == "macro_avg":
                cls_acc = metrics["cls"]
                patch_acc = metrics["patch_max"]
            else:
                cls_acc = metrics["cls_accuracy"]
                patch_acc = metrics["patch_max_accuracy"]
            rows.append(
                {
                    "run_id": payload["run_id"],
                    "dataset": payload["dataset"],
                    "seed": payload["seed"],
                    "subcategory": subcat,
                    "cls_accuracy": cls_acc,
                    "patch_max_accuracy": patch_acc,
                    "delta": metrics["delta"],
                }
            )

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_id",
            "dataset",
            "seed",
            "subcategory",
            "cls_accuracy",
            "patch_max_accuracy",
            "delta",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run patch-level SugarCrepe diagnostic")
    parser.add_argument("--manifest", type=str, default=str(PROJECT_ROOT / "sugarcrepe_manifest.csv"))
    parser.add_argument("--data_dir", type=str, default="/Volumes/T7/Research/experiments/datasets/sugarcrepe")
    parser.add_argument("--images_dir", type=str, default="/Volumes/T7/Research/experiments/datasets/coco/val2017")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--results-csv", type=str, default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    entries = []
    for entry in load_manifest(args.manifest):
        run_id, dataset, _seed = parse_identity(entry)
        if run_id in EXCLUDE:
            continue
        if dataset not in {"coco", "flickr30k"}:
            logger.warning("Skipping entry with unknown dataset: %s", entry)
            continue
        entries.append(entry)

    total = len(entries)
    entries_to_run = entries[args.start :]
    if args.limit is not None:
        entries_to_run = entries_to_run[: args.limit]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(args.results_csv)
    succeeded = failed = skipped = 0
    batch_start = time.time()

    logger.info("Loaded %d entries, processing %d.", total, len(entries_to_run))
    for offset, entry in enumerate(entries_to_run):
        idx = args.start + offset
        run_id, dataset, seed = parse_identity(entry)
        output = output_json_path(cache_dir, run_id, dataset, seed)

        print(f"\n{'=' * 72}")
        print(f"  [{idx + 1}/{total}] {run_id} {dataset} seed={seed}")
        print(f"  Checkpoint: {entry['checkpoint_path']}")
        print(f"  Output:     {output}")
        print(f"{'=' * 72}")

        if output.exists() and not args.force:
            logger.info("Skipping existing result: %s", output)
            skipped += 1
            continue
        if not os.path.isfile(entry["checkpoint_path"]):
            logger.error("Checkpoint not found: %s", entry["checkpoint_path"])
            failed += 1
            if args.continue_on_error:
                continue
            sys.exit(1)

        cmd = build_command(entry, args.data_dir, args.images_dir, output, args.device, args.log_wandb)
        if args.dry_run:
            print("  [DRY RUN] " + " ".join(cmd))
            continue

        start = time.time()
        try:
            subprocess.run(cmd, check=True)
            logger.info("Completed in %s", format_duration(time.time() - start))
            succeeded += 1
        except subprocess.CalledProcessError as exc:
            logger.error("FAILED with exit code %s after %s", exc.returncode, format_duration(time.time() - start))
            failed += 1
            if not args.continue_on_error:
                sys.exit(exc.returncode)

    if args.dry_run:
        print(f"\n{'=' * 72}")
        print("  DRY RUN COMPLETE")
        print("  No diagnostics were executed and no aggregate CSV was written.")
        print(f"{'=' * 72}")
        return

    rows = aggregate_jsons(cache_dir, results_csv)
    if rows and any(row[col] in ("", None) for row in rows for col in ("cls_accuracy", "patch_max_accuracy", "delta")):
        raise RuntimeError("Aggregate CSV contains missing patch-level values.")

    print(f"\n{'=' * 72}")
    print("  PATCH-LEVEL BATCH COMPLETE")
    print(f"  Total time: {format_duration(time.time() - batch_start)}")
    print(f"  Succeeded:  {succeeded}")
    print(f"  Skipped:    {skipped}")
    print(f"  Failed:     {failed}")
    print(f"  CSV:        {results_csv}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

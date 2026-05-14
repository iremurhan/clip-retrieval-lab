"""
Batch-run alignment/uniformity diagnostics from sugarcrepe_manifest.csv.

Outputs per-checkpoint JSON files under the external artifact cache and an
aggregate CSV alongside them.
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
EVAL_SCRIPT = Path(__file__).resolve().with_name("diagnostic_alignment_uniformity.py")
ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
DEFAULT_CACHE_DIR = ARTIFACT_ROOT / "cache" / "alignment_uniformity"
DEFAULT_RESULTS_CSV = ARTIFACT_ROOT / "cache" / "alignment_uniformity_results.csv"
EXCLUDE = {"B0v2", "B0plus_fixed"}


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
            dataset = "flickr30k" if match.group("dataset") == "flickr" else match.group("dataset")
            return match.group("run"), dataset, int(match.group("seed"))
    dataset = entry.get("dataset", "").strip()
    return Path(entry["checkpoint_path"]).parent.parent.name, dataset, -1


def output_json_path(cache_dir: Path, run_id: str, dataset: str, seed: int) -> Path:
    safe_run = run_id.replace("/", "_")
    return cache_dir / f"{dataset}_{safe_run}_s{seed}.json"


def build_command(entry: dict, data_root: str, output: Path, device: str, batch_size: int, num_workers: int, uniformity_pairs: int) -> list[str]:
    run_id, dataset, _seed = parse_identity(entry)
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        "--checkpoint",
        entry["checkpoint_path"],
        "--dataset",
        dataset,
        "--data-root",
        data_root,
        "--output",
        str(output),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--uniformity-pairs",
        str(uniformity_pairs),
    ]


def aggregate_jsons(cache_dir: Path, results_csv: Path) -> list[dict]:
    rows = []
    for path in sorted(cache_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "run_id": data["run_id"],
                "dataset": data["dataset"],
                "seed": data["seed"],
                "alignment": data["alignment"],
                "uniformity_img": data["uniformity_img"],
                "uniformity_txt": data["uniformity_txt"],
                "uniformity_mean": data["uniformity_mean"],
            }
        )

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "run_id",
            "dataset",
            "seed",
            "alignment",
            "uniformity_img",
            "uniformity_txt",
            "uniformity_mean",
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
    parser = argparse.ArgumentParser(description="Batch-run alignment/uniformity diagnostics")
    parser.add_argument("--manifest", type=str, default=str(PROJECT_ROOT / "sugarcrepe_manifest.csv"))
    parser.add_argument("--data-root", type=str, default="/Volumes/T7/Research/experiments/datasets")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--results-csv", type=str, default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--uniformity-pairs", type=int, default=2_000_000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    entries = load_manifest(args.manifest)
    filtered = []
    for entry in entries:
        run_id, dataset, seed = parse_identity(entry)
        if run_id in EXCLUDE:
            continue
        if dataset not in {"coco", "flickr30k"}:
            logger.warning("Skipping entry with unknown dataset: %s", entry)
            continue
        filtered.append(entry)

    total = len(filtered)
    entries_to_run = filtered[args.start :]
    if args.limit is not None:
        entries_to_run = entries_to_run[: args.limit]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(args.results_csv)

    logger.info("Loaded %d entries, processing %d.", total, len(entries_to_run))
    succeeded = failed = skipped = 0
    batch_start = time.time()

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

        cmd = build_command(
            entry,
            data_root=args.data_root,
            output=output,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            uniformity_pairs=args.uniformity_pairs,
        )
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
    missing = [
        row
        for row in rows
        if any(row[col] in ("", None) for col in ("alignment", "uniformity_img", "uniformity_txt", "uniformity_mean"))
    ]
    if missing:
        raise RuntimeError(f"Aggregate CSV contains missing diagnostic values: {missing[:3]}")

    print(f"\n{'=' * 72}")
    print("  ALIGNMENT/UNIFORMITY BATCH COMPLETE")
    print(f"  Total time: {format_duration(time.time() - batch_start)}")
    print(f"  Succeeded:  {succeeded}")
    print(f"  Skipped:    {skipped}")
    print(f"  Failed:     {failed}")
    print(f"  CSV:        {results_csv}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plan missing experiment/dataset/seed combinations from local W&B caches.

This is a lightweight fallback when the W&B API export is unavailable or queued.
It reads local W&B debug logs and training logs under the shared results root.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


RUN_ORDER = [
    "B5a_seg_spatial",
    "B5b_seg_semantic",
    "B5c_seg_continuous",
    "B5d_multistream_gate",
    "B5d_multistream_crossattn",
    "B5d_multistream_concat",
    "B5e_sam_skip",
    "B0plus_fixed",
    "B0_projonly",
    "B0_uf1",
    "B0_uf2",
    "B0_uf3",
    "B0_uf5",
    "B0_uf6",
    "B0_uf7",
]
DATASET_ORDER = ["coco", "flickr30k"]
SEED_ORDER = [42, 123, 456]


def collect_present(results_root: Path) -> set[tuple[str, str, int]]:
    present: set[tuple[str, str, int]] = set()

    config_re = re.compile(r"config: (\{.*?\})(?:\n|$)")
    for path in results_root.rglob("debug.log"):
        text = path.read_text(errors="ignore")
        for match in config_re.finditer(text):
            try:
                config = ast.literal_eval(match.group(1))
            except Exception:
                continue
            run_id = config.get("run_id")
            dataset = config.get("dataset")
            seed = config.get("seed")
            if run_id and dataset and seed is not None:
                present.add((str(run_id), str(dataset), int(seed)))

    run_re = re.compile(r"WandB run: ([A-Za-z0-9_]+)_(coco|flickr30k)_s(\d+)")
    for log_name in ("training.log", "output.log"):
        for path in results_root.rglob(log_name):
            text = path.read_text(errors="ignore")
            for run_id, dataset, seed in run_re.findall(text):
                present.add((run_id, dataset, int(seed)))

    return present


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/users/beyza.urhan/experiments/results"),
    )
    parser.add_argument(
        "--include-queued-lowuf-s42",
        action="store_true",
        help="Treat previously queued B0 projection/uf1/uf2/uf3 seed-42 jobs as present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    present = collect_present(args.results_root)

    if args.include_queued_lowuf_s42:
        for dataset in DATASET_ORDER:
            for run_id in ("B0_projonly", "B0_uf1", "B0_uf2", "B0_uf3"):
                present.add((run_id, dataset, 42))

    print("run_id,dataset,missing_seeds")
    for run_id in RUN_ORDER:
        for dataset in DATASET_ORDER:
            missing = [
                seed for seed in SEED_ORDER
                if (run_id, dataset, seed) not in present
            ]
            if missing:
                print(f"{run_id},{dataset},{' '.join(str(s) for s in missing)}")


if __name__ == "__main__":
    main()

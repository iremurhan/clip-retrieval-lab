#!/usr/bin/env python3
"""Submit missing experiment/dataset/seed groups as chained Slurm jobs."""

from __future__ import annotations

import argparse
import subprocess

from plan_missing_experiments import (
    DATASET_ORDER,
    RUN_ORDER,
    SEED_ORDER,
    collect_present,
)


def mem_per_gpu(dataset: str) -> str:
    return "50G" if dataset == "flickr30k" else "60G"


def missing_groups(include_queued_lowuf_s42: bool) -> list[tuple[str, str, list[int]]]:
    present = collect_present(args.results_root)
    if include_queued_lowuf_s42:
        for dataset in DATASET_ORDER:
            for run_id in ("B0_projonly", "B0_uf1", "B0_uf2", "B0_uf3"):
                present.add((run_id, dataset, 42))

    groups: list[tuple[str, str, list[int]]] = []
    for run_id in RUN_ORDER:
        for dataset in DATASET_ORDER:
            seeds = [
                seed
                for seed in SEED_ORDER
                if (run_id, dataset, seed) not in present
            ]
            if seeds:
                groups.append((run_id, dataset, seeds))
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="/users/beyza.urhan/experiments/results",
        type=__import__("pathlib").Path,
    )
    parser.add_argument(
        "--initial-dependency",
        default="afterany:26755:26758",
        help="Dependency for the first queued job.",
    )
    parser.add_argument(
        "--include-queued-lowuf-s42",
        action="store_true",
        help="Do not resubmit B0 projection/uf1/uf2/uf3 seed-42 jobs already queued earlier.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually call sbatch. Without this flag, print the plan only.",
    )
    return parser.parse_args()


args = parse_args()


def main() -> None:
    groups = missing_groups(args.include_queued_lowuf_s42)
    dependency = args.initial_dependency
    print("run_id,dataset,seeds,dependency,job_id")

    for run_id, dataset, seeds in groups:
        seed_label = "-".join(str(seed) for seed in seeds)
        job_name = f"{run_id}_{dataset}_missing"
        cmd = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            f"--dependency={dependency}",
            f"--mem-per-gpu={mem_per_gpu(dataset)}",
            "scripts/train/train_seed_sequence.slurm",
            dataset,
            run_id,
            *[str(seed) for seed in seeds],
        ]

        if args.submit:
            job_id = subprocess.check_output(cmd, text=True).strip()
        else:
            job_id = "DRY_RUN"

        print(f"{run_id},{dataset},{seed_label},{dependency},{job_id}")
        dependency = f"afterany:{job_id}" if args.submit else f"afterany:{job_name}"


if __name__ == "__main__":
    main()

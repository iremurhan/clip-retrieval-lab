#!/usr/bin/env python3
"""Submit missing experiment/dataset/seed work as chained Slurm jobs.

B5 runs are submitted one seed per allocation because the multi-stream and
segmentation variants can exceed the 7-day wall time when seeds are grouped.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from plan_missing_experiments import (
    DATASET_ORDER,
    RUN_ORDER,
    SEED_ORDER,
    collect_present,
)


def is_b5_run(run_id: str) -> bool:
    return run_id.startswith("B5")


def config_for_dataset(dataset: str) -> str:
    return "configs/config_flickr30k.yaml" if dataset == "flickr30k" else "configs/config_coco.yaml"


def mem_per_gpu(run_id: str, dataset: str) -> str:
    if is_b5_run(run_id):
        return "64G"
    return "50G" if dataset == "flickr30k" else "60G"


def missing_groups(results_root: Path, include_queued_lowuf_s42: bool) -> list[tuple[str, str, list[int]]]:
    present = collect_present(results_root)
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


def submit_or_dry_run(cmd: list[str], job_name: str) -> tuple[str, str]:
    if args.submit:
        job_id = subprocess.check_output(cmd, text=True).strip()
        return job_id, job_id
    return "DRY_RUN", job_name


def main() -> None:
    groups = missing_groups(args.results_root, args.include_queued_lowuf_s42)
    dependency = args.initial_dependency
    print("run_id,dataset,seeds,dependency,job_id")

    for run_id, dataset, seeds in groups:
        if is_b5_run(run_id):
            for seed in seeds:
                job_name = f"{run_id}_{dataset}_s{seed}_missing"
                cmd = [
                    "sbatch",
                    "--parsable",
                    f"--job-name={job_name}",
                    f"--dependency={dependency}",
                    f"--mem-per-gpu={mem_per_gpu(run_id, dataset)}",
                    "scripts/train/train.slurm",
                    run_id,
                    config_for_dataset(dataset),
                    "--seed",
                    str(seed),
                ]

                job_id, dependency_id = submit_or_dry_run(cmd, job_name)
                print(f"{run_id},{dataset},{seed},{dependency},{job_id}")
                dependency = f"afterany:{dependency_id}"
            continue

        seed_label = "-".join(str(seed) for seed in seeds)
        job_name = f"{run_id}_{dataset}_missing"
        cmd = [
            "sbatch",
            "--parsable",
            f"--job-name={job_name}",
            f"--dependency={dependency}",
            f"--mem-per-gpu={mem_per_gpu(run_id, dataset)}",
            "scripts/train/train_seed_sequence.slurm",
            dataset,
            run_id,
            *[str(seed) for seed in seeds],
        ]

        job_id, dependency_id = submit_or_dry_run(cmd, job_name)
        print(f"{run_id},{dataset},{seed_label},{dependency},{job_id}")
        dependency = f"afterany:{dependency_id}"


if __name__ == "__main__":
    main()

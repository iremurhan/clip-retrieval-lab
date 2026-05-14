#!/usr/bin/env python3
"""Requeue pending B5 work as single-seed Slurm jobs.

This avoids losing second COCO seeds to the 7-day Slurm wall-time limit.
"""

from __future__ import annotations

import subprocess


def submit(run_id: str, dataset: str, seed: int, dependency: str) -> str:
    dataset_label = "flickr" if dataset == "flickr30k" else dataset
    config = "configs/config_flickr30k.yaml" if dataset == "flickr30k" else "configs/config_coco.yaml"
    short = run_id.replace("B5d_multistream_", "B5d_").replace("B5", "B5")
    job_name = f"{short}_{dataset_label}_s{seed}_single"
    cmd = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--dependency={dependency}",
        "scripts/train/train.slurm",
        run_id,
        config,
        "--seed",
        str(seed),
    ]
    return subprocess.check_output(cmd, text=True).strip()


branches = [
    (
        "b5e_flickr",
        "afterany:26962",
        [
            ("B5e_sam_skip", "flickr30k", 123),
            ("B5e_sam_skip", "flickr30k", 456),
        ],
    ),
    (
        "b5_remaining_coco",
        "afterany:26961:26963",
        [
            ("B5a_seg_spatial", "coco", 456),
            ("B5b_seg_semantic", "coco", 456),
            ("B5c_seg_continuous", "coco", 42),
            ("B5c_seg_continuous", "coco", 456),
            ("B5d_multistream_gate", "coco", 42),
            ("B5d_multistream_gate", "coco", 456),
            ("B5d_multistream_crossattn", "coco", 42),
            ("B5d_multistream_crossattn", "coco", 456),
            ("B5d_multistream_concat", "coco", 42),
            ("B5d_multistream_concat", "coco", 456),
        ],
    ),
    (
        "b5_remaining_flickr",
        "afterany:26992",
        [
            ("B5b_seg_semantic", "flickr30k", 42),
            ("B5b_seg_semantic", "flickr30k", 456),
            ("B5c_seg_continuous", "flickr30k", 42),
            ("B5c_seg_continuous", "flickr30k", 456),
            ("B5d_multistream_gate", "flickr30k", 42),
            ("B5d_multistream_gate", "flickr30k", 456),
            ("B5d_multistream_crossattn", "flickr30k", 42),
            ("B5d_multistream_crossattn", "flickr30k", 456),
            ("B5d_multistream_concat", "flickr30k", 42),
            ("B5d_multistream_concat", "flickr30k", 456),
        ],
    ),
]


def main() -> None:
    print("branch,run_id,dataset,seed,dependency,job_id")
    branch_last: dict[str, str] = {}
    for branch_name, initial_dependency, jobs in branches:
        dependency = initial_dependency
        if branch_name == "b5_remaining_flickr":
            dependency = f"afterany:{branch_last['b5e_flickr']}"
        for run_id, dataset, seed in jobs:
            job_id = submit(run_id, dataset, seed, dependency)
            print(f"{branch_name},{run_id},{dataset},{seed},{dependency},{job_id}")
            dependency = f"afterany:{job_id}"
        branch_last[branch_name] = (
            dependency[len("afterany:"):]
            if dependency.startswith("afterany:")
            else dependency
        )


if __name__ == "__main__":
    main()

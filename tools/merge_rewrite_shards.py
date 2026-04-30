"""
tools/merge_rewrite_shards.py
-----------------------------
Merge sharded paraphrase outputs (from a SLURM array run of
legacy/paraphrase_generation/generate_rewrites.py) into a single JSON.

Each shard writes to <output>.shard{i}of{N}.json, where N is the SLURM array
size and i is the array task index. Shards are disjoint by construction
(stride sharding: i % num_shards == shard_id), so a sentid collision between
shards indicates a sharding bug and aborts the merge.

Failure JSONLs (.shard{i}of{N}.failed.jsonl) are concatenated into the
unsharded .failed.jsonl for downstream retry-pass logic.

Usage:
    python tools/merge_rewrite_shards.py \
        --output datasets/coco/caption_rewrites_meta-llama-3-8b-instruct.json \
        --num_shards 4
"""

import argparse
import json
import os
import sys


def shard_path_for(output_path: str, shard_id: int, num_shards: int) -> str:
    base, ext = os.path.splitext(output_path)
    return f"{base}.shard{shard_id}of{num_shards}{ext}"


def shard_failed_path_for(output_path: str, shard_id: int, num_shards: int) -> str:
    # Failed-log derivation matches generate_rewrites._failed_path_for():
    # strip the trailing .json, then append .failed.jsonl.
    sp = shard_path_for(output_path, shard_id, num_shards)
    base, _ = os.path.splitext(sp)
    return f"{base}.failed.jsonl"


def merge(output_path: str, num_shards: int) -> None:
    if num_shards < 2:
        sys.exit(f"--num_shards must be >= 2 (got {num_shards})")

    merged: dict = {}
    seen: set = set()
    for i in range(num_shards):
        sp = shard_path_for(output_path, i, num_shards)
        if not os.path.exists(sp):
            sys.exit(f"ERROR: missing shard file {sp}")
        with open(sp, "r") as f:
            shard = json.load(f)
        # Detect sentid collisions across shards (would indicate a sharding bug).
        collisions = seen & set(shard.keys())
        if collisions:
            sample = list(collisions)[:5]
            sys.exit(
                f"ERROR: shard {i} ({sp}) has {len(collisions)} sentid "
                f"collisions with prior shards (sample: {sample}). Aborting."
            )
        merged.update(shard)
        seen.update(shard.keys())
        print(f"  shard {i}/{num_shards}: {len(shard)} sentids ({sp})")

    # Atomic write
    tmp = output_path + ".tmp"
    os.makedirs(os.path.dirname(tmp) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(merged, f)
    os.replace(tmp, output_path)
    print(f"\nMerged {len(merged)} sentids -> {output_path}")

    # Concatenate per-shard .failed.jsonl into the unsharded one (if any exist).
    base, _ = os.path.splitext(output_path)
    out_failed = f"{base}.failed.jsonl"
    shard_failed_paths = [
        shard_failed_path_for(output_path, i, num_shards)
        for i in range(num_shards)
    ]
    existing_failed = [p for p in shard_failed_paths if os.path.exists(p)]
    if existing_failed:
        n_failed = 0
        with open(out_failed, "w") as out:
            for p in existing_failed:
                with open(p, "r") as f:
                    for line in f:
                        out.write(line)
                        n_failed += 1
        print(f"Concatenated {n_failed} failure records -> {out_failed}")
    else:
        print("No per-shard .failed.jsonl files found; skipping failure-log merge.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--output", type=str, required=True,
        help="Final merged output path (the unsharded path the model would "
             "have written if num_shards were 1).",
    )
    parser.add_argument(
        "--num_shards", type=int, required=True,
        help="Total number of shards in the SLURM array.",
    )
    args = parser.parse_args()
    merge(args.output, args.num_shards)


if __name__ == "__main__":
    main()

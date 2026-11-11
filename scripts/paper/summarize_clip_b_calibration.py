#!/usr/bin/env python3
"""
Summarize CLIP-B paper baseline calibration results.

Reads zero-shot JSON outputs and trained-run validation_metrics.jsonl files from
the paper output tree, then selects the trained unfreezing depth using
validation bidirectional R@1 mean only.
"""

import argparse
import json
import math
from pathlib import Path


ZERO_ORDER = {"zero_shot_336": 0}


def latest_json(paths):
    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def read_zero(path):
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics", {})
    return {
        "status": payload.get("status", "completed"),
        "zero_shot": True,
        "resolution_role": payload.get("wandb_config", {}).get("resolution_role"),
        "registry_id": payload.get("registry_id"),
        "image_resolution": payload.get("image_resolution"),
        "native_image_resolution": payload.get("native_image_resolution"),
        "positional_interpolation": payload.get("positional_interpolation"),
        "patch_grid": f"{payload.get('patch_grid_height')}x{payload.get('patch_grid_width')}",
        "unfrozen_blocks": 0,
        "trainable_parameter_count": 0,
        "selected_checkpoint_epoch": None,
        "i2t_r1": metrics.get("r1_i2t"),
        "t2i_r1": metrics.get("r1_t2i"),
        "bidirectional_r1_mean": metrics.get("bidirectional_r1_mean"),
        "source": str(path),
    }


def read_validation_history(path):
    best = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metrics = record.get("metrics", {})
        score = metrics.get("val/bidirectional_r1_mean", metrics.get("val_bidirectional_r1_mean"))
        if score is None:
            continue
        if best is None or score > best["score"]:
            best = {"epoch": record.get("epoch"), "metrics": metrics, "score": score}
    return best


def read_trained(run_dir):
    history_path = run_dir / "validation_metrics.jsonl"
    if not history_path.exists():
        return {
            "status": "missing_validation_metrics",
            "zero_shot": False,
            "registry_id": run_dir.parents[3].name if len(run_dir.parents) > 3 else run_dir.name,
            "source": str(run_dir),
        }
    best = read_validation_history(history_path)
    if best is None:
        return {
            "status": "no_bidirectional_metric",
            "zero_shot": False,
            "registry_id": run_dir.parents[3].name if len(run_dir.parents) > 3 else run_dir.name,
            "source": str(history_path),
        }

    parts = run_dir.parts
    registry_id = None
    image_resolution = None
    seed = None
    for idx, part in enumerate(parts):
        if part.startswith("minimal_"):
            registry_id = part
        if part.startswith("r") and part[1:].isdigit():
            image_resolution = int(part[1:])
        if part.startswith("s") and part[1:].isdigit():
            seed = int(part[1:])

    metadata_path = run_dir / "paper_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metrics = best["metrics"]
    return {
        "status": "completed",
        "zero_shot": False,
        "resolution_role": metadata.get("resolution_role", "trained_non_native_336"),
        "registry_id": registry_id,
        "image_resolution": image_resolution,
        "native_image_resolution": metadata.get("native_image_resolution", 224),
        "positional_interpolation": metadata.get("positional_interpolation", False),
        "patch_grid": metadata.get("patch_grid"),
        "unfrozen_blocks": metadata.get("unfrozen_vision_blocks"),
        "trainable_parameter_count": metadata.get("trainable_parameter_count"),
        "selected_checkpoint_epoch": best["epoch"],
        "val_i2t_r1": metrics.get("val/r1_i2t"),
        "val_t2i_r1": metrics.get("val/r1_t2i"),
        "val_bidirectional_r1_mean": metrics.get("val/bidirectional_r1_mean", metrics.get("val_bidirectional_r1_mean")),
        "i2t_r1": metrics.get("val/r1_i2t"),
        "t2i_r1": metrics.get("val/r1_t2i"),
        "bidirectional_r1_mean": metrics.get("val/bidirectional_r1_mean", metrics.get("val_bidirectional_r1_mean")),
        "seed": seed,
        "source": str(history_path),
    }


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    headers = [
        "registry_id", "status", "role", "res", "native", "interp", "grid",
        "uf", "trainable", "epoch", "i2t_r1", "t2i_r1", "bidir_r1",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join([
            fmt(row.get("registry_id")),
            fmt(row.get("status")),
            fmt(row.get("resolution_role")),
            fmt(row.get("image_resolution")),
            fmt(row.get("native_image_resolution")),
            fmt(row.get("positional_interpolation")),
            fmt(row.get("patch_grid")),
            fmt(row.get("unfrozen_blocks")),
            fmt(row.get("trainable_parameter_count")),
            fmt(row.get("selected_checkpoint_epoch")),
            fmt(row.get("i2t_r1")),
            fmt(row.get("t2i_r1")),
            fmt(row.get("bidirectional_r1_mean")),
        ]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/users/beyza.urhan/experiments/results/paper/baseline-calibration")
    args = p.parse_args()
    root = Path(args.root)

    zero_rows = []
    for run_id in ("zero_shot_336",):
        path = latest_json(root.glob(f"clipb16/{run_id}/r*/coco/s42/*/zero_shot.json"))
        if path:
            zero_rows.append(read_zero(path))
        else:
            zero_rows.append({"status": "missing", "zero_shot": True, "registry_id": run_id})
    zero_rows.sort(key=lambda row: ZERO_ORDER.get(row.get("registry_id"), 99))

    trained_rows = []
    for run_id in ["minimal_projonly"] + [f"minimal_uf{i}" for i in range(1, 8)]:
        run_dirs = sorted(root.glob(f"clipb16/{run_id}/r336/coco/s42/*"), key=lambda p: p.stat().st_mtime, reverse=True)
        trained_rows.append(read_trained(run_dirs[0]) if run_dirs else {
            "status": "missing",
            "zero_shot": False,
            "registry_id": run_id,
        })

    print("Single-seed CLIP-B baseline calibration (seed=42).")
    print_table("Zero-Shot Test Reference", zero_rows)

    completed = [
        row for row in trained_rows
        if row.get("status") == "completed" and row.get("val_bidirectional_r1_mean") is not None
    ]
    completed.sort(key=lambda row: (-row["val_bidirectional_r1_mean"], row.get("unfrozen_blocks") or 0))
    print_table("Trained Unfreezing Calibration", completed + [r for r in trained_rows if r not in completed])

    if completed:
        best = completed[0]
        for candidate in completed[1:]:
            diff = best["val_bidirectional_r1_mean"] - candidate["val_bidirectional_r1_mean"]
            if 0 <= diff <= 0.1 and (candidate.get("unfrozen_blocks") or math.inf) < (best.get("unfrozen_blocks") or math.inf):
                best = candidate
        print(
            "\nSelected trained calibration candidate: "
            f"{best['registry_id']} by validation bidirectional R@1 mean "
            f"({best['val_bidirectional_r1_mean']:.3f}); zero-shot rows were not candidates."
        )
    else:
        print("\nNo completed trained candidates yet; unfreezing depth is not selected.")


if __name__ == "__main__":
    main()

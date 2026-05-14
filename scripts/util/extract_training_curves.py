from __future__ import annotations

import argparse
import math
import os
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PROJECT = "iremurhan-bogazici-university/clip-retrieval"
DEFAULT_SUMMARY_CSV = Path("/Volumes/T7/Research/wandb/runs_summary.csv")
CURVES_DIR = Path("figs/cache/training_curves")
AGGREGATED_PATH = Path("figs/cache/training_curves_aggregated.csv")
SUMMARY_PATH = Path("figs/cache/training_summary.csv")
FIGURE_STEM = Path("figs/draft_training_curves")
EXCLUDE = {"B0v2", "B0plus_fixed", "B5_seg", "B0_proj1024"}

REQUESTED_KEYS = [
    "train/loss",
    "train/loss_clip",
    "train/loss_intra_img",
    "train/loss_intra_txt",
    "val/r1_i2t",
    "val/r1_t2i",
    "val/r5_i2t",
    "val/r5_t2i",
    "epoch",
    "_step",
]

ALIASES = {
    "train_loss": ["train/loss", "train/loss_total"],
    "train_loss_clip": ["train/loss_clip", "train/loss_inter"],
    "train_loss_intra_img": ["train/loss_intra_img"],
    "train_loss_intra_txt": ["train/loss_intra_txt"],
    "val_r1_i2t": ["val/r1_i2t"],
    "val_r1_t2i": ["val/r1_t2i"],
    "val_r5_i2t": ["val/r5_i2t"],
    "val_r5_t2i": ["val/r5_t2i"],
}

PER_RUN_COLUMNS = [
    "epoch",
    "step",
    "train_loss",
    "train_loss_clip",
    "train_loss_intra_img",
    "train_loss_intra_txt",
    "val_r1_i2t",
    "val_r1_t2i",
    "val_r5_i2t",
    "val_r5_t2i",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fresh W&B training curves and draft convergence plot.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project path: entity/project")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--curves-dir", type=Path, default=CURVES_DIR)
    parser.add_argument("--aggregated-path", type=Path, default=AGGREGATED_PATH)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--figure-stem", type=Path, default=FIGURE_STEM)
    return parser.parse_args()


def parse_identity_from_name(name: str | None) -> tuple[str | None, str | None, int | None]:
    match = re.match(r"(?P<run>.+?)_(?P<dataset>coco|flickr30k|flickr)_s?(?P<seed>\d+)(?:_\d+)?$", name or "")
    if not match:
        return None, None, None
    dataset = match.group("dataset")
    if dataset == "flickr":
        dataset = "flickr30k"
    return match.group("run"), dataset, int(match.group("seed"))


def config_value(run: Any, key: str) -> Any:
    config = dict(run.config or {})
    if key in config:
        return config[key]
    if "/" in key:
        value: Any = config
        for part in key.split("/"):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value
    return None


def scalar_to_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_identity(run: Any) -> tuple[str | None, str | None, int | None]:
    parsed_run_id, parsed_dataset, parsed_seed = parse_identity_from_name(getattr(run, "name", None))
    run_id = config_value(run, "run_id") or parsed_run_id
    dataset = config_value(run, "dataset") or parsed_dataset
    seed = scalar_to_int(config_value(run, "seed"))
    if seed is None:
        seed = parsed_seed
    return (str(run_id) if run_id is not None else None, str(dataset) if dataset is not None else None, seed)


def sanitize_filename(value: object) -> str:
    text = str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def curve_path(
    curves_dir: Path,
    run_id: str,
    seed: int,
    dataset: str | None,
    wandb_id: str,
    needs_dataset: bool,
    needs_wandb_id: bool,
) -> Path:
    safe_run = sanitize_filename(run_id)
    dataset_part = f"_{sanitize_filename(dataset)}" if needs_dataset and dataset else ""
    wandb_part = f"_{sanitize_filename(wandb_id)}" if needs_wandb_id else ""
    return curves_dir / f"{safe_run}{dataset_part}_seed{seed}{wandb_part}.csv"


def read_full_history(run: Any) -> pd.DataFrame:
    history = run.history(pandas=True, samples=100_000)
    if isinstance(history, pd.DataFrame) and not history.empty:
        return history
    rows = list(run.scan_history(page_size=10_000))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def best_source(history: pd.DataFrame, sources: list[str]) -> str | None:
    for source in sources:
        if source in history.columns and history[source].notna().any():
            return source
    return None


def normalize_history(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = pd.DataFrame(index=history.index)
    used: dict[str, list[str]] = defaultdict(list)

    if "epoch" in history.columns:
        out["epoch"] = pd.to_numeric(history["epoch"], errors="coerce")
        used["epoch"].append("epoch")
    elif "train/epoch" in history.columns:
        out["epoch"] = pd.to_numeric(history["train/epoch"], errors="coerce")
        used["epoch"].append("train/epoch")
    else:
        out["epoch"] = np.nan

    if "_step" in history.columns:
        out["step"] = pd.to_numeric(history["_step"], errors="coerce")
        used["step"].append("_step")
    elif "train/step" in history.columns:
        out["step"] = pd.to_numeric(history["train/step"], errors="coerce")
        used["step"].append("train/step")
    else:
        out["step"] = np.nan

    for output_col, sources in ALIASES.items():
        source = best_source(history, sources)
        if source is None:
            out[output_col] = np.nan
        else:
            out[output_col] = pd.to_numeric(history[source], errors="coerce")
            used[output_col].append(source)

    out = out[PER_RUN_COLUMNS].dropna(how="all", subset=[col for col in PER_RUN_COLUMNS if col not in {"epoch", "step"}])
    out = out.sort_values(["epoch", "step"], na_position="last", kind="mergesort").reset_index(drop=True)
    return out, dict(used)


def has_one_full_epoch(curve: pd.DataFrame) -> bool:
    if curve.empty or "epoch" not in curve:
        return False
    epochs = pd.to_numeric(curve["epoch"], errors="coerce").dropna()
    return bool((epochs >= 1).any())


def to_long(curve: pd.DataFrame, run_id: str, seed: int) -> pd.DataFrame:
    metric_cols = [col for col in PER_RUN_COLUMNS if col not in {"epoch", "step"}]
    long = curve.melt(id_vars=["epoch", "step"], value_vars=metric_cols, var_name="metric", value_name="value")
    long = long[long["value"].notna()].copy()
    long.insert(0, "seed", seed)
    long.insert(0, "run_id", run_id)
    return long[["run_id", "seed", "epoch", "step", "metric", "value"]]


def collapse_by_epoch(curve: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = curve[["epoch", "step", metric]].copy()
    sub = sub[sub["epoch"].notna() & sub[metric].notna()]
    if sub.empty:
        return pd.DataFrame(columns=["epoch", "step", metric])
    grouped = sub.groupby("epoch", sort=True, dropna=True)
    idx = grouped["step"].idxmax()
    return sub.loc[idx].sort_values("epoch").reset_index(drop=True)


def convergence_flag(curve: pd.DataFrame) -> bool:
    sub = curve[["epoch", "train_loss"]].copy()
    sub = sub[sub["epoch"].notna() & sub["train_loss"].notna()]
    if sub.empty:
        return False
    epoch_loss = sub.groupby("epoch", sort=True, dropna=True)["train_loss"].mean()
    losses = pd.to_numeric(epoch_loss, errors="coerce").dropna().to_numpy(dtype=float)
    if len(losses) < 3:
        return False
    total_range = float(np.nanmax(losses) - np.nanmin(losses))
    if not np.isfinite(total_range) or total_range <= 0:
        return True
    tail_n = max(2, int(math.ceil(0.20 * len(losses))))
    tail = losses[-tail_n:]
    tail_change = abs(float(tail[-1] - tail[0]))
    return bool(tail_change < 0.05 * total_range)


def summarize_curve(run_id: str, seed: int, curve: pd.DataFrame) -> dict[str, Any]:
    train = curve[curve["train_loss"].notna()].copy()
    final_train_loss = np.nan
    total_epochs = 0
    if not train.empty:
        train = train.sort_values(["epoch", "step"], na_position="last")
        final_train_loss = float(train.iloc[-1]["train_loss"])
    epochs = pd.to_numeric(curve["epoch"], errors="coerce").dropna()
    if not epochs.empty:
        total_epochs = int(math.floor(float(epochs.max())))

    row: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "total_epochs": total_epochs,
        "final_train_loss": final_train_loss,
        "converged": convergence_flag(curve),
    }
    for metric in ["val_r1_i2t", "val_r1_t2i"]:
        sub = curve[curve[metric].notna()].copy()
        if sub.empty:
            row[f"best_{metric}"] = np.nan
            row[f"epoch_at_best_{metric.removeprefix('val_')}"] = np.nan
            continue
        best_idx = sub[metric].astype(float).idxmax()
        row[f"best_{metric}"] = float(sub.loc[best_idx, metric])
        row[f"epoch_at_best_{metric.removeprefix('val_')}"] = sub.loc[best_idx, "epoch"]

    return {
        "run_id": row["run_id"],
        "seed": row["seed"],
        "total_epochs": row["total_epochs"],
        "final_train_loss": row["final_train_loss"],
        "best_val_r1_i2t": row["best_val_r1_i2t"],
        "epoch_at_best_r1_i2t": row["epoch_at_best_r1_i2t"],
        "best_val_r1_t2i": row["best_val_r1_t2i"],
        "epoch_at_best_r1_t2i": row["epoch_at_best_r1_t2i"],
        "converged": row["converged"],
    }


def mean_std_text(values: pd.Series) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "NA"
    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else np.nan
    if pd.isna(std):
        return f"{mean:.4g}"
    return f"{mean:.4g} +/- {std:.4g}"


def print_grouped_summary(summary: pd.DataFrame) -> None:
    print("\nSummary by run_id (mean +/- std across seeds when n>1):")
    numeric_cols = [
        "total_epochs",
        "final_train_loss",
        "best_val_r1_i2t",
        "epoch_at_best_r1_i2t",
        "best_val_r1_t2i",
        "epoch_at_best_r1_t2i",
    ]
    for run_id, sub in summary.groupby("run_id", sort=True):
        converged_count = int(sub["converged"].sum())
        cells = [f"n={len(sub)}", f"converged={converged_count}/{len(sub)}"]
        cells.extend(f"{col}={mean_std_text(sub[col])}" for col in numeric_cols)
        print(f"  {run_id}: " + "; ".join(cells))


def plot_draft(summary: pd.DataFrame, curves: list[tuple[str, int, pd.DataFrame]], figure_stem: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("results/cache/mplconfig")))
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        colors = sns.color_palette("colorblind", n_colors=max(1, summary["run_id"].nunique()))
    except Exception:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(max(1, summary["run_id"].nunique()))]

    run_ids = sorted(summary["run_id"].unique())
    color_by_run = dict(zip(run_ids, colors))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    panels = [("A", "train_loss", "Train loss"), ("B", "val_r1_i2t", "Val R@1 I2T")]
    for ax, (panel, metric, ylabel) in zip(axes, panels):
        for run_id, seed, curve in sorted(curves, key=lambda item: (item[0], item[1])):
            epoch_curve = collapse_by_epoch(curve, metric)
            if epoch_curve.empty:
                continue
            ax.plot(
                epoch_curve["epoch"],
                epoch_curve[metric],
                color=color_by_run.get(run_id, "0.4"),
                lw=0.9,
                alpha=0.75,
            )
            endpoint = epoch_curve.iloc[-1]
            ax.annotate(
                run_id,
                xy=(endpoint["epoch"], endpoint[metric]),
                xytext=(3, 0),
                textcoords="offset points",
                fontsize=5.5,
                color=color_by_run.get(run_id, "0.4"),
                va="center",
                clip_on=True,
            )
        ax.set_title(f"Panel {panel}: {ylabel}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    figure_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_stem.with_suffix(".pdf"))
    fig.savefig(figure_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def expected_run_ids(summary_csv: Path) -> set[str]:
    if not summary_csv.exists():
        print(f"WARNING: summary CSV not found for cross-check: {summary_csv}")
        return set()
    df = pd.read_csv(summary_csv)
    if "config/run_id" not in df.columns:
        print(f"WARNING: summary CSV lacks config/run_id: {summary_csv}")
        return set()
    return set(df.loc[df["config/run_id"].notna(), "config/run_id"].astype(str)) - EXCLUDE


def main() -> None:
    args = parse_args()
    import wandb

    args.curves_dir.mkdir(parents=True, exist_ok=True)
    args.aggregated_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = list(api.runs(args.project))
    identities = {run.id: run_identity(run) for run in runs}
    key_counts = Counter(
        (run_id, seed)
        for run_id, _dataset, seed in identities.values()
        if run_id is not None and seed is not None and run_id not in EXCLUDE
    )
    full_key_counts = Counter(
        (run_id, dataset, seed)
        for run_id, dataset, seed in identities.values()
        if run_id is not None and seed is not None and run_id not in EXCLUDE
    )

    curves: list[tuple[str, int, pd.DataFrame]] = []
    long_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    missing_key_warnings: list[str] = []
    skipped: list[str] = []

    for run in sorted(runs, key=lambda item: (getattr(item, "name", "") or "", item.id)):
        run_id, dataset, seed = identities[run.id]
        if run_id is None or seed is None:
            skipped.append(f"{run.name} ({run.id}): missing run_id or seed")
            continue
        if run_id in EXCLUDE:
            continue

        history = read_full_history(run)
        curve, used = normalize_history(history)
        if not has_one_full_epoch(curve):
            skipped.append(f"{run.name} ({run.id}): no complete epoch in history")
            continue

        missing_requested = []
        for key in REQUESTED_KEYS:
            if key == "train/loss" and "train_loss" in used:
                if "train/loss" not in used["train_loss"]:
                    missing_requested.append(f"{key} (using {used['train_loss'][0]})")
                continue
            if key == "train/loss_clip" and "train_loss_clip" in used:
                if "train/loss_clip" not in used["train_loss_clip"]:
                    missing_requested.append(f"{key} (using {used['train_loss_clip'][0]})")
                continue
            if key == "_step" and "step" in used:
                if "_step" not in used["step"]:
                    missing_requested.append(f"{key} (using {used['step'][0]})")
                continue
            if key == "epoch" and "epoch" in used:
                continue
            if key not in history.columns or not history[key].notna().any():
                missing_requested.append(key)
        if missing_requested:
            missing_key_warnings.append(f"{run.name} ({run.id}): " + ", ".join(missing_requested))

        needs_dataset = key_counts[(run_id, seed)] > 1
        needs_wandb_id = full_key_counts[(run_id, dataset, seed)] > 1
        path = curve_path(args.curves_dir, run_id, seed, dataset, run.id, needs_dataset, needs_wandb_id)
        curve.to_csv(path, index=False)

        curves.append((run_id, seed, curve))
        long_frames.append(to_long(curve, run_id, seed))
        summary_rows.append(summarize_curve(run_id, seed, curve))

    aggregated = (
        pd.concat(long_frames, ignore_index=True)
        if long_frames
        else pd.DataFrame(columns=["run_id", "seed", "epoch", "step", "metric", "value"])
    )
    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "run_id",
            "seed",
            "total_epochs",
            "final_train_loss",
            "best_val_r1_i2t",
            "epoch_at_best_r1_i2t",
            "best_val_r1_t2i",
            "epoch_at_best_r1_t2i",
            "converged",
        ],
    ).sort_values(["run_id", "seed"], ignore_index=True)

    aggregated.to_csv(args.aggregated_path, index=False)
    summary.to_csv(args.summary_path, index=False)

    if not summary.empty:
        plot_draft(summary, curves, args.figure_stem)

    extracted_run_ids = set(summary["run_id"].astype(str)) if not summary.empty else set()
    expected_ids = expected_run_ids(args.summary_csv)
    missing_vs_summary = sorted(expected_ids - extracted_run_ids)
    extra_vs_summary = sorted(extracted_run_ids - expected_ids)
    not_converged = summary.loc[~summary["converged"], ["run_id", "seed"]] if not summary.empty else pd.DataFrame()

    print(f"Total W&B runs seen: {len(runs)}")
    print(f"Total runs extracted: {len(summary)}")
    print("Runs per config:")
    for run_id, count in summary["run_id"].value_counts().sort_index().items():
        print(f"  {run_id}: {int(count)}")

    print("\nNot converged:")
    if not_converged.empty:
        print("  (none)")
    else:
        for row in not_converged.itertuples(index=False):
            print(f"  {row.run_id} seed={row.seed}")

    print("\nCross-check against /Volumes/T7/Research/wandb/runs_summary.csv:")
    if missing_vs_summary:
        print("  Missing run_ids: " + ", ".join(missing_vs_summary))
    else:
        print("  Missing run_ids: (none)")
    if extra_vs_summary:
        print("  Extra run_ids from fresh W&B: " + ", ".join(extra_vs_summary))
    else:
        print("  Extra run_ids from fresh W&B: (none)")

    if skipped:
        print("\nSkipped runs:")
        for item in skipped:
            print(f"  {item}")

    if missing_key_warnings:
        print("\nMissing/aliased history keys:")
        for item in missing_key_warnings:
            print(f"  {item}")

    print_grouped_summary(summary)
    print(f"\nSaved per-run curves to {args.curves_dir}")
    print(f"Saved long CSV to {args.aggregated_path}")
    print(f"Saved summary CSV to {args.summary_path}")
    print(f"Saved draft figure to {args.figure_stem.with_suffix('.pdf')} and {args.figure_stem.with_suffix('.png')}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()

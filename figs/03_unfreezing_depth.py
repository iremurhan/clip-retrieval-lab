from __future__ import annotations

import os
import re
from pathlib import Path

ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/figures")
)
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_ROOT / "cache" / "mplconfig"))
RESULTS_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_RESULTS_ROOT", "/Volumes/T7/Research/experiments/results")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers import (
    DEFAULT_CSV_PATH,
    EXCLUDE,
    SAVE_DATA_DIR,
    SAVE_FIG_DIR,
    load_runs,
)


# Each preset: the training dataset whose runs to read, output stem, and title.
PRESETS = {
    "coco": {
        "dataset": "coco",
        "title": "Unfreezing depth on COCO 5K",
        "output_stem": "03_unfreezing_depth_coco",
        "column_template": "summary/test/coco_5k_r{k}_{direction}",
    },
    "flickr30k": {
        "dataset": "flickr30k",
        "title": "Unfreezing depth on Flickr30K",
        "output_stem": "03B_unfreezing_depth_flickr",
        "column_template": "summary/test/r{k}_{direction}",
    },
}

SWEEP_DEPTHS = [0, 1, 2, 3, 4, 5, 6, 7]
RETRIEVAL_DIRECTIONS = [("i2t", "Image-to-text"), ("t2i", "Text-to-image")]
RECALL_SERIES = [
    (1, "R@1", "o", "-"),
    (5, "R@5", "s", "--"),
    (10, "R@10", "^", ":"),
]

UNFREEZE_DEPTH_COLUMNS = (
    "config/unfreeze_layers",
    "config/model/unfreeze_vision_layers",
    "config/model.unfreeze_vision_layers",
)


def _configure_matplotlib() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
            "axes.linewidth": 0.7,
            "axes.edgecolor": "0.2",
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _to_percent(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().empty:
        return numeric
    scale = 100.0 if numeric.dropna().max() <= 1.5 else 1.0
    return numeric * scale


def _unfreeze_depth_series(df: pd.DataFrame) -> pd.Series:
    for col in UNFREEZE_DEPTH_COLUMNS:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    raise KeyError(
        "Expected an exported unfreezing-depth config column. "
        f"Tried: {', '.join(UNFREEZE_DEPTH_COLUMNS)}. "
        "Re-export clean results with config/unfreeze_layers present."
    )


def _infer_unfreezing_log_identity(path: Path, dataset: str) -> tuple[int, int, str] | None:
    text = str(path)
    proj_match = re.search(r"B0_projonly", text)
    run_match = re.search(r"B0_uf(?P<depth>[1-7])", text)
    seed_match = re.search(r"_s(?P<seed>\d+)(?:/|$)", text)
    if not seed_match or f"/{dataset}/" not in text:
        return None
    if proj_match:
        depth = 0
    elif run_match:
        depth = int(run_match.group("depth"))
    else:
        return None
    seed = int(seed_match.group("seed"))
    run_id = "B0_projonly" if depth == 0 else f"B0_uf{depth}"
    return depth, seed, run_id


def _parse_log_test_results(path: Path) -> dict[tuple[str, int], float]:
    line_re = re.compile(
        r".*test/(?:coco_5k_)?r(?P<cutoff>1|5|10)_(?P<direction>i2t|t2i):\s+"
        r"(?P<value>[-+]?\d+(?:\.\d+)?)"
    )
    found: dict[tuple[str, int], float] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return found
    for line in lines:
        match = line_re.match(line)
        if not match:
            continue
        value = float(match.group("value"))
        if abs(value) <= 1.5:
            value *= 100.0
        found[(match.group("direction"), int(match.group("cutoff")))] = value
    return found


def _parse_log_validation_max(path: Path) -> dict[tuple[str, int], float]:
    """Return W&B-style val/<metric>/max values from epoch evaluation blocks."""
    line_re = re.compile(
        r"\s*(?P<direction>I2T|T2I):\s+R@1:\s+(?P<r1>[-+]?\d+(?:\.\d+)?)\s+\|\s+"
        r"R@5:\s+(?P<r5>[-+]?\d+(?:\.\d+)?)\s+\|\s+"
        r"R@10:\s+(?P<r10>[-+]?\d+(?:\.\d+)?)"
    )
    best: dict[tuple[str, int], float] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return best
    for line in lines:
        match = line_re.match(line)
        if not match:
            continue
        direction = match.group("direction").lower()
        for cutoff, group in ((1, "r1"), (5, "r5"), (10, "r10")):
            key = (direction, cutoff)
            value = float(match.group(group))
            best[key] = max(best.get(key, float("-inf")), value)
    return best


def _supplemental_rows_from_logs(preset: dict, existing_keys: set[tuple[int, str, int, int]]) -> list[dict]:
    """Fill missing sweep cells from local logs when W&B summary lacks test metrics.

    Prefer final test metrics from logs. Some lower-depth Flickr runs failed checkpoint
    persistence and skipped final test evaluation; for those, fall back to validation maxima
    and mark the source explicitly in the generated data CSV.
    """
    dataset = preset["dataset"]
    if not RESULTS_ROOT.exists():
        return []

    candidates = sorted((RESULTS_ROOT / dataset).glob("**/training.log"))
    rows = []
    for log_path in candidates:
        identity = _infer_unfreezing_log_identity(log_path, dataset)
        if identity is None:
            continue
        depth, seed, run_id = identity
        if depth not in {0, 1, 2, 3}:
            continue
        values = _parse_log_test_results(log_path)
        metric_source = "test_log"
        if not values and dataset == "flickr30k" and depth in {1, 2, 3}:
            values = _parse_log_validation_max(log_path)
            metric_source = "val_log_max"
        if not values:
            continue
        for direction, _direction_label in RETRIEVAL_DIRECTIONS:
            for cutoff, metric, _marker, _ls in RECALL_SERIES:
                key = (depth, direction, cutoff, seed)
                if key in existing_keys or (direction, cutoff) not in values:
                    continue
                rows.append(
                    {
                        "depth": depth,
                        "sweep_label": f"Unfreeze-{depth}",
                        "direction": direction,
                        "cutoff": cutoff,
                        "metric": metric,
                        "source_col": preset["column_template"].format(k=cutoff, direction=direction),
                        "metric_source": metric_source,
                        "seed": seed,
                        "run_ids": run_id,
                        "value": values[(direction, cutoff)],
                    }
                )
    return rows


def _contiguous_depth_frames(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if frame.empty:
        return []
    ordered = frame.sort_values("depth").reset_index(drop=True)
    segments = []
    start = 0
    depths = ordered["depth"].to_numpy(dtype=float)
    for idx in range(1, len(ordered)):
        if depths[idx] - depths[idx - 1] > 1.0:
            segments.append(ordered.iloc[start:idx])
            start = idx
    segments.append(ordered.iloc[start:])
    return segments


def build_data(preset: dict, csv_path=DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Per-(depth, metric, seed) recall for the unfreezing sweep of one dataset."""
    df = load_runs(csv_path, EXCLUDE, result_scope="sweep")
    df = df[df["include_in_sweep_figures"].astype(bool)].copy()
    df["dataset"] = df["config/dataset"].replace({"flickr": "flickr30k"})
    df = df[df["dataset"].eq(preset["dataset"])].copy()
    df["depth"] = _unfreeze_depth_series(df)
    df = df[df["depth"].notna()].copy()
    df = df[df["depth"].isin(SWEEP_DEPTHS)].copy()

    rows = []
    for (depth, sweep_label), group in df.groupby(["depth", "sweep_display_label"], dropna=False):
        for direction, _direction_label in RETRIEVAL_DIRECTIONS:
            for cutoff, metric, _marker, _ls in RECALL_SERIES:
                source_col = preset["column_template"].format(k=cutoff, direction=direction)
                if source_col not in group.columns:
                    continue
                seed_frame = (
                    pd.DataFrame(
                        {
                            "seed": group["config/seed"].values,
                            "run_id": group["internal_run_id"].astype(str).values,
                            "value": _to_percent(group[source_col]).values,
                        }
                    )
                    .dropna(subset=["value"])
                )
                per_seed = seed_frame.groupby("seed", dropna=False).agg(
                    value=("value", "mean"),
                    run_ids=("run_id", lambda values: ",".join(sorted(set(values)))),
                )
                for seed, seed_row in per_seed.iterrows():
                    rows.append(
                        {
                            "depth": int(depth) if float(depth).is_integer() else depth,
                            "sweep_label": sweep_label,
                            "direction": direction,
                            "cutoff": cutoff,
                            "metric": metric,
                            "source_col": source_col,
                            "metric_source": "summary_test",
                            "seed": seed,
                            "run_ids": seed_row["run_ids"],
                            "value": float(seed_row["value"]),
                        }
                    )

    existing = {
        (int(row["depth"]), str(row["direction"]), int(row["cutoff"]), int(row["seed"]))
        for row in rows
    }
    rows.extend(_supplemental_rows_from_logs(preset, existing))

    if not rows:
        raise ValueError(f"No {preset['dataset']} unfreezing-sweep rows found for k={SWEEP_DEPTHS}.")
    return pd.DataFrame(rows).sort_values(["direction", "cutoff", "depth", "seed"]).reset_index(drop=True)


def plot(data: pd.DataFrame, preset: dict) -> None:
    _configure_matplotlib()
    palette = sns.color_palette("colorblind", n_colors=len(RECALL_SERIES))
    color_by_metric = {metric: palette[i] for i, (_cutoff, metric, *_rest) in enumerate(RECALL_SERIES)}

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.15), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.43, wspace=0.14)
    fig.patch.set_facecolor("none")
    for ax in axes:
        ax.set_facecolor("none")

    summary = (
        data.groupby(["direction", "cutoff", "metric", "depth"], sort=True)["value"]
        .agg(mean="mean", std=lambda values: values.std(ddof=1), n_seeds="count")
        .reset_index()
    )
    finite_low = summary["mean"] - summary["std"].fillna(0.0)
    finite_high = summary["mean"] + summary["std"].fillna(0.0)
    ymin = float(finite_low.min())
    ymax = float(finite_high.max())
    pad = max((ymax - ymin) * 0.08, 0.25)
    depths = SWEEP_DEPTHS

    legend_handles = []
    for ax, (direction, direction_label) in zip(axes, RETRIEVAL_DIRECTIONS):
        for cutoff, metric, marker, linestyle in RECALL_SERIES:
            sub = summary[(summary["direction"].eq(direction)) & (summary["cutoff"].eq(cutoff))]
            if sub.empty:
                continue
            sub = sub.sort_values("depth")
            color = color_by_metric[metric]
            line = None
            for segment_idx, segment in enumerate(_contiguous_depth_frames(sub)):
                (segment_line,) = ax.plot(
                    segment["depth"].to_numpy(dtype=float),
                    segment["mean"].to_numpy(dtype=float),
                    marker=marker,
                    markersize=5.4,
                    linestyle=linestyle,
                    linewidth=1.45,
                    color=color,
                    markeredgecolor="white",
                    markeredgewidth=0.55,
                    zorder=3,
                    label=metric if segment_idx == 0 else None,
                )
                if line is None:
                    line = segment_line
            if direction == "i2t":
                legend_handles.append(line)
            multi = sub[sub["n_seeds"].gt(1)]
            if not multi.empty:
                ax.errorbar(
                    multi["depth"].to_numpy(dtype=float),
                    multi["mean"].to_numpy(dtype=float),
                    yerr=multi["std"].fillna(0.0).to_numpy(dtype=float),
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.9,
                    capsize=2.6,
                    capthick=0.8,
                    zorder=2,
                )
            singles = sub[sub["n_seeds"].eq(1)]
            for _, row in singles.iterrows():
                ax.annotate(
                    "*",
                    (float(row["depth"]), float(row["mean"])),
                    xytext=(3, 3),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color=color,
                    zorder=5,
                )

        ax.set_title(direction_label)
        ax.set_xticks(depths)
        ax.set_xticklabels(["proj\nonly", "1", "2", "3", "4\n(Base-min)", "5", "6", "7"])
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlim(min(depths) - 0.4, max(depths) + 0.4)
        ax.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)
        ax.tick_params(width=0.7, length=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axes[0].set_ylabel("Recall (%)")
    fig.suptitle(preset["title"], y=0.96, fontsize=10)
    fig.supxlabel("Trainable ViT block depth $k$", y=0.235, fontsize=8.5)
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.8,
    )

    fig.text(
        0.5,
        0.035,
        "* single seed (n=1)",
        ha="center",
        fontsize=6.8,
        color="0.35",
    )

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG_DIR / f"{preset['output_stem']}.pdf", bbox_inches="tight")
    fig.savefig(SAVE_FIG_DIR / f"{preset['output_stem']}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_report(data: pd.DataFrame, preset: dict) -> None:
    print(f"Unfreezing depth sweep ({preset['dataset']}):")
    print("Run mapping:")
    coverage = (
        data.groupby(["depth", "sweep_label"], sort=True)
        .agg(
            n_seeds=("seed", "nunique"),
            seeds=("seed", lambda values: ",".join(str(int(v)) for v in sorted(set(values)))),
            run_ids=("run_ids", lambda values: ",".join(sorted(set(",".join(values).split(","))))),
        )
        .reset_index()
    )
    for _, row in coverage.iterrows():
        depth = row["depth"]
        label = "Base-min" if int(depth) == 4 else str(row["sweep_label"])
        print(
            f"  k={int(depth)} {label}: runs={row['run_ids']} "
            f"dataset={preset['dataset']} seeds={row['seeds']} n={int(row['n_seeds'])}"
        )
    source_counts = data.groupby("metric_source").size().to_dict() if "metric_source" in data.columns else {}
    if source_counts:
        print("Metric sources:", ", ".join(f"{key}={value}" for key, value in sorted(source_counts.items())))
    for (depth, direction, metric), grp in data.groupby(["depth", "direction", "metric"]):
        n = grp["seed"].nunique()
        sweep_label = grp["sweep_label"].dropna().iloc[0] if grp["sweep_label"].notna().any() else f"k={depth}"
        flag = "  (single seed)" if n == 1 else ""
        print(
            f"  k={depth} {sweep_label:22s} {direction.upper()} {metric:4s} "
            f"n_seeds={n} mean={grp['value'].mean():.1f}{flag}"
        )
    available = set(data["source_col"])
    expected = {
        preset["column_template"].format(k=cutoff, direction=direction)
        for direction, _direction_label in RETRIEVAL_DIRECTIONS
        for cutoff, _metric, _marker, _ls in RECALL_SERIES
    }
    missing = sorted(expected - available)
    print("Missing metric columns:", ", ".join(missing) if missing else "(none)")
    print("Depths plotted: proj-only, k=1, k=2, k=3, k=4, k=5, k=6, k=7.")


def run_preset(name: str) -> None:
    preset = PRESETS[name]
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data(preset)
    print_report(data, preset)
    data.to_csv(SAVE_DATA_DIR / f"{preset['output_stem']}_data.csv", index=False)
    plot(data, preset)


def main() -> None:
    import sys

    names = sys.argv[1:] or ["coco", "flickr30k"]
    for name in names:
        if name not in PRESETS:
            raise SystemExit(f"Unknown preset {name!r}; choose from {list(PRESETS)}")
        run_preset(name)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_ROOT = Path(
    os.environ.get("CLIP_RETRIEVAL_ARTIFACT_ROOT", "/Volumes/T7/Research/artifacts/clip-retrieval-lab")
)
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_ROOT / "cache" / "mplconfig"))

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
    SINGLE_SEED_FOOTNOTE,
    load_runs,
    single_seed_marker_kw,
)


# Each preset: the training dataset whose runs to read, the metric series (label,
# source_col, marker, linestyle), the output stem, and a title.
PRESETS = {
    "coco": {
        "dataset": "coco",
        "title": "Unfreezing depth vs. COCO 5K text-to-image recall",
        "output_stem": "03_unfreezing_depth",
        "metrics": [
            ("COCO 5K T2I R@1", "summary/test/coco_5k_r1_t2i", "o", "-"),
            ("COCO 5K T2I R@5", "summary/test/coco_5k_r5_t2i", "s", "--"),
            ("COCO 5K T2I R@10", "summary/test/coco_5k_r10_t2i", "^", ":"),
        ],
    },
    # Flickr R@1 is near-saturated, so we show both directions at R@1 plus T2I R@5/R@10.
    "flickr30k": {
        "dataset": "flickr30k",
        "title": "Unfreezing depth vs. Flickr30K recall",
        "output_stem": "03B_unfreezing_depth_flickr",
        "metrics": [
            ("Flickr30K I2T R@1", "summary/test/r1_i2t", "D", "-."),
            ("Flickr30K T2I R@1", "summary/test/r1_t2i", "o", "-"),
            ("Flickr30K T2I R@5", "summary/test/r5_t2i", "s", "--"),
            ("Flickr30K T2I R@10", "summary/test/r10_t2i", "^", ":"),
        ],
    },
}


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


def build_data(preset: dict, csv_path=DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Per-(depth, metric, seed) recall for the unfreezing sweep of one dataset."""
    df = load_runs(csv_path, EXCLUDE, result_scope="sweep")
    df = df[df["include_in_sweep_figures"].astype(bool)].copy()
    df["dataset"] = df["config/dataset"].replace({"flickr": "flickr30k"})
    df = df[df["dataset"].eq(preset["dataset"])].copy()
    df["depth"] = pd.to_numeric(df["sweep_index"], errors="coerce")
    df = df[df["depth"].notna()].copy()

    rows = []
    for (depth, sweep_label), group in df.groupby(["depth", "sweep_display_label"], dropna=False):
        for metric, source_col, _marker, _ls in preset["metrics"]:
            if source_col not in group.columns:
                continue
            per_seed = (
                pd.DataFrame(
                    {"seed": group["config/seed"].values, "value": _to_percent(group[source_col]).values}
                )
                .dropna(subset=["value"])
                .groupby("seed", dropna=False)["value"]
                .mean()
            )
            for seed, value in per_seed.items():
                rows.append(
                    {
                        "depth": int(depth) if float(depth).is_integer() else depth,
                        "sweep_label": sweep_label,
                        "metric": metric,
                        "source_col": source_col,
                        "seed": seed,
                        "value": float(value),
                    }
                )

    if not rows:
        raise ValueError(f"No {preset['dataset']} unfreezing-sweep rows found for {preset['metrics']}.")
    return pd.DataFrame(rows).sort_values(["metric", "depth", "seed"]).reset_index(drop=True)


def plot(data: pd.DataFrame, preset: dict) -> None:
    _configure_matplotlib()
    metrics = preset["metrics"]
    palette = sns.color_palette("colorblind", n_colors=len(metrics))
    color_by_metric = {metric: palette[i] for i, (metric, *_rest) in enumerate(metrics)}

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.16)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    depths = sorted(data["depth"].unique())

    for metric, source_col, marker, linestyle in metrics:
        sub = data[data["metric"].eq(metric)]
        if sub.empty:
            continue
        color = color_by_metric[metric]

        # Subtle mean line per metric (visually secondary to the seed points).
        means = sub.groupby("depth")["value"].mean().sort_index()
        ax.plot(
            means.index.to_numpy(dtype=float),
            means.to_numpy(dtype=float),
            linestyle=linestyle,
            linewidth=1.0,
            color=color,
            alpha=0.45,
            zorder=2,
            label=metric,
        )

        # Prominent individual seed points; single-seed depths use the hollow convention.
        counts = sub.groupby("depth")["seed"].nunique()
        for depth, grp in sub.groupby("depth"):
            x = np.full(len(grp), float(depth))
            y = grp["value"].to_numpy(dtype=float)
            if int(counts.loc[depth]) == 1:
                # Hollow marker = single seed (shared convention). No inline "n=" label
                # here, to avoid confusion with the k (unfreeze-depth) x-axis.
                ax.scatter(x, y, marker=marker, s=58, zorder=4, **single_seed_marker_kw(color))
            else:
                ax.scatter(
                    x, y, marker=marker, s=46, color=color, edgecolors="white", linewidths=0.6, zorder=4
                )

    ax.set_title(preset["title"])
    ax.set_xlabel("Trainable ViT block depth $k$  (k=4 = Base-min)")
    ax.set_ylabel("Recall (%)")
    ax.set_xticks(depths)
    if depths:
        ax.set_xlim(min(depths) - 0.4, max(depths) + 0.4)
    ax.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)
    ax.tick_params(width=0.7, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, handlelength=2.2)

    fig.text(0.5, 0.015, SINGLE_SEED_FOOTNOTE, ha="center", fontsize=6.8, color="0.35")

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG_DIR / f"{preset['output_stem']}.pdf")
    fig.savefig(SAVE_FIG_DIR / f"{preset['output_stem']}.png", dpi=300)
    plt.close(fig)


def print_report(data: pd.DataFrame, preset: dict) -> None:
    print(f"Unfreezing depth sweep ({preset['dataset']}):")
    for (depth, metric), grp in data.groupby(["depth", "metric"]):
        n = grp["seed"].nunique()
        sweep_label = grp["sweep_label"].dropna().iloc[0] if grp["sweep_label"].notna().any() else f"k={depth}"
        flag = "  (single seed)" if n == 1 else ""
        print(f"  k={depth} {sweep_label:22s} {metric:18s} n_seeds={n} mean={grp['value'].mean():.1f}{flag}")
    print(f"Metrics: {', '.join(m for m, *_ in preset['metrics'])}. Excluded: Proj-1024, Text-BLIP, main interventions.")


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

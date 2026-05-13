from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/cache/mplconfig")))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers import DEFAULT_CSV_PATH, SAVE_DATA_DIR, SAVE_FIG_DIR, load_runs


EXCLUDE_LIST = ["B0v2", "B0plus_fixed", "B5_seg"]

PANELS = {
    "flickr30k": {
        "title": "Flickr30K",
        "i2t": "summary/test/r1_i2t",
        "t2i": "summary/test/r1_t2i",
    },
    "coco": {
        "title": "COCO 5K",
        "i2t": "summary/test/coco_5k_r1_i2t",
        "t2i": "summary/test/coco_5k_r1_t2i",
    },
}

DIRECTIONS = {
    "i2t": {
        "label": "Image-to-Text R@1",
        "linestyle": "-",
        "marker": "o",
    },
    "t2i": {
        "label": "Text-to-Image R@1",
        "linestyle": "--",
        "marker": "s",
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
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _scale_for_percent(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return 1.0
    return 100.0 if values.max() <= 1.5 else 1.0


def build_data(csv_path=DEFAULT_CSV_PATH) -> pd.DataFrame:
    df = load_runs(csv_path, EXCLUDE_LIST)
    run_id = df["config/run_id"].astype(str)
    family = run_id.str.fullmatch(r"B0|B0_uf\d+")
    df = df[family & df["config/unfreeze_layers"].notna()].copy()
    df["unfreeze_depth"] = pd.to_numeric(df["config/unfreeze_layers"], errors="coerce")
    df = df[df["unfreeze_depth"].notna()].copy()

    rows = []
    for dataset, panel in PANELS.items():
        metric_cols = [panel["i2t"], panel["t2i"]]
        sub = df[(df["config/dataset"] == dataset) & df[metric_cols].notna().any(axis=1)].copy()
        if sub.empty:
            continue

        scales = {metric: _scale_for_percent(sub[metric]) for metric in metric_cols}
        grouped = sub.groupby(["config/run_id", "unfreeze_depth"], dropna=False)
        for (config, depth), group in grouped:
            for direction, metric in (("i2t", panel["i2t"]), ("t2i", panel["t2i"])):
                values = group[metric].dropna() * scales[metric]
                if values.empty:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "config": config,
                        "unfreeze_depth": int(depth) if float(depth).is_integer() else depth,
                        "direction": direction,
                        "mean": values.mean(),
                        "std": values.std(ddof=1) if len(values) >= 2 else np.nan,
                        "n_seeds": len(values),
                    }
                )

    if not rows:
        raise ValueError("No B0/B0_uf* unfreezing rows found for Flickr30K or COCO.")

    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "direction", "unfreeze_depth", "config"])
        .reset_index(drop=True)
    )


def _plot_direction(ax: plt.Axes, sub: pd.DataFrame, direction: str, color) -> None:
    style = DIRECTIONS[direction]
    line = sub[sub["direction"] == direction].sort_values("unfreeze_depth").copy()
    if line.empty:
        return

    x = line["unfreeze_depth"].to_numpy(dtype=float)
    y = line["mean"].to_numpy(dtype=float)
    std = line["std"].to_numpy(dtype=float)
    n_seeds = line["n_seeds"].to_numpy(dtype=int)
    multi = n_seeds >= 2

    ax.plot(
        x,
        y,
        linestyle=style["linestyle"],
        marker=style["marker"],
        markersize=4.5,
        linewidth=1.35,
        color=color,
        markerfacecolor=color,
        markeredgecolor=color,
        label=style["label"],
        zorder=3,
    )

    if multi.any():
        lower = np.where(multi, y - std, np.nan)
        upper = np.where(multi, y + std, np.nan)
        ax.fill_between(x, lower, upper, color=color, alpha=0.16, linewidth=0, zorder=2)

    single = ~multi
    if single.any():
        ax.scatter(
            x[single],
            y[single],
            marker=style["marker"],
            s=38,
            facecolors="white",
            edgecolors=[color],
            linewidths=1.2,
            zorder=4,
        )
        for x_val, y_val in zip(x[single], y[single], strict=True):
            ax.annotate(
                "n=1",
                xy=(x_val, y_val),
                xytext=(4, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=6,
                color="0.35",
            )


def plot(data: pd.DataFrame) -> None:
    _configure_matplotlib()
    palette = sns.color_palette("colorblind", n_colors=2)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.24, wspace=0.28)
    fig.patch.set_facecolor("none")

    handles = None
    labels = None
    for ax, (dataset, panel) in zip(axes, PANELS.items(), strict=True):
        sub = data[data["dataset"] == dataset].copy()
        ax.set_facecolor("none")
        if sub.empty:
            ax.set_title(panel["title"])
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="0.4")
            continue

        _plot_direction(ax, sub, "i2t", palette[0])
        _plot_direction(ax, sub, "t2i", palette[1])

        y_min = sub["mean"].min() - 2.0
        y_max = sub["mean"].max() + 1.0
        depths = sorted(sub["unfreeze_depth"].unique())
        x_min = min(depths) - 0.35
        x_max = max(depths) + 0.35

        ax.set_title(panel["title"])
        ax.set_xlabel("Unfrozen ViT blocks")
        ax.set_ylabel("R@1 (%)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(depths)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(width=0.7, length=3)
        ax.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.04),
            ncol=2,
            frameon=False,
            handlelength=2.2,
            columnspacing=1.4,
        )

    fig.savefig(SAVE_FIG_DIR / "03_unfreezing_depth.pdf")
    fig.savefig(SAVE_FIG_DIR / "03_unfreezing_depth.png", dpi=300)
    plt.close(fig)


def main() -> None:
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()

    print("Detected unfreezing depths:")
    summary = data.groupby(["dataset", "config", "unfreeze_depth"], sort=True)["n_seeds"].max()
    for (dataset, config, depth), n_seeds in summary.items():
        print(f"  {dataset} {config} depth={depth}: n_seeds={int(n_seeds)}")

    data.to_csv(SAVE_DATA_DIR / "03_unfreezing_depth_data.csv", index=False)
    plot(data)


if __name__ == "__main__":
    main()

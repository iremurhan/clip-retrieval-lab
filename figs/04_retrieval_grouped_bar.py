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

from helpers import (
    DEFAULT_CSV_PATH,
    RETRIEVAL_DATASETS,
    RETRIEVAL_DIRECTIONS,
    RETRIEVAL_KS,
    SAVE_DATA_DIR,
    SAVE_FIG_DIR,
    build_retrieval_data,
    print_retrieval_report,
    write_retrieval_table,
)


OUTPUT_STEM = "04A_retrieval_grouped_bar"


def configure_matplotlib() -> None:
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
            "legend.fontsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def axis_limits(panel: pd.DataFrame) -> tuple[float, float]:
    values = panel["mean"].dropna()
    if values.empty:
        return 0.0, 1.0
    lower = panel["mean"] - panel["std"].where(panel["n_seeds"].ge(2), 0.0).fillna(0.0)
    upper = panel["mean"] + panel["std"].where(panel["n_seeds"].ge(2), 0.0).fillna(0.0)
    lo = float(lower.dropna().min())
    hi = float(upper.dropna().max())
    pad = max((hi - lo) * 0.14, 0.75)
    return max(0.0, lo - pad), hi + pad


def config_labels(data: pd.DataFrame, configs: list[str]) -> dict[str, str]:
    labels = {}
    for run_id in configs:
        sub = data[data["run_id"].eq(run_id)]
        labels[run_id] = f"{run_id}{'*' if sub['n_seeds'].eq(1).any() else ''}"
    return labels


def plot(data: pd.DataFrame) -> None:
    configure_matplotlib()
    configs = list(data.attrs["config_order"])
    colors = dict(zip(configs, sns.color_palette("colorblind", n_colors=len(configs)), strict=True))
    labels = config_labels(data, configs)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.93, bottom=0.24, wspace=0.16, hspace=0.30)
    fig.patch.set_facecolor("none")

    for row_idx, (dataset, dataset_spec) in enumerate(RETRIEVAL_DATASETS.items()):
        for col_idx, direction in enumerate(RETRIEVAL_DIRECTIONS):
            ax = axes[row_idx, col_idx]
            panel = data[data["dataset"].eq(dataset) & data["direction"].eq(direction)].copy()
            ax.set_facecolor("none")

            x = np.arange(len(RETRIEVAL_KS), dtype=float)
            width = min(0.78 / max(len(configs), 1), 0.055)
            offsets = (np.arange(len(configs)) - (len(configs) - 1) / 2.0) * width

            for idx, run_id in enumerate(configs):
                sub = panel[panel["run_id"].eq(run_id)].set_index("k").reindex(RETRIEVAL_KS)
                means = sub["mean"].to_numpy(dtype=float)
                stds = sub["std"].to_numpy(dtype=float)
                n_seeds = sub["n_seeds"].fillna(0).to_numpy(dtype=int)
                valid = np.isfinite(means)
                if not valid.any():
                    continue
                yerr = np.where((n_seeds >= 2) & np.isfinite(stds), stds, 0.0)
                ax.bar(
                    x[valid] + offsets[idx],
                    means[valid],
                    width=width * 0.90,
                    color=colors[run_id],
                    edgecolor="white",
                    linewidth=0.35,
                    yerr=yerr[valid],
                    error_kw={"ecolor": "0.18", "elinewidth": 0.55, "capsize": 1.2, "capthick": 0.55},
                    zorder=3,
                )

            ax.set_title(f"{dataset_spec['label']} {RETRIEVAL_DIRECTIONS[direction]}", pad=5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"R@{k}" for k in RETRIEVAL_KS])
            ax.set_xlim(-0.55, len(RETRIEVAL_KS) - 0.45)
            ax.set_ylim(*axis_limits(panel))
            ax.grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
            ax.tick_params(width=0.7, length=3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel("Recall (%)")
            else:
                ax.tick_params(axis="y", labelleft=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[run_id]) for run_id in configs]
    fig.legend(
        handles,
        [labels[run_id] for run_id in configs],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.85,
        handletextpad=0.35,
    )
    fig.text(0.5, 0.02, "* at least one dataset is single-seed", ha="center", va="bottom", fontsize=7, color="0.35")

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG_DIR / f"{OUTPUT_STEM}.pdf")
    fig.savefig(SAVE_FIG_DIR / f"{OUTPUT_STEM}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = build_retrieval_data(DEFAULT_CSV_PATH)
    print_retrieval_report(data)
    data.to_csv(SAVE_DATA_DIR / f"{OUTPUT_STEM}_data.csv", index=False)
    write_retrieval_table(data)
    plot(data)


if __name__ == "__main__":
    main()

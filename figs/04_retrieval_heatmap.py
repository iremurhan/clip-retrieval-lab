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
from matplotlib.colors import TwoSlopeNorm

from helpers import (
    DEFAULT_CSV_PATH,
    HEATMAP_COLUMNS,
    SAVE_DATA_DIR,
    SAVE_FIG_DIR,
    SINGLE_SEED_FOOTNOTE,
    build_heatmap_retrieval_data,
    print_heatmap_report,
    single_seed_cell_patch,
)


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
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def annotate(ax: plt.Axes, mean, delta, n_seeds) -> None:
    n_rows, n_cols = delta.shape
    for r in range(n_rows):
        for c in range(n_cols):
            if not np.isfinite(delta[r, c]):
                ax.text(c, r, "--", ha="center", va="center", fontsize=7, color="0.4")
                continue
            single = n_seeds[r, c] == 1
            text = f"{delta[r, c]:+.1f}{'*' if single else ''}"
            ax.text(c, r, text, ha="center", va="center", fontsize=7, color="0.05")
            if single:
                ax.add_patch(single_seed_cell_patch(c, r))


def dataset_group_bands(ax: plt.Axes, columns) -> None:
    """Draw dataset group brackets and labels below the column axis."""
    groups = []
    start = 0
    current = columns[0][0]
    for idx, (group, *_rest) in enumerate(columns):
        if group != current:
            groups.append((current, start, idx - 1))
            current = group
            start = idx
    groups.append((current, start, len(columns) - 1))

    n_rows = ax.get_ylim()[0]  # bottom (rows are top-down after invert_yaxis)
    y = n_rows + 0.5
    for group, lo, hi in groups:
        ax.plot([lo - 0.4, hi + 0.4], [y, y], color="0.3", linewidth=0.9, clip_on=False)
        ax.text((lo + hi) / 2.0, y + 0.28, group, ha="center", va="top", fontsize=8, color="0.15", clip_on=False)


def plot(data: pd.DataFrame) -> None:
    configure_matplotlib()
    rows = data.attrs["row_labels"]
    columns = data.attrs["columns"]
    mean = data.attrs["mean"]
    delta = data.attrs["delta"]
    n_seeds = data.attrs["n_seeds"]

    if not rows:
        raise ValueError("No main-intervention rows available for the heatmap.")

    finite = delta[np.isfinite(delta)]
    max_abs = max(float(np.max(np.abs(finite))) if finite.size else 0.5, 0.5)

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("0.92")
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    height = max(3.2, 0.42 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(8.0, height))
    # Reserve a fixed ~0.8in band at the bottom for the dataset brackets + footnote, so short
    # (few-row) panels do not collide their group labels with the footnote.
    bottom = 0.8 / height
    fig.subplots_adjust(left=0.16, right=0.86, top=1 - 0.9 / height, bottom=bottom)
    fig.patch.set_facecolor("none")

    # pcolormesh draws vector quad cells (fully scalable PDF); mask NaN so set_bad applies.
    masked = np.ma.masked_invalid(delta)
    edges_x = np.arange(len(columns) + 1) - 0.5
    edges_y = np.arange(len(rows) + 1) - 0.5
    image = ax.pcolormesh(edges_x, edges_y, masked, cmap=cmap, norm=norm, edgecolors="white", linewidth=0.7)
    ax.set_aspect("auto")
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.invert_yaxis()  # row 0 (Base-min) on top, matching imshow convention

    metric_labels = [m for _g, m, *_ in columns]
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(metric_labels)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.tick_params(width=0.6, length=2.5)
    ax.tick_params(which="minor", top=False, bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    annotate(ax, mean, delta, n_seeds)
    dataset_group_bands(ax, columns)

    reference_label = data.attrs["reference_label"]
    cax = fig.add_axes([0.875, 0.20, 0.02, 0.55])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(f"Gain over {reference_label} (pp)", fontsize=8)
    cbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    cbar.solids.set_rasterized(False)  # keep the colorbar vector in the PDF

    fig.text(
        0.5,
        0.012,
        f"Signed change vs. {reference_label} per metric (red = below, blue = above).  " + SINGLE_SEED_FOOTNOTE,
        ha="center",
        fontsize=6.8,
        color="0.35",
    )

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = data.attrs["output_stem"]
    fig.savefig(SAVE_FIG_DIR / f"{stem}.pdf")
    fig.savefig(SAVE_FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    import sys

    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    panels = sys.argv[1:] or ["zeroshot", "basemin", "intrareg"]
    for panel in panels:
        print(f"\n===== heatmap panel: {panel} =====")
        data = build_heatmap_retrieval_data(panel, DEFAULT_CSV_PATH)
        print_heatmap_report(data)
        data.to_csv(SAVE_DATA_DIR / f"{data.attrs['output_stem']}_data.csv", index=False)
        plot(data)


if __name__ == "__main__":
    main()

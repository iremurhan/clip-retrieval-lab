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
from matplotlib.patches import Rectangle

from helpers import (
    DEFAULT_CSV_PATH,
    RETRIEVAL_DATASETS,
    SAVE_DATA_DIR,
    SAVE_FIG_DIR,
    build_retrieval_data,
    print_retrieval_report,
    write_retrieval_table,
)


OUTPUT_STEM = "04B_retrieval_heatmap"
METRICS = [
    ("i2t", 1, "R@1\nI2T"),
    ("i2t", 5, "R@5\nI2T"),
    ("i2t", 10, "R@10\nI2T"),
    ("t2i", 1, "R@1\nT2I"),
    ("t2i", 5, "R@5\nT2I"),
    ("t2i", 10, "R@10\nT2I"),
]


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
            "ytick.labelsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def display_run_id(run_id: str) -> str:
    if run_id.startswith("B5d_multistream_"):
        return "B5d_multistream\n" + run_id.removeprefix("B5d_multistream_")
    if run_id.startswith("B5") and "_seg_" in run_id:
        prefix, suffix = run_id.split("_seg_", maxsplit=1)
        return f"{prefix}_seg\n{suffix}"
    return run_id


def matrices(data: pd.DataFrame, dataset: str, configs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = [run_id for run_id in configs if not data[data["dataset"].eq(dataset) & data["run_id"].eq(run_id)].empty]
    mean = np.full((len(rows), len(METRICS)), np.nan)
    std = np.full_like(mean, np.nan)
    n_seeds = np.zeros_like(mean, dtype=int)
    delta = np.full_like(mean, np.nan)
    lookup = data.set_index(["dataset", "run_id", "direction", "k"])

    for row_idx, run_id in enumerate(rows):
        for col_idx, (direction, k, _label) in enumerate(METRICS):
            key = (dataset, run_id, direction, k)
            if key not in lookup.index:
                continue
            record = lookup.loc[key]
            mean[row_idx, col_idx] = record["mean"]
            std[row_idx, col_idx] = record["std"]
            n_seeds[row_idx, col_idx] = int(record["n_seeds"])

    if "B0" in rows:
        baseline = mean[rows.index("B0"), :]
        delta = mean - baseline
    return mean, std, n_seeds, delta


def annotate_heatmap(ax: plt.Axes, mean: np.ndarray, std: np.ndarray, n_seeds: np.ndarray) -> None:
    for row_idx in range(mean.shape[0]):
        for col_idx in range(mean.shape[1]):
            if not np.isfinite(mean[row_idx, col_idx]):
                ax.text(col_idx, row_idx, "--", ha="center", va="center", fontsize=6.5, color="0.35")
                continue
            single = n_seeds[row_idx, col_idx] == 1
            mean_text = f"{mean[row_idx, col_idx]:.1f}{'*' if single else ''}"
            ax.text(col_idx, row_idx - 0.10, mean_text, ha="center", va="center", fontsize=6.4, color="0.05")
            if n_seeds[row_idx, col_idx] >= 2 and np.isfinite(std[row_idx, col_idx]):
                ax.text(
                    col_idx,
                    row_idx + 0.23,
                    f"+/- {std[row_idx, col_idx]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=4.7,
                    color="0.20",
                )
            if single:
                ax.add_patch(Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1, fill=False, edgecolor="0.05", linewidth=0.8))


def plot(data: pd.DataFrame) -> None:
    configure_matplotlib()
    configs = list(data.attrs["config_order"])
    by_dataset = {}
    max_abs = 0.0
    for dataset in RETRIEVAL_DATASETS:
        mean, std, n_seeds, delta = matrices(data, dataset, configs)
        rows = [run_id for run_id in configs if not data[data["dataset"].eq(dataset) & data["run_id"].eq(run_id)].empty]
        by_dataset[dataset] = (rows, mean, std, n_seeds, delta)
        if np.isfinite(delta).any():
            max_abs = max(max_abs, float(np.nanmax(np.abs(delta))))
    max_abs = max(max_abs, 0.5)

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("0.92")
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, axes = plt.subplots(1, 2, figsize=(9, 6))
    fig.subplots_adjust(left=0.16, right=0.88, top=0.92, bottom=0.12, wspace=0.34)
    fig.patch.set_facecolor("none")
    image = None

    for ax, (dataset, spec) in zip(axes, RETRIEVAL_DATASETS.items(), strict=True):
        rows, mean, std, n_seeds, delta = by_dataset[dataset]
        image = ax.imshow(delta, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(spec["label"], pad=7)
        ax.set_xticks(np.arange(len(METRICS)))
        ax.set_xticklabels([label for *_rest, label in METRICS])
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([display_run_id(run_id) for run_id in rows])
        ax.tick_params(width=0.6, length=2.5)
        ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        annotate_heatmap(ax, mean, std, n_seeds)

    if image is not None:
        cax = fig.add_axes([0.905, 0.20, 0.018, 0.60])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("Delta from B0 (pp)", fontsize=8)
        cbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    fig.text(0.5, 0.04, "* single seed; color encodes change from B0 within each metric column", ha="center", fontsize=7, color="0.35")

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

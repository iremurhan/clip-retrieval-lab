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

from helpers import DEFAULT_CSV_PATH, SAVE_DATA_DIR, SAVE_FIG_DIR, aggregate_seeds, load_runs

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - optional plotting nicety
    adjust_text = None


EXCLUDE_LIST = ["B0v2", "B0plus_fixed", "B5_seg"]

METRICS = {
    "i2t": {
        "x": "summary/test/coco_5k_r1_i2t",
        "y": "summary/test/eccv_map_at_r_i2t",
        "x_label": "COCO 5K R@1 (I2T) [%]",
        "y_label": "ECCV mAP@R (I2T) [%]",
        "title": "Image-to-Text",
    },
    "t2i": {
        "x": "summary/test/coco_5k_r1_t2i",
        "y": "summary/test/eccv_map_at_r_t2i",
        "x_label": "COCO 5K R@1 (T2I) [%]",
        "y_label": "ECCV mAP@R (T2I) [%]",
        "title": "Text-to-Image",
    },
}

VALUE_COLS = sorted({metric[key] for metric in METRICS.values() for key in ("x", "y")})


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
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _as_percent_if_fraction(series: pd.Series) -> pd.Series:
    return series * 100.0 if series.dropna().max() <= 1.5 else series


def _palette(configs: list[str]) -> dict[str, tuple[float, float, float]]:
    if len(configs) <= 10:
        colors = sns.color_palette("colorblind", n_colors=len(configs))
    else:
        colors = sns.color_palette("tab20", n_colors=len(configs))
    return dict(zip(configs, colors, strict=True))


def build_data(csv_path=DEFAULT_CSV_PATH) -> pd.DataFrame:
    df = load_runs(csv_path, EXCLUDE_LIST)
    filtered = df[(df["config/dataset"] == "coco") & df["summary/test/eccv_map_at_r_t2i"].notna()].copy()
    if filtered.empty:
        raise ValueError("No COCO rows with ECCV mAP@R results found.")

    aggregate = aggregate_seeds(filtered, VALUE_COLS)
    aggregate = aggregate[aggregate["config/dataset"] == "coco"].copy()

    rows = []
    for _, run in aggregate.iterrows():
        row = {"config": run["config/run_id"], "n_seeds": int(run["n_seeds"])}
        for metric_name, metric in METRICS.items():
            for axis_name in ("x", "y"):
                col = metric[axis_name]
                mean_value = run[f"{col}_mean"]
                std_value = run[f"{col}_std"]
                scale = 100.0 if filtered[col].dropna().max() <= 1.5 else 1.0
                row[f"{metric_name}_{axis_name}_mean"] = mean_value * scale
                row[f"{metric_name}_{axis_name}_std"] = std_value * scale if pd.notna(std_value) else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("config").reset_index(drop=True)


def _axis_limits(data: pd.DataFrame, x_col: str, y_col: str) -> tuple[tuple[float, float], tuple[float, float]]:
    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()
    x_pad = max((x.max() - x.min()) * 0.34, 1.2)
    y_pad = max((y.max() - y.min()) * 0.45, 1.0)
    return (x.min() - x_pad, x.max() + x_pad), (y.min() - y_pad, y.max() + y_pad)


def _plot_reference_lines(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    line_x = np.linspace(x_min, x_max, 100)

    ax.plot(line_x, line_x, color="0.8", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    if len(x) >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, deg=1)
        ax.plot(line_x, slope * line_x + intercept, color="0.55", linewidth=0.9, zorder=2)
        label_x = x_min + 0.24 * (x_max - x_min)
        label_y = slope * label_x + intercept + 0.03 * (y_max - y_min)
        ax.text(label_x, label_y, "linear fit", color="0.5", fontsize=6.5, va="bottom", ha="left")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def _annotate_points(ax: plt.Axes, data: pd.DataFrame, x_col: str, y_col: str) -> None:
    texts = []
    x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    offsets = {
        "B0": (0.02, 0.07),
        "B0_proj1024": (-0.12, -0.08),
        "B0_uf5": (0.04, -0.09),
        "B0_uf6": (-0.10, 0.09),
        "B0_uf7": (0.10, 0.10),
        "B0plus": (-0.12, 0.04),
        "B1": (0.05, -0.13),
        "B2": (-0.08, -0.12),
        "B4": (0.08, 0.13),
        "B5a_seg_spatial": (-0.16, 0.12),
        "B5b_seg_semantic": (-0.15, -0.03),
        "B5c_seg_continuous": (0.06, -0.08),
    }
    fallback = [(0.05, 0.05), (-0.08, 0.06), (0.07, -0.07), (-0.10, -0.05)]

    for idx, row in data.reset_index(drop=True).iterrows():
        dx_frac, dy_frac = offsets.get(row["config"], fallback[idx % len(fallback)])
        text = ax.text(
            row[x_col] + dx_frac * x_span,
            row[y_col] + dy_frac * y_span,
            row["config"],
            ha="center",
            va="center",
            fontsize=5.2,
            color="0.18",
            zorder=6,
        )
        texts.append(text)

    if adjust_text is not None:
        adjust_text(
            texts,
            ax=ax,
            x=data[x_col].to_numpy(),
            y=data[y_col].to_numpy(),
            ensure_inside_axes=True,
            expand=(1.25, 1.45),
            force_text=(0.18, 0.32),
            force_static=(0.08, 0.16),
            force_pull=(0.01, 0.01),
            prevent_crossings=True,
            iter_lim=300,
        )


def plot(data: pd.DataFrame) -> None:
    _configure_matplotlib()
    configs = data["config"].tolist()
    colors = _palette(configs)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.75))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.30, wspace=0.32)
    fig.patch.set_facecolor("none")

    handles = []
    labels = []
    for ax, (metric_name, metric) in zip(axes, METRICS.items(), strict=True):
        x_mean = f"{metric_name}_x_mean"
        y_mean = f"{metric_name}_y_mean"
        x_std = f"{metric_name}_x_std"
        y_std = f"{metric_name}_y_std"

        ax.set_facecolor("none")
        xlim, ylim = _axis_limits(data, x_mean, y_mean)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        _plot_reference_lines(ax, data[x_mean].to_numpy(), data[y_mean].to_numpy())

        for _, row in data.iterrows():
            config = row["config"]
            single_seed = int(row["n_seeds"]) == 1
            color = colors[config]
            label = f"{config}{'*' if single_seed else ''}"

            if single_seed:
                handle = ax.scatter(
                    row[x_mean],
                    row[y_mean],
                    s=34,
                    facecolors="white",
                    edgecolors=[color],
                    linewidths=1.2,
                    zorder=5,
                )
            else:
                ax.errorbar(
                    row[x_mean],
                    row[y_mean],
                    xerr=row[x_std],
                    yerr=row[y_std],
                    fmt="o",
                    markersize=4.8,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    ecolor="black",
                    elinewidth=0.7,
                    capsize=1.8,
                    capthick=0.7,
                    zorder=5,
                )
                handle = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=5,
                )

            if metric_name == "i2t":
                handles.append(handle)
                labels.append(label)

        _annotate_points(ax, data, x_mean, y_mean)

        ax.set_title(metric["title"])
        ax.set_xlabel(metric["x_label"])
        ax.set_ylabel(metric["y_label"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(width=0.7, length=3)
        ax.grid(axis="both", color="0.9", linewidth=0.5, zorder=0)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    fig.text(0.5, 0.01, "* single seed", ha="center", va="bottom", color="0.35", fontsize=7)

    fig.savefig(SAVE_FIG_DIR / "02_r1_vs_eccv_mapr.pdf")
    fig.savefig(SAVE_FIG_DIR / "02_r1_vs_eccv_mapr.png", dpi=300)
    plt.close(fig)


def main() -> None:
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()

    print("Detected ECCV/COCO configs:")
    for _, row in data.iterrows():
        print(f"  {row['config']}: n_seeds={int(row['n_seeds'])}")

    data.to_csv(SAVE_DATA_DIR / "02_r1_vs_eccv_mapr_data.csv", index=False)
    plot(data)


if __name__ == "__main__":
    main()

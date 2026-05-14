from __future__ import annotations

import os
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import DEFAULT_CSV_PATH, SAVE_DATA_DIR, SAVE_FIG_DIR, aggregate_seeds, load_runs

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover
    adjust_text = None


EXCLUDE_LIST = ["B0v2", "B0plus_fixed", "B5_seg", "B0_proj1024"]
OUTPUT_STEM = "02_coco1k_vs_eccv_mapr"
CONFIG_ORDER = [
    "B0",
    "B0plus",
    "B0_uf5",
    "B0_uf6",
    "B0_uf7",
    "B1",
    "B2",
    "B4",
    "B5a_seg_spatial",
    "B5b_seg_semantic",
    "B5c_seg_continuous",
]

METRICS = {
    "i2t": {
        "x": "summary/test/coco_1k_r1_i2t",
        "y": "summary/test/eccv_map_at_r_i2t",
        "x_label": "Original COCO 1K R@1 (I2T) [%]",
        "y_label": "ECCV mAP@R (I2T) [%]",
        "title": "Image-to-Text",
    },
    "t2i": {
        "x": "summary/test/coco_1k_r1_t2i",
        "y": "summary/test/eccv_map_at_r_t2i",
        "x_label": "Original COCO 1K R@1 (T2I) [%]",
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
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _scale_for_percent(df: pd.DataFrame, col: str) -> float:
    return 100.0 if df[col].dropna().max() <= 1.5 else 1.0


def _ordered_configs(configs: list[str]) -> list[str]:
    ordered = [config for config in CONFIG_ORDER if config in configs]
    extras = sorted(config for config in configs if config not in ordered)
    return ordered + extras


def build_data(csv_path=DEFAULT_CSV_PATH) -> pd.DataFrame:
    df = load_runs(csv_path, EXCLUDE_LIST)
    required = ["config/dataset", *VALUE_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

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
                scale = _scale_for_percent(filtered, col)
                mean = run[f"{col}_mean"] * scale
                std = run[f"{col}_std"] * scale
                row[f"{metric_name}_{axis_name}_mean"] = mean
                row[f"{metric_name}_{axis_name}_std"] = std if pd.notna(std) else np.nan
        rows.append(row)

    data = pd.DataFrame(rows)
    config_order = _ordered_configs(data["config"].dropna().astype(str).tolist())
    data["order"] = data["config"].map({config: idx for idx, config in enumerate(config_order)})
    return data.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def _axis_limits(data: pd.DataFrame, x_col: str, y_col: str) -> tuple[tuple[float, float], tuple[float, float]]:
    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()
    x_pad = max((x.max() - x.min()) * 0.30, 0.8)
    y_pad = max((y.max() - y.min()) * 0.35, 0.8)
    return (x.min() - x_pad, x.max() + x_pad), (y.min() - y_pad, y.max() + y_pad)


def _plot_trend(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    if len(x) < 2 or np.ptp(x) <= 0:
        return
    x_min, x_max = ax.get_xlim()
    line_x = np.linspace(x_min, x_max, 100)
    slope, intercept = np.polyfit(x, y, deg=1)
    ax.plot(
        line_x,
        slope * line_x + intercept,
        color="0.45",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=1,
        label="Trend across configs",
    )
    label_x = x_min + 0.05 * (x_max - x_min)
    label_y = slope * label_x + intercept
    ax.text(
        label_x,
        label_y,
        "trend across configs",
        color="0.4",
        fontsize=7,
        ha="left",
        va="bottom",
        rotation=0,
    )


def _annotate_points(ax: plt.Axes, data: pd.DataFrame, x_col: str, y_col: str) -> None:
    texts = []
    for _, row in data.iterrows():
        text = ax.text(
            row[x_col],
            row[y_col],
            row["config"],
            ha="center",
            va="center",
            fontsize=5.5,
            color="0.15",
            zorder=5,
        )
        texts.append(text)

    if adjust_text is not None:
        adjust_text(
            texts,
            ax=ax,
            x=data[x_col].to_numpy(),
            y=data[y_col].to_numpy(),
            ensure_inside_axes=True,
            expand=(1.2, 1.35),
            force_text=(0.12, 0.24),
            force_static=(0.08, 0.14),
            force_pull=(0.01, 0.01),
            prevent_crossings=True,
            iter_lim=250,
        )


def plot(data: pd.DataFrame) -> None:
    _configure_matplotlib()
    configs = data["config"].tolist()
    colors = dict(zip(configs, sns.color_palette("colorblind", n_colors=len(configs)), strict=True))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.26, wspace=0.30)
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
        _plot_trend(ax, data[x_mean].to_numpy(), data[y_mean].to_numpy())

        for _, row in data.iterrows():
            config = row["config"]
            single_seed = int(row["n_seeds"]) == 1
            color = colors[config]
            label = f"{config}{'*' if single_seed else ''}"
            if single_seed:
                handle = ax.scatter(
                    row[x_mean],
                    row[y_mean],
                    s=36,
                    facecolors="white",
                    edgecolors=[color],
                    linewidths=1.2,
                    zorder=4,
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
                    zorder=4,
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
        ax.grid(axis="both", color="0.9", linewidth=0.5, zorder=0)
        ax.tick_params(width=0.7, length=3)

    fig.suptitle("Original COCO 1K Retrieval vs ECCV Reannotated Retrieval", fontsize=11, y=0.965)
    fig.text(
        0.5,
        0.875,
        "The dashed line is a fitted trend across configurations, not a parity line; ECCV mAP@R uses expanded positives and a different ranking metric.",
        ha="center",
        va="center",
        fontsize=8,
        color="0.35",
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=False,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    fig.text(0.5, 0.018, "* single seed", ha="center", va="bottom", color="0.35", fontsize=7)

    fig.savefig(SAVE_FIG_DIR / f"{OUTPUT_STEM}.pdf")
    fig.savefig(SAVE_FIG_DIR / f"{OUTPUT_STEM}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()

    print("Detected COCO configs with COCO 1K R@1 and ECCV mAP@R:")
    for _, row in data.iterrows():
        print(
            f"  {row['config']}: n_seeds={int(row['n_seeds'])}, "
            f"I2T=({row['i2t_x_mean']:.2f}, {row['i2t_y_mean']:.2f}), "
            f"T2I=({row['t2i_x_mean']:.2f}, {row['t2i_y_mean']:.2f})"
        )

    data.to_csv(SAVE_DATA_DIR / f"{OUTPUT_STEM}_data.csv", index=False)
    plot(data)


if __name__ == "__main__":
    main()

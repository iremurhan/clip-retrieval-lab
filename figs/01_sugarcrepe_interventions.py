from __future__ import annotations

import numpy as np
import pandas as pd

from helpers import (
    DEFAULT_CSV_PATH,
    SAVE_DATA_DIR,
    SAVE_FIG_DIR,
    SC_CATEGORIES,
    SINGLE_SEED_FOOTNOTE,
    aggregate_by_config,
    filter_sugarcrepe_coco,
    get_palette,
    load_runs,
    print_color_report,
    setup_thesis_style,
    single_seed_label,
    sugarcrepe_aggregate_to_long,
)


CSV_PATH = DEFAULT_CSV_PATH

FAMILIES = [
    {
        "name": "sam_fusion",
        "output_stem": "01_sugarcrepe_sam_fusion",
        "series": ["Base-min", "Sam-Gate", "Sam-XAttn", "Sam-Concat", "Sam-Skip"],
    },
    {
        "name": "patch_segment",
        "output_stem": "01B_sugarcrepe_patch_segment",
        "series": ["Base-min", "Seg-Spatial", "Seg-Semantic", "Seg-Geom"],
    },
    {
        "name": "loss_data",
        "output_stem": "01C_sugarcrepe_loss_data",
        "series": ["Base-min", "Loss-SigLIP", "HN-Syntactic", "Aux-ObjCls"],
    },
]


def _family_order(labels: set[str], requested: list[str]) -> list[str]:
    return [label for label in requested if label in labels]


def _color_by_config(configs: list[str]) -> dict[str, object]:
    non_baseline = [config for config in configs if config != "Base-min"]
    palette = get_palette(len(non_baseline))
    colors = dict(zip(non_baseline, palette))
    if "Base-min" in configs:
        colors["Base-min"] = (0.18, 0.18, 0.18)
    return colors


def _axis_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 100.0
    ymin = max(0.0, float(finite.min()) - 2.0)
    ymax = min(100.0, float(finite.max()) + 1.0)
    if ymax - ymin < 6.0:
        pad = (6.0 - (ymax - ymin)) / 2.0
        ymin = max(0.0, ymin - pad)
        ymax = min(100.0, ymax + pad)
    return ymin, ymax


def plot_family(data: pd.DataFrame, output_stem: str) -> None:
    import matplotlib.pyplot as plt

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    configs = list(data["run_id"].drop_duplicates())
    if not configs:
        raise ValueError(f"No configurations available for {output_stem}.")

    color_by_config = _color_by_config(configs)
    print_color_report(color_by_config)

    n_by_config = data.groupby("run_id", sort=False)["n_seeds"].max().to_dict()
    display_by_config = data.groupby("run_id", sort=False)["display_label"].first().to_dict()
    legend_labels = [
        single_seed_label(display_by_config.get(run_id, run_id), n_by_config[run_id])
        for run_id in configs
    ]

    category_order = [category for category, _label in SC_CATEGORIES]
    category_labels = [label for _category, label in SC_CATEGORIES]
    category_data = data[data["category"].isin(category_order)]
    category_values = category_data["mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.30)

    x = np.arange(len(category_order), dtype=float)
    width = min(0.82 / len(configs), 0.16)
    offsets = (np.arange(len(configs)) - (len(configs) - 1) / 2.0) * width
    error_kw = {"capsize": 2.5, "capthick": 0.9, "ecolor": "black", "elinewidth": 0.9}

    for config_idx, run_id in enumerate(configs):
        sub = data[data["run_id"].eq(run_id)].set_index("category").reindex(category_order)
        means = sub["mean"].to_numpy(dtype=float)
        stds = sub["std"].to_numpy(dtype=float)
        single = int(n_by_config[run_id]) == 1
        yerr = None if single else np.nan_to_num(stds, nan=0.0)
        ax.bar(
            x + offsets[config_idx],
            means,
            width=width * 0.92,
            color=color_by_config[run_id],
            edgecolor="white",
            linewidth=0.55,
            yerr=yerr,
            error_kw=error_kw,
            zorder=3,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels)
    ax.set_ylim(*_axis_limits(category_values))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis="y", which="minor", color="0.82", linewidth=0.45, alpha=0.3, zorder=0)
    ax.grid(axis="y", which="major", color="0.84", linewidth=0.5, alpha=0.3, zorder=0)
    ax.tick_params(width=0.7, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_by_config[run_id]) for run_id in configs]
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=min(len(configs), 5),
        frameon=False,
        handlelength=1.2,
        columnspacing=0.9,
    )
    fig.text(
        0.5,
        0.025,
        f"{SINGLE_SEED_FOOTNOTE} y-axis truncated to show differences.",
        ha="center",
        va="center",
        fontsize=7.5,
        color="0.35",
    )

    fig.savefig(SAVE_FIG_DIR / f"{output_stem}.pdf")
    fig.savefig(SAVE_FIG_DIR / f"{output_stem}.png", dpi=300)
    plt.close(fig)


def print_report(data: pd.DataFrame, family: dict) -> None:
    print(f"SugarCrepe family ({family['name']}):")
    print("Actually plotted (label, dataset, n_seeds):")
    for run_id, sub in data.groupby("run_id", sort=False):
        n = int(sub["n_seeds"].max())
        print(f"  ({run_id}, coco, {n})")


def build_family_data(df: pd.DataFrame, family: dict) -> pd.DataFrame:
    labels = set(df["thesis_label"].dropna().astype(str))
    config_order = _family_order(labels, family["series"])
    if not config_order:
        raise ValueError(f"No SugarCrepe rows found for family {family['name']}.")
    subset = df[df["thesis_label"].isin(config_order)].copy()
    value_cols = [f"sc_{category}" for category, _label in SC_CATEGORIES] + ["sc_overall"]
    aggregate = aggregate_by_config(subset, value_cols)
    return sugarcrepe_aggregate_to_long(aggregate, config_order)


def main() -> None:
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    setup_thesis_style()

    df = load_runs(CSV_PATH)
    df = filter_sugarcrepe_coco(df)

    for family in FAMILIES:
        data = build_family_data(df, family)
        print_report(data, family)
        data.to_csv(SAVE_DATA_DIR / f"{family['output_stem']}_data.csv", index=False)
        plot_family(data, family["output_stem"])


if __name__ == "__main__":
    main()

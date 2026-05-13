from __future__ import annotations

import json
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
    EXCLUDE,
    SAVE_DATA_DIR,
    SAVE_FIG_DIR,
    SAVE_TABLE_DIR,
    _to_percent,
    latex_escape,
    load_runs,
    retrieval_config_order,
)


OOD_CACHE_DIR = Path("results/cache/ood_eval")
OUTPUT_STEM = "05_cross_dataset_ood"
DATASET_LABELS = {"flickr30k": "Flickr30K", "coco": "COCO 5K"}
OPPOSITE = {"flickr30k": "coco", "coco": "flickr30k"}
METRICS = [
    ("r1_i2t", "R@1 I2T"),
    ("r5_i2t", "R@5 I2T"),
    ("r10_i2t", "R@10 I2T"),
    ("r1_t2i", "R@1 T2I"),
    ("r5_t2i", "R@5 T2I"),
    ("r10_t2i", "R@10 T2I"),
]
SOURCE_COLUMNS = {
    "flickr30k": {
        "r1_i2t": "summary/test/r1_i2t",
        "r5_i2t": "summary/test/r5_i2t",
        "r10_i2t": "summary/test/r10_i2t",
        "r1_t2i": "summary/test/r1_t2i",
        "r5_t2i": "summary/test/r5_t2i",
        "r10_t2i": "summary/test/r10_t2i",
    },
    "coco": {
        "r1_i2t": "summary/test/coco_5k_r1_i2t",
        "r5_i2t": "summary/test/coco_5k_r5_i2t",
        "r10_i2t": "summary/test/coco_5k_r10_i2t",
        "r1_t2i": "summary/test/coco_5k_r1_t2i",
        "r5_t2i": "summary/test/coco_5k_r5_t2i",
        "r10_t2i": "summary/test/coco_5k_r10_t2i",
    },
}


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
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def metric_value(metrics: dict, eval_dataset: str, metric: str) -> float:
    if eval_dataset == "coco":
        coco_key = f"coco_5k_{metric}"
        if coco_key in metrics:
            return _to_percent(metrics[coco_key])
    return _to_percent(metrics.get(metric, np.nan))


def read_ood_cache(cache_dir: Path = OOD_CACHE_DIR) -> pd.DataFrame:
    paths = sorted(cache_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No OOD cache JSON files found under {cache_dir}. "
            "Run scripts/eval/run_ood_eval_local.py first."
        )

    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        for metric, metric_label in METRICS:
            rows.append(
                {
                    "run_id": payload["run_id"],
                    "train_dataset": payload["train_dataset"],
                    "eval_dataset": payload["eval_dataset"],
                    "seed": int(payload.get("seed", -1)),
                    "condition": "OOD",
                    "metric": metric,
                    "metric_label": metric_label,
                    "mean": metric_value(metrics, payload["eval_dataset"], metric),
                }
            )
    return pd.DataFrame(rows)


def build_id_rows(csv_path: str | Path, needed: pd.DataFrame) -> pd.DataFrame:
    runs = load_runs(csv_path, EXCLUDE).copy()
    runs["train_dataset"] = runs["config/dataset"].replace({"flickr": "flickr30k"})
    rows = []
    wanted = needed[["run_id", "train_dataset", "seed"]].drop_duplicates()
    for _, item in wanted.iterrows():
        subset = runs[
            runs["config/run_id"].eq(item["run_id"])
            & runs["train_dataset"].eq(item["train_dataset"])
            & pd.to_numeric(runs["config/seed"], errors="coerce").eq(item["seed"])
        ]
        if subset.empty:
            continue
        record = subset.iloc[0]
        for metric, metric_label in METRICS:
            col = SOURCE_COLUMNS[item["train_dataset"]][metric]
            rows.append(
                {
                    "run_id": item["run_id"],
                    "train_dataset": item["train_dataset"],
                    "eval_dataset": item["train_dataset"],
                    "seed": int(item["seed"]),
                    "condition": "ID",
                    "metric": metric,
                    "metric_label": metric_label,
                    "mean": _to_percent(record.get(col, np.nan)),
                }
            )
    return pd.DataFrame(rows)


def aggregate(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["run_id", "train_dataset", "eval_dataset", "condition", "metric", "metric_label"], sort=False)
    out = grouped["mean"].agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"count": "n_seeds"})
    return out


def build_data(csv_path: str | Path = DEFAULT_CSV_PATH, cache_dir: Path = OOD_CACHE_DIR) -> pd.DataFrame:
    ood = read_ood_cache(cache_dir)
    id_rows = build_id_rows(csv_path, ood)
    data = aggregate(pd.concat([id_rows, ood], ignore_index=True))
    config_order = retrieval_config_order(data["run_id"].unique())
    data["config_order"] = data["run_id"].map({run_id: idx for idx, run_id in enumerate(config_order)})
    data.attrs["config_order"] = config_order
    return data.sort_values(["config_order", "train_dataset", "condition", "metric"]).reset_index(drop=True)


def fmt_cell(mean: float, std: float, n_seeds: int) -> str:
    if pd.isna(mean):
        return "--"
    text = f"{float(mean):.1f}"
    if n_seeds >= 2 and not pd.isna(std):
        return rf"{text} $\pm$ {float(std):.1f}"
    return text + ("*" if n_seeds == 1 else "")


def write_table(data: pd.DataFrame) -> None:
    SAVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lookup = data.set_index(["run_id", "train_dataset", "condition", "metric"])
    lines = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Run & Train & Eval & R@1 I2T & R@5 I2T & R@10 I2T & R@1 T2I & R@5 T2I & R@10 T2I \\",
        r"\midrule",
    ]
    for run_id in data.attrs["config_order"]:
        for train_dataset in ["flickr30k", "coco"]:
            if data[data["run_id"].eq(run_id) & data["train_dataset"].eq(train_dataset)].empty:
                continue
            for condition in ["ID", "OOD"]:
                sub = data[data["run_id"].eq(run_id) & data["train_dataset"].eq(train_dataset) & data["condition"].eq(condition)]
                if sub.empty:
                    continue
                eval_label = DATASET_LABELS[train_dataset] if condition == "ID" else DATASET_LABELS[OPPOSITE[train_dataset]]
                cells = [latex_escape(run_id), latex_escape(DATASET_LABELS[train_dataset]), latex_escape(f"{condition}: {eval_label}")]
                for metric, _label in METRICS:
                    row = lookup.loc[(run_id, train_dataset, condition, metric)]
                    cells.append(fmt_cell(row["mean"], row["std"], int(row["n_seeds"])))
                lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (SAVE_TABLE_DIR / f"{OUTPUT_STEM}.tex").write_text("\n".join(lines), encoding="utf-8")


def display_run_id(run_id: str) -> str:
    if run_id.startswith("B5d_multistream_"):
        return "B5d_multistream\n" + run_id.removeprefix("B5d_multistream_")
    if run_id.startswith("B5") and "_seg_" in run_id:
        prefix, suffix = run_id.split("_seg_", maxsplit=1)
        return f"{prefix}_seg\n{suffix}"
    return run_id


def matrix_for(data: pd.DataFrame, train_dataset: str, configs: list[str]) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        run_id
        for run_id in configs
        if not data[data["run_id"].eq(run_id) & data["train_dataset"].eq(train_dataset)].empty
    ]
    columns = [f"ID\n{label}" for _metric, label in METRICS] + [f"OOD\n{label}" for _metric, label in METRICS]
    values = np.full((len(rows), len(columns)), np.nan)
    stds = np.full_like(values, np.nan)
    n_seeds = np.zeros_like(values, dtype=int)
    lookup = data.set_index(["run_id", "train_dataset", "condition", "metric"])

    for row_idx, run_id in enumerate(rows):
        for offset, condition in enumerate(["ID", "OOD"]):
            for col_idx, (metric, _label) in enumerate(METRICS):
                key = (run_id, train_dataset, condition, metric)
                if key not in lookup.index:
                    continue
                record = lookup.loc[key]
                out_col = offset * len(METRICS) + col_idx
                values[row_idx, out_col] = record["mean"]
                stds[row_idx, out_col] = record["std"]
                n_seeds[row_idx, out_col] = int(record["n_seeds"])
    return rows, columns, values, stds, n_seeds


def annotate(ax: plt.Axes, values: np.ndarray, stds: np.ndarray, n_seeds: np.ndarray) -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isfinite(values[i, j]):
                ax.text(j, i, "--", ha="center", va="center", fontsize=5.2, color="0.4")
                continue
            suffix = "*" if n_seeds[i, j] == 1 else ""
            ax.text(j, i - 0.10, f"{values[i, j]:.1f}{suffix}", ha="center", va="center", fontsize=5.2, color="0.05")
            if n_seeds[i, j] >= 2 and np.isfinite(stds[i, j]):
                ax.text(j, i + 0.22, f"+/- {stds[i, j]:.1f}", ha="center", va="center", fontsize=3.9, color="0.25")


def plot(data: pd.DataFrame) -> None:
    configure_matplotlib()
    configs = data.attrs.get("config_order", RETRIEVAL_CONFIG_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.11, right=0.93, top=0.91, bottom=0.14, wspace=0.28)
    fig.patch.set_facecolor("none")

    finite = data["mean"].dropna()
    vmin = float(finite.min()) if not finite.empty else 0.0
    vmax = float(finite.max()) if not finite.empty else 100.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.92")
    image = None

    for ax, train_dataset in zip(axes, ["flickr30k", "coco"], strict=True):
        rows, columns, values, stds, n_seeds = matrix_for(data, train_dataset, configs)
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{DATASET_LABELS[train_dataset]}-trained checkpoints", pad=8)
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([display_run_id(run_id) for run_id in rows])
        ax.axvline(len(METRICS) - 0.5, color="white", linewidth=2.0)
        ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        annotate(ax, values, stds, n_seeds)

    if image is not None:
        cax = fig.add_axes([0.95, 0.23, 0.015, 0.54])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("Recall (%)", fontsize=8)
        cbar.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    fig.text(0.5, 0.04, "* single seed; ID is the training dataset test split, OOD is the opposite dataset test split", ha="center", fontsize=7, color="0.35")

    SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAVE_FIG_DIR / f"{OUTPUT_STEM}.pdf")
    fig.savefig(SAVE_FIG_DIR / f"{OUTPUT_STEM}.png", dpi=300)
    plt.close(fig)


def print_report(data: pd.DataFrame) -> None:
    print("Cross-dataset OOD rows detected:")
    summary = data.groupby(["run_id", "train_dataset", "condition"], sort=False)["n_seeds"].max().reset_index()
    for _, row in summary.iterrows():
        print(f"  {row['run_id']} train={row['train_dataset']} {row['condition']} n={int(row['n_seeds'])}")


def main() -> None:
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()
    print_report(data)
    data.to_csv(SAVE_DATA_DIR / f"{OUTPUT_STEM}_data.csv", index=False)
    write_table(data)
    plot(data)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from helpers import CACHE_DIR as BASE_CACHE_DIR
from helpers import SAVE_DATA_DIR, SAVE_TABLE_DIR, label_cache_frame, latex_escape, retrieval_config_order


CACHE_DIR = BASE_CACHE_DIR / "missing_positive_stats" / "stats"
OUTPUT_TEX = SAVE_TABLE_DIR / "06_missing_positive_stats.tex"
OUTPUT_CSV = SAVE_DATA_DIR / "06_missing_positive_data.csv"
METRICS = [
    ("eccv_top1_recovered_i2t", "top1_recovered I2T"),
    ("eccv_top1_recovered_t2i", "top1_recovered T2I"),
    ("eccv_top5_neighborhood_i2t", "top5_neigh I2T"),
    ("eccv_top5_neighborhood_t2i", "top5_neigh T2I"),
]


def load_stats(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    paths = sorted(cache_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No missing-positive stats JSON files found under {cache_dir}. Run scripts/eval/eval_missing_positive_stats.py first.")
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "run_id": payload["run_id"],
            "dataset": payload.get("dataset", "coco"),
            "seed": int(payload.get("seed", -1)),
        }
        row.update(payload.get("metrics", {}))
        rows.append(row)
    df = label_cache_frame(pd.DataFrame(rows), id_col="run_id", context="missing-positive cache")
    df["raw_run_id"] = df["run_id"]
    df["run_id"] = df["thesis_label"]
    return df


def fmt(mean: float, std: float, n: int) -> str:
    if pd.isna(mean):
        return "--"
    if n >= 2 and not pd.isna(std):
        return rf"{mean:.1f} $\pm$ {std:.1f}"
    return f"{mean:.1f}*"


def build_table(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["run_id", "latex_label"], sort=False)[[m for m, _ in METRICS]].agg(["mean", "std", "count"])
    rows = []
    for run_id in retrieval_config_order(data["run_id"].unique()):
        matches = [idx for idx in grouped.index if idx[0] == run_id]
        if not matches:
            continue
        idx = matches[0]
        row = {"run_id": run_id, "latex_label": idx[1]}
        for metric, _label in METRICS:
            row[f"{metric}_mean"] = grouped.loc[idx, (metric, "mean")]
            row[f"{metric}_std"] = grouped.loc[idx, (metric, "std")]
            row[f"{metric}_n"] = int(grouped.loc[idx, (metric, "count")])
        rows.append(row)
    return pd.DataFrame(rows)


def write_latex(table: pd.DataFrame) -> None:
    SAVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Configuration & " + " & ".join(label for _metric, label in METRICS) + r" \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = [latex_escape(row["latex_label"])]
        for metric, _label in METRICS:
            cells.append(fmt(row[f"{metric}_mean"], row[f"{metric}_std"], int(row[f"{metric}_n"])))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def print_headlines(table: pd.DataFrame) -> None:
    print("Highest ECCV top1_recovered values:")
    for metric, label in METRICS[:2]:
        ranked = table.sort_values(f"{metric}_mean", ascending=False).head(5)
        print(f"  {label}:")
        for _, row in ranked.iterrows():
            print(f"    {row['run_id']}: {row[f'{metric}_mean']:.1f}")

    b4 = table[table["run_id"].eq("Aux-ObjCls")]
    if b4.empty:
        print("Aux-ObjCls check: no rows found.")
    else:
        i2t = float(b4.iloc[0]["eccv_top5_neighborhood_i2t_mean"])
        t2i = float(b4.iloc[0]["eccv_top5_neighborhood_t2i_mean"])
        med_i2t = float(table["eccv_top5_neighborhood_i2t_mean"].median())
        med_t2i = float(table["eccv_top5_neighborhood_t2i_mean"].median())
        status = "noteworthy" if i2t > med_i2t or t2i > med_t2i else "not notably above median"
        print(f"B4 top5_neighborhood check: {status} (I2T={i2t:.1f}, T2I={t2i:.1f}).")


def main() -> None:
    data = load_stats()
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_CSV, index=False)
    table = build_table(data)
    write_latex(table)
    print_headlines(table)


if __name__ == "__main__":
    main()

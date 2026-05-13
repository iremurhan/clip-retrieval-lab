from __future__ import annotations

import re

import pandas as pd

from helpers import (
    DEFAULT_CSV_PATH,
    SAVE_DATA_DIR,
    aggregate_by_config,
    filter_sugarcrepe_coco,
    load_runs,
    plot_sugarcrepe_panels,
    print_family_report,
    SC_CATEGORIES,
    setup_thesis_style,
    split_by_baseline,
    sugarcrepe_aggregate_to_long,
)


CSV_PATH = DEFAULT_CSV_PATH
OUTPUT_STEM = "02_sugarcrepe_capacity"


def sort_configs(df: pd.DataFrame) -> list[str]:
    unfreeze = (
        df.groupby("config/run_id", sort=False)["config/unfreeze_layers"]
        .first()
        .pipe(pd.to_numeric, errors="coerce")
        .to_dict()
    )
    run_ids = list(df["config/run_id"].dropna().unique())

    def key(run_id: str):
        if run_id == "B0":
            return (0, 0, run_id)
        match = re.fullmatch(r"B0_uf(\d+)", run_id)
        if match:
            depth = unfreeze.get(run_id)
            if pd.isna(depth):
                depth = int(match.group(1))
            return (1, int(depth), run_id)
        if run_id == "B0_proj1024":
            return (3, 0, run_id)
        return (2, 0, run_id)

    return sorted(run_ids, key=key)


def main() -> None:
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    setup_thesis_style()

    df = load_runs(CSV_PATH)
    df = filter_sugarcrepe_coco(df)
    b0plus_df, b0_df = split_by_baseline(df, include_b0_in_interventions=False)
    print_family_report(df, b0plus_df, b0_df)

    if b0_df.empty:
        raise ValueError("No B0-family COCO SugarCrepe rows found after filtering.")

    value_cols = [f"sc_{category}" for category, _ in SC_CATEGORIES] + ["sc_overall"]
    aggregate = aggregate_by_config(b0_df, value_cols)
    config_order = sort_configs(b0_df)
    data = sugarcrepe_aggregate_to_long(aggregate, config_order)
    data.to_csv(SAVE_DATA_DIR / f"{OUTPUT_STEM}_data.csv", index=False)

    plot_sugarcrepe_panels(data, OUTPUT_STEM)


if __name__ == "__main__":
    main()

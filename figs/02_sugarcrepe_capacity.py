from __future__ import annotations

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
    THESIS_LABEL_ORDER,
)


CSV_PATH = DEFAULT_CSV_PATH
OUTPUT_STEM = "02_sugarcrepe_capacity"


def sort_configs(df: pd.DataFrame) -> list[str]:
    labels = set(df["thesis_label"].dropna().astype(str))
    ordered = [label for label in THESIS_LABEL_ORDER if label in labels]
    return ordered + sorted(labels - set(ordered))


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

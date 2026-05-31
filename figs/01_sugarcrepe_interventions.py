from __future__ import annotations

from helpers import (
    DEFAULT_CSV_PATH,
    MAIN_INTERVENTION_GROUPS,
    SAVE_DATA_DIR,
    aggregate_by_config,
    filter_sugarcrepe_coco,
    load_runs,
    plot_sugarcrepe_panels,
    SC_CATEGORIES,
    setup_thesis_style,
    sugarcrepe_aggregate_to_long,
    THESIS_LABEL_ORDER,
)


CSV_PATH = DEFAULT_CSV_PATH
OUTPUT_STEM = "01_sugarcrepe_interventions"


def sort_configs(df):
    labels = set(df["thesis_label"].dropna().astype(str))
    ordered = [label for label in THESIS_LABEL_ORDER if label in labels]
    return ordered + sorted(labels - set(ordered))


def print_report(intervention_df, dropped) -> None:
    print("Main-intervention SugarCrepe configs (COCO):")
    counts = intervention_df.groupby(["thesis_label", "intervention_group"], sort=False).size()
    if counts.empty:
        print("  (none)")
    for (label, group), n in counts.items():
        print(f"  {label:14s} {group:34s} n_seeds={int(n)}")
    if dropped:
        print("Excluded (non-intervention groups):", ", ".join(sorted(dropped)))


def main() -> None:
    SAVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    setup_thesis_style()

    df = load_runs(CSV_PATH)
    df = filter_sugarcrepe_coco(df)

    # Main intervention variants only: exclude capacity variants (unfreezing_depth,
    # projection_capacity) and the baseline references (baseline, intra_regularization).
    is_main = df["intervention_group"].isin(MAIN_INTERVENTION_GROUPS)
    dropped = set(df.loc[~is_main, "display_label"].dropna().astype(str))
    intervention_df = df.loc[is_main].copy()
    print_report(intervention_df, dropped)

    if intervention_df.empty:
        raise ValueError("No main-intervention COCO SugarCrepe rows found after filtering.")

    value_cols = [f"sc_{category}" for category, _ in SC_CATEGORIES] + ["sc_overall"]
    aggregate = aggregate_by_config(intervention_df, value_cols)
    config_order = sort_configs(intervention_df)
    data = sugarcrepe_aggregate_to_long(aggregate, config_order)
    data.to_csv(SAVE_DATA_DIR / f"{OUTPUT_STEM}_data.csv", index=False)

    plot_sugarcrepe_panels(data, OUTPUT_STEM, save_colorblind_check=True)


if __name__ == "__main__":
    main()

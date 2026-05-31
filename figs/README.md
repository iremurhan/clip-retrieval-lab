# Figure Scripts

This directory contains the source scripts for thesis figures and diagnostic
plots. Generated CSV caches and rendered figures are intentionally not tracked.

Figure scripts read the cleaned wide table produced by
`scripts/util/export_clean_results.py` by default:
`$CLIP_RETRIEVAL_ARTIFACT_ROOT/clean/clean_results_wide.csv`. Override with
`CLEAN_RESULTS_WIDE_PATH` when needed.

All plotted/table labels are resolved through `configs/thesis_label_map.yaml`.
Figure scripts should use `display_label`, `thesis_label`, or `latex_label`,
not raw W&B/config run IDs.

| Script | Purpose | Thesis placement |
| --- | --- | --- |
| `00_methodology_overview.py` | Draws the high-level method overview for the experimental pipeline. | Thesis-useful |
| `01_sugarcrepe_interventions.py` | Compares SugarCrepe compositional scores for intra-regularized intervention configs. | Thesis-useful |
| `02_sugarcrepe_capacity.py` | Summarizes SugarCrepe capacity/variant comparisons across model configurations. | Thesis-useful |
| `03_unfreezing_depth.py` | Visualizes the effect of unfreezing more CLIP vision layers. | Thesis-useful |
| `04_retrieval_grouped_bar.py` | Shows grouped retrieval R@K results across datasets/configurations. | Thesis-useful |
| `04_retrieval_heatmap.py` | Shows retrieval metrics as a configuration-by-metric heatmap. | Appendix candidate |
| `04_retrieval_slope_chart.py` | Shows paired retrieval changes between related settings/datasets. | Appendix candidate |
| `05_cross_dataset_ood.py` | Summarizes cross-dataset out-of-domain retrieval evaluation. | Thesis-useful |
| `06_missing_positive_table.py` | Builds the missing-positive analysis table. | Appendix candidate |
| `07_alignment_uniformity.py` | Plots alignment and uniformity diagnostics. | Appendix candidate |
| `08_patch_vs_cls.py` | Compares patch-level and CLS-level diagnostic behavior. | Debugging-only unless cited as diagnostic evidence |
| `09_mmvp_vlm.py` | Plots MMVP-VLM post-hoc compositional benchmark results. | Thesis-useful |
| `10_intra_modal_flow.py` | Draws the intra-modal training objective schematic. | Thesis-useful |
| `11_retrieval_by_bucket.py` | Plots retrieval R@1 by caption length/complexity buckets. | Appendix candidate |
| `12_sam_fusion_variants.py` | Draws SAM fusion variant schematics. | Thesis-useful |
| `helpers.py` | Shared loading, aggregation, style, color, and table helpers used by figure scripts. |

Archived exploratory scripts live under `archive/`. The current archived
candidate, `02_coco_false_positive_gap.py`, checks whether standard COCO R@1
undercounts retrieval performance relative to CxC multi-positive R@1.

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
| `01_sugarcrepe_interventions.py` | Builds the family-split SugarCrepe compositional score figures. | Thesis-useful |
| `03_unfreezing_depth.py` | Visualizes the effect of unfreezing more CLIP vision layers. | Thesis-useful |
| `04_retrieval_heatmap.py` | Shows retrieval metrics as a configuration-by-metric heatmap. | Appendix candidate |
| `05_cross_dataset_ood.py` | Summarizes cross-dataset out-of-domain retrieval evaluation. | Thesis-useful |
| `06_missing_positive_table.py` | Builds the missing-positive analysis table. | Appendix candidate |
| `07_alignment_uniformity.py` | Plots alignment and uniformity diagnostics. | Appendix candidate |
| `08_patch_vs_cls.py` | Compares patch-level and CLS-level diagnostic behavior. | Debugging-only unless cited as diagnostic evidence |
| `09_mmvp_vlm.py` | Plots MMVP-VLM post-hoc compositional benchmark results. | Thesis-useful |
| `11_retrieval_by_bucket.py` | Plots retrieval R@1 by caption length/complexity buckets. | Appendix candidate |
| `helpers.py` | Shared loading, aggregation, style, color, and table helpers used by figure scripts. |

Archived exploratory scripts live under `archive/`. The current archived
candidate, `02_coco_false_positive_gap.py`, checks whether standard COCO R@1
undercounts retrieval performance relative to CxC multi-positive R@1.

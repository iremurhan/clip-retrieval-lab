# Figure Scripts

This directory contains the source scripts for thesis figures and diagnostic
plots. Generated CSV caches and rendered figures are intentionally not tracked.

| Script | Purpose |
| --- | --- |
| `00_methodology_overview.py` | Draws the high-level method overview for the experimental pipeline. |
| `01_sugarcrepe_interventions.py` | Compares SugarCrepe compositional scores for B0/B0plus-family intervention configs. |
| `02_sugarcrepe_capacity.py` | Summarizes SugarCrepe capacity/variant comparisons across model configurations. |
| `03_unfreezing_depth.py` | Visualizes the effect of unfreezing more CLIP vision layers. |
| `04_retrieval_grouped_bar.py` | Shows grouped retrieval R@K results across datasets/configurations. |
| `04_retrieval_heatmap.py` | Shows retrieval metrics as a configuration-by-metric heatmap. |
| `04_retrieval_slope_chart.py` | Shows paired retrieval changes between related settings/datasets. |
| `05_cross_dataset_ood.py` | Summarizes cross-dataset out-of-domain retrieval evaluation. |
| `06_missing_positive_table.py` | Builds the missing-positive analysis table. |
| `07_alignment_uniformity.py` | Plots alignment and uniformity diagnostics. |
| `08_patch_vs_cls.py` | Compares patch-level and CLS-level diagnostic behavior. |
| `09_mmvp_vlm.py` | Plots MMVP-VLM post-hoc compositional benchmark results. |
| `11_retrieval_by_bucket.py` | Plots retrieval R@1 by caption length/complexity buckets. |
| `helpers.py` | Shared loading, aggregation, style, color, and table helpers used by figure scripts. |

Archived exploratory scripts live under `archive/`. The current archived
candidate, `02_coco_false_positive_gap.py`, checks whether standard COCO R@1
undercounts retrieval performance relative to CxC multi-positive R@1.

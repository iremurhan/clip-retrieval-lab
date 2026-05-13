# Verification Report — Figures 01, 02, 07, 08

## Summary

- Figure 01 — FAIL at requested location: all expected `figs/01_*` artifacts are missing. Complete counterparts exist under `results/` and were inspected as relocated artifacts.
- Figure 02 — FAIL at requested location: all expected `figs/02_*` artifacts are missing. Counterparts exist under `results/`, but the data contains only `B0`, `B0_uf5`, and `B0_proj1024`; expected `B0_uf6` and `B0_uf7` are absent.
- Figure 07 — MISSING: all expected rendered files, tables, cache CSV, and per-checkpoint JSON directory under `figs/` are missing. The corresponding `results/cache/alignment_uniformity/` directory exists but is empty.
- Figure 08 — MISSING: all expected rendered files, table, cache CSV, and per-checkpoint JSON directory under `figs/` are missing. The corresponding `results/cache/patch_level/` directory exists but is empty.

Global finding: `figs/helpers.py` currently defines output roots as `results/figures`, `results/tables`, `results/data`, and `results/cache`, not `figs/`. That explains why the requested `figs/` artifacts are absent for Figure 01/02 even though relocated outputs exist.

## Figure 01 — SugarCrepe interventions

### Expected artifacts in `figs/`

All four expected artifacts are missing:

| Artifact | Status |
|---|---|
| `figs/01_sugarcrepe_interventions.pdf` | missing |
| `figs/01_sugarcrepe_interventions.png` | missing |
| `figs/01_sugarcrepe_interventions_data.csv` | missing |
| `figs/01_sugarcrepe_interventions_colorblind_check.png` | missing |

Because these files are missing at the requested paths, the required file-size/timestamp, CSV, PDF, PNG, visual, and color checks cannot pass for `figs/`.

### Relocated artifacts found under `results/`

These are not in the expected folder, but appear to be the generated counterparts:

| Artifact | Size | Modified |
|---|---:|---|
| `results/figures/01_sugarcrepe_interventions.pdf` | 16,751 bytes | May 11 22:48:58 2026 |
| `results/figures/01_sugarcrepe_interventions.png` | 150,762 bytes | May 11 22:48:58 2026 |
| `results/data/01_sugarcrepe_interventions_data.csv` | 2,425 bytes | May 11 22:48:58 2026 |
| `results/figures/01_sugarcrepe_interventions_colorblind_check.png` | 150,758 bytes | May 11 22:48:58 2026 |

Relocated data CSV columns are exactly:

`run_id, category, mean, std, n_seeds`

Unique run IDs:

`B0`, `B0plus`, `B1`, `B2`, `B5a_seg_spatial`, `B5b_seg_semantic`, `B5c_seg_continuous`

`n_seeds` min/max across categories:

| run_id | min | max |
|---|---:|---:|
| `B0` | 2 | 2 |
| `B0plus` | 1 | 1 |
| `B1` | 1 | 1 |
| `B2` | 1 | 1 |
| `B5a_seg_spatial` | 1 | 1 |
| `B5b_seg_semantic` | 1 | 1 |
| `B5c_seg_continuous` | 1 | 1 |

`B0` is present and is the leftmost/reference configuration in the relocated data and visual.

Categories present:

`add_att`, `add_obj`, `replace_att`, `replace_obj`, `replace_rel`, `swap_att`, `swap_obj`, `overall`

Cross-check against `/Volumes/T7/Research/wandb/runs_summary.csv`:

| run_id | source COCO SugarCrepe rows | data CSV n_seeds | Match |
|---|---:|---:|---|
| `B0` | 2 | 2 | yes |
| `B0plus` | 1 | 1 | yes |
| `B1` | 1 | 1 | yes |
| `B2` | 1 | 1 | yes |
| `B5a_seg_spatial` | 1 | 1 | yes |
| `B5b_seg_semantic` | 1 | 1 | yes |
| `B5c_seg_continuous` | 1 | 1 | yes |

No `n_seeds` mismatches were found in the relocated data.

PDF inspection of relocated file:

- Page count: 1.
- Embedded image references: 0.
- Font references: 2.
- Decompressed streams: 4.
- Text operators detected: 86.
- Path operators detected: 863.
- Interpretation: vector/text content is present; no embedded raster image references were detected.

PNG inspection of relocated file:

- Dimensions: `2550 x 1350` px.
- DPI metadata: approximately `300 x 300`.
- This is consistent with an `8.5 x 4.5` inch figure at 300 DPI.

Visual inspection of relocated PNG:

- Three top panels labeled `Add`, `Replace`, and `Swap`: present.
- One `Overall` subplot below: present.
- Width ratios appear roughly `[2, 3, 2]`: yes.
- Bars are solid with no hatching: yes.
- Single-seed configurations have `*` suffixes in the legend: yes.
- Legend is below the figure, but it wraps into two rows rather than a single horizontal row.

Colors used in relocated figure, from the script/helper colorblind palette:

| run_id | hex |
|---|---|
| `B0` | `#0173b2` |
| `B0plus` | `#de8f05` |
| `B1` | `#029e73` |
| `B2` | `#d55e00` |
| `B5a_seg_spatial` | `#cc78bc` |
| `B5b_seg_semantic` | `#ca9161` |
| `B5c_seg_continuous` | `#fbafe4` |

## Figure 02 — SugarCrepe capacity

### Expected artifacts in `figs/`

All three expected artifacts are missing:

| Artifact | Status |
|---|---|
| `figs/02_sugarcrepe_capacity.pdf` | missing |
| `figs/02_sugarcrepe_capacity.png` | missing |
| `figs/02_sugarcrepe_capacity_data.csv` | missing |

No colorblind check PNG is expected for this figure.

Because these files are missing at the requested paths, the required file-size/timestamp, CSV, PDF, PNG, visual, and color checks cannot pass for `figs/`.

### Relocated artifacts found under `results/`

| Artifact | Size | Modified |
|---|---:|---|
| `results/figures/02_sugarcrepe_capacity.pdf` | 16,431 bytes | May 11 22:48:58 2026 |
| `results/figures/02_sugarcrepe_capacity.png` | 106,802 bytes | May 11 22:48:58 2026 |
| `results/data/02_sugarcrepe_capacity_data.csv` | 1,226 bytes | May 11 22:48:58 2026 |

Relocated data CSV columns are exactly:

`run_id, category, mean, std, n_seeds`

Unique run IDs, in order:

`B0`, `B0_uf5`, `B0_proj1024`

Expected configs were:

`B0`, `B0_uf5`, `B0_uf6`, `B0_uf7`, `B0_proj1024`

`B0_uf6` and `B0_uf7` are missing from the relocated data. In `runs_summary.csv`, COCO SugarCrepe rows with non-null `overall` or `macro_avg` are present only for `B0`, `B0_uf5`, and `B0_proj1024` in the B0 family.

`n_seeds` min/max across categories:

| run_id | min | max |
|---|---:|---:|
| `B0` | 2 | 2 |
| `B0_uf5` | 2 | 2 |
| `B0_proj1024` | 1 | 1 |

Categories present:

`add_att`, `add_obj`, `replace_att`, `replace_obj`, `replace_rel`, `swap_att`, `swap_obj`, `overall`

Cross-check against `/Volumes/T7/Research/wandb/runs_summary.csv`:

| run_id | source COCO SugarCrepe rows | data CSV n_seeds | Match |
|---|---:|---:|---|
| `B0` | 2 | 2 | yes |
| `B0_uf5` | 2 | 2 | yes |
| `B0_proj1024` | 1 | 1 | yes |

No `n_seeds` mismatches were found in the relocated data.

PDF inspection of relocated file:

- Page count: 1.
- Embedded image references: 0.
- Font references: 2.
- Decompressed streams: 4.
- Text operators detected: 70.
- Path operators detected: 725.
- Interpretation: vector/text content is present; no embedded raster image references were detected.

PNG inspection of relocated file:

- Dimensions: `2550 x 1350` px.
- DPI metadata: approximately `300 x 300`.
- This is consistent with an `8.5 x 4.5` inch figure at 300 DPI.

Visual inspection of relocated PNG:

- Three top panels labeled `Add`, `Replace`, and `Swap`: present.
- One `Overall` subplot below: present.
- Width ratios appear roughly `[2, 3, 2]`: yes.
- Bars are solid with no hatching: yes.
- Single-seed configs have `*` suffix in the legend: `B0_proj1024*` present.
- Legend is below the figure and horizontal.

Colors used in relocated figure, from the script/helper colorblind palette:

| run_id | hex |
|---|---|
| `B0` | `#0173b2` |
| `B0_uf5` | `#de8f05` |
| `B0_proj1024` | `#029e73` |

## Figure 07 — Alignment & Uniformity

### Expected artifacts

All expected artifacts under `figs/` are missing:

| Artifact | Status |
|---|---|
| `figs/07_alignment_uniformity_scatter.pdf` | missing |
| `figs/07_alignment_uniformity_scatter.png` | missing |
| `figs/07_alignment_uniformity_table.tex` | missing |
| `figs/07_alignment_uniformity_correlations.tex` | missing |
| `figs/cache/alignment_uniformity_results.csv` | missing |
| `figs/cache/alignment_uniformity/` | missing |

Related `results/` state:

- `results/cache/alignment_uniformity/` exists but contains no per-checkpoint JSON files.
- `results/cache/alignment_uniformity_results.csv` is missing.
- No Figure 07 PDF/PNG/TEX outputs were found under `results/figures` or `results/tables`.

### CSV checks

Not inspectable because `figs/cache/alignment_uniformity_results.csv` is missing.

Expected columns could not be confirmed:

`run_id, dataset, seed, alignment, uniformity_img, uniformity_txt, uniformity_mean`

No unique `(run_id, dataset)` pairs, seed counts, value ranges, NaN/Inf checks, or metric sanity checks can be computed from the requested cache.

### Manifest coverage

`sugarcrepe_manifest.csv` exists with 23 rows and columns:

`checkpoint_path, dataset, wandb_run_name, wandb_run_id`

Because the alignment/uniformity results CSV is missing, all 23 manifest checkpoints currently lack a matching row in the expected results CSV. The manifest entries are:

`B0_coco_s456`, `B0_coco_s42`, `B1_coco_s42`, `B1_coco_s456`, `B2_coco_s456`, `B4_coco_s123`, `B5_seg_coco_s123`, `B5_seg_coco_s42`, `B0_flickr30k_s123`, `B0_flickr30k_s42`, `B0_flickr30k_s456`, `B0_proj1024_flickr30k_s42`, `B0plus_flickr30k_s123`, `B0plus_flickr30k_s42`, `B0plus_flickr30k_s456`, `B1_flickr30k_s42`, `B1_flickr30k_s456`, `B2_flickr30k_s123`, `B2_flickr30k_s42`, `B2_flickr30k_s456`, `B5_seg_flickr30k_s123`, `B5_seg_flickr30k_s42`, `BLIP_TEXT_flickr30k_s42`

Important source-code issue: the requested global exclusion list includes `B5_seg`, but both `figs/07_alignment_uniformity.py` and `scripts/eval/run_alignment_uniformity_local.py` define `EXCLUDE = ["B0v2", "B0plus_fixed"]` / `{"B0v2", "B0plus_fixed"}` and do not include `B5_seg`.

### LaTeX checks

Both expected `.tex` files are missing, so LaTeX/booktabs parsing and first-20-line printing cannot be performed.

The correlations table cannot be inspected, so separate Flickr30K and COCO columns cannot be confirmed from an artifact. In source, `write_correlation_table()` does define separate Flickr and COCO columns.

### Visual checks

The scatter PNG is missing, so the following cannot be visually confirmed from an artifact:

- x-axis = alignment, y-axis = uniformity_mean.
- One point per `(run_id x dataset)`.
- Colored by dataset.
- Marker shape varies by intervention family.
- run_id annotations near each point.

In source, `plot_scatter()` sets x label to `Alignment (lower is better)`, y label to `Mean uniformity (lower is better)`, colors by dataset, maps marker shape through `family_for_run()`, and annotates run IDs.

## Figure 08 — Patch vs CLS

### Expected artifacts

All expected artifacts under `figs/` are missing:

| Artifact | Status |
|---|---|
| `figs/08_patch_vs_cls.pdf` | missing |
| `figs/08_patch_vs_cls.png` | missing |
| `figs/08_patch_vs_cls_delta_heatmap.pdf` | missing |
| `figs/08_patch_vs_cls_delta_heatmap.png` | missing |
| `figs/08_patch_vs_cls_table.tex` | missing |
| `figs/cache/patch_level_results.csv` | missing |
| `figs/cache/patch_level/` | missing |

Related `results/` state:

- `results/cache/patch_level/` exists but contains no per-checkpoint JSON files.
- `results/cache/patch_level_results.csv` is missing.
- No Figure 08 PDF/PNG/TEX outputs were found under `results/figures` or `results/tables`.

### CSV checks

Not inspectable because `figs/cache/patch_level_results.csv` is missing.

Expected columns could not be confirmed:

`run_id, dataset, seed, subcategory, cls_accuracy, patch_max_accuracy, delta`

No unique run IDs, subcategories, coverage matrix, seed counts, scale checks, delta checks, or large-negative-delta checks can be computed from the requested cache.

BLIP_TEXT status:

- No cache exists, so artifact-level inclusion/exclusion cannot be determined.
- `scripts/eval/diagnostic_patch_level.py` explicitly says `BLIP_TEXT checkpoints are supported because they share the CLIP vision tower`.
- `scripts/eval/run_patch_level_local.py` excludes only `B0v2` and `B0plus_fixed`; it does not exclude `BLIP_TEXT`.
- `figs/08_patch_vs_cls.py` also excludes only `B0v2` and `B0plus_fixed`.

Important source-code issue: the requested global exclusion list includes `B5_seg`, but both `figs/08_patch_vs_cls.py` and `scripts/eval/run_patch_level_local.py` do not include `B5_seg` in their local `EXCLUDE`.

### Model wrapper inspection

`src/model.py`:

- `DualEncoder.encode_image_patches()` exists.
- `DualEncoder.encode_image()` exists immediately above it.
- `git diff -- src/model.py` shows only an insertion of `encode_image_patches()` after `encode_image()`; the existing `encode_image()` body is unchanged in the diff.

`src/model_blip.py`:

- `DualEncoderBLIPText.encode_image_patches()` exists.
- `git diff -- src/model_blip.py` shows it was added after `encode_image()`.

### Visual checks

Both Figure 08 PNGs are missing, so the requested visual checks cannot be confirmed from artifacts.

In source:

- Main figure layout is small multiples: `plot_grouped_bars()` creates one subplot per `config = run_id (dataset)`, arranged in up to four columns, with grouped CLS vs Patch-max bars inside each panel.
- Heatmap source uses rows = configs, columns = subcategories, diverging `RdBu` palette centered at 0, and annotated delta values.

### LaTeX checks

`figs/08_patch_vs_cls_table.tex` is missing, so LaTeX/booktabs parsing cannot be performed.

In source, `write_delta_table()` would write a booktabs-style table with `\toprule`, `\midrule`, and `\bottomrule`.

## Cross-figure consistency

### Configuration coverage

Because Figure 07 and Figure 08 artifacts/caches are missing, cross-figure coverage cannot be confirmed from artifacts.

Using the relocated Figure 01 data as the only available Figure 01 artifact, the COCO configurations are:

`B0`, `B0plus`, `B1`, `B2`, `B5a_seg_spatial`, `B5b_seg_semantic`, `B5c_seg_continuous`

None of these can be confirmed in Figure 07 or Figure 08 artifacts because those artifacts and cache CSVs are missing.

Using the relocated Figure 02 data, the B0-family configurations are:

`B0`, `B0_uf5`, `B0_proj1024`

Expected but missing from relocated Figure 02 data:

`B0_uf6`, `B0_uf7`

### Color consistency

Could not compare artifact colors across Figure 01 vs Figure 07/08 because Figure 07/08 PNGs are missing.

Source-level note: Figure 01/02 use a per-configuration seaborn colorblind palette. Figure 07 colors points by dataset, not run ID. Figure 08 colors bars by representation type (`CLS` vs `Patch-max`), not run ID. So run-level color consistency across 01, 07, and 08 does not appear to be implemented by design.

### Helper utilities and imports

`figs/helpers.py` exists and defines the requested utilities:

- `load_runs`
- `filter_sugarcrepe_coco`
- `split_by_baseline`

Imports:

- `figs/01_sugarcrepe_interventions.py` imports `load_runs`, `filter_sugarcrepe_coco`, and `split_by_baseline`.
- `figs/02_sugarcrepe_capacity.py` imports `load_runs`, `filter_sugarcrepe_coco`, and `split_by_baseline`.
- `figs/07_alignment_uniformity.py` imports `CACHE_DIR`, `DEFAULT_CSV_PATH`, `SAVE_FIG_DIR`, `SAVE_TABLE_DIR`, and `load_runs` from helpers.
- `figs/08_patch_vs_cls.py` imports `CACHE_DIR`, `SAVE_FIG_DIR`, and `SAVE_TABLE_DIR` from helpers.

Output-root issue:

- `figs/helpers.py` sets `SAVE_FIG_DIR = Path("results/figures")`.
- `figs/helpers.py` sets `SAVE_TABLE_DIR = Path("results/tables")`.
- `figs/helpers.py` sets `SAVE_DATA_DIR = Path("results/data")`.
- `figs/helpers.py` sets `CACHE_DIR = Path("results/cache")`.

This conflicts with the requested expected artifact location under `figs/`.

## Issues requiring decisions

1. Decide whether artifacts under `results/` are acceptable, or whether final thesis artifacts must exist under `figs/` as requested.
2. Decide how to handle the output-root mismatch in `figs/helpers.py`; current scripts write to `results/...`, not `figs/...`.
3. Decide whether Figure 02 should include `B0_uf6` and `B0_uf7`; the current `runs_summary.csv` has no COCO SugarCrepe rows for them, so they are absent from the relocated data.
4. Decide whether the Figure 01 legend wrapping into two rows is acceptable, since the requested visual check says the legend should be below the figure and horizontal.
5. Decide whether `B5_seg` must be added to the local exclusion lists in Figure 07/08 scripts and their runner scripts; current local lists omit it despite the requested global exclusion.
6. Decide whether `BLIP_TEXT` should be included in Figure 08. The diagnostic script header documents support for `BLIP_TEXT`, and the runner does not exclude it, but no cache exists to confirm actual inclusion.
7. Decide whether Figure 07 and Figure 08 should be generated or copied into place later; all expected rendered artifacts and cache CSVs are currently missing.

#!/usr/bin/env python3
"""Generate thesis result tables from a W&B run-summary CSV.

Each output file notes that the thesis preamble needs \\usepackage{multirow} and
\\usepackage{makecell}. Tables using \\resizebox also need \\usepackage{graphicx}.
"""

from __future__ import annotations

import csv
import os
import statistics
import sys


RUN_LABELS = {
    "B0": "Base-min",
    "B0plus": "Intra-Reg",
    "B0_uf5": "Unfreeze-5",
    "B0_uf6": "Unfreeze-6",
    "B0_uf7": "Unfreeze-7",
    "B0_proj1024": "Proj-1024",
    "B1": "Loss-SigLIP",
    "B2": "HN-Syntactic",
    "B4": "Aux-ObjCls",
    "B5a_seg_spatial": "Seg-Spatial",
    "B5b_seg_semantic": "Seg-Semantic",
    "B5c_seg_continuous": "Seg-Geom",
    "B5d_multistream_gate": "Sam-Gate",
    "B5d_multistream_crossattn": "Sam-XAttn",
    "B5d_multistream_concat": "Sam-Concat",
    "B5e_sam_skip": "Sam-Skip",
    "BLIP_TEXT": "Text-BLIP",
}

IGNORED_RUN_IDS = {"B0plus_fixed", "B0v2", "B5_seg", "B0_projonly"}
DATASET_ALIASES = {"flickr": "flickr30k", "flickr30k": "flickr30k", "coco": "coco"}

CAPACITY_VARIANT_ROWS = ["Base-min", "Intra-Reg", "Unfreeze-5", "Unfreeze-6", "Unfreeze-7", "Proj-1024"]
PRIMARY_ROWS = [
    "Base-min",
    "Intra-Reg",
    "Loss-SigLIP",
    "HN-Syntactic",
    "Aux-ObjCls",
    "Seg-Spatial",
    "Seg-Semantic",
    "Seg-Geom",
    "Sam-Gate",
    "Sam-XAttn",
    "Sam-Concat",
    "Sam-Skip",
]
COCO_INTERVENTION_ROWS = [
    "Base-min",
    "Intra-Reg",
    "Loss-SigLIP",
    "HN-Syntactic",
    "Aux-ObjCls",
    "Seg-Spatial",
    "Seg-Semantic",
    "Seg-Geom",
    "Sam-Gate",
    "Sam-XAttn",
    "Sam-Concat",
    "Sam-Skip",
]
APPENDIX_ROWS = CAPACITY_VARIANT_ROWS + [
    "Loss-SigLIP",
    "HN-Syntactic",
    "Aux-ObjCls",
    "Seg-Spatial",
    "Seg-Semantic",
    "Seg-Geom",
    "Sam-Gate",
    "Sam-XAttn",
    "Sam-Concat",
    "Sam-Skip",
    "Text-BLIP",
]

FORCED_MISSING_DATASETS = {("Aux-ObjCls", "flickr30k"), ("Text-BLIP", "coco")}
DEPRECATED_OUTPUT_FILES = {"table_capacity_calibration.tex"}


def metric(title, dataset, key):
    return {"title": title, "dataset": dataset, "key": key, "metric": True}


FLICKR_R1 = [
    metric("I2T", "flickr30k", "summary/test/r1_i2t"),
    metric("T2I", "flickr30k", "summary/test/r1_t2i"),
]
COCO_I2T = [
    metric("R@1", "coco", "summary/test/coco_5k_r1_i2t"),
    metric("R@5", "coco", "summary/test/coco_5k_r5_i2t"),
    metric("R@10", "coco", "summary/test/coco_5k_r10_i2t"),
]
COCO_T2I = [
    metric("R@1", "coco", "summary/test/coco_5k_r1_t2i"),
    metric("R@5", "coco", "summary/test/coco_5k_r5_t2i"),
    metric("R@10", "coco", "summary/test/coco_5k_r10_t2i"),
]
FLICKR_I2T_FULL = [
    metric("R@1", "flickr30k", "summary/test/r1_i2t"),
    metric("R@5", "flickr30k", "summary/test/r5_i2t"),
    metric("R@10", "flickr30k", "summary/test/r10_i2t"),
]
FLICKR_T2I_FULL = [
    metric("R@1", "flickr30k", "summary/test/r1_t2i"),
    metric("R@5", "flickr30k", "summary/test/r5_t2i"),
    metric("R@10", "flickr30k", "summary/test/r10_t2i"),
]


TABLES = [
    {
        "filename": "table_primary_retrieval.tex",
        "latex_label": "tab:main-retrieval",
        "position": "t",
        "resize": True,
        "rows": PRIMARY_ROWS,
        "columns": FLICKR_R1 + COCO_I2T + COCO_T2I,
        "caption_prefix": "Primary retrieval results.",
        "short_caption": "Primary retrieval results.",
        "header": "primary",
    },
    {
        "filename": "table_missing_positive_coco.tex",
        "latex_label": "tab:coco-missing-positive",
        "position": "t",
        "resize": False,
        "rows": COCO_INTERVENTION_ROWS,
        "columns": [
            metric("\\makecell{CxC I2T\\\\R@1}", "coco", "summary/test/cxc_r1_i2t"),
            metric("\\makecell{CxC T2I\\\\R@1}", "coco", "summary/test/cxc_r1_t2i"),
            metric("\\makecell{ECCV I2T\\\\mAP@R}", "coco", "summary/test/eccv_map_at_r_i2t"),
            metric("\\makecell{ECCV T2I\\\\mAP@R}", "coco", "summary/test/eccv_map_at_r_t2i"),
        ],
        "caption_prefix": "COCO missing-positive retrieval results.",
        "short_caption": "MS-COCO missing-positive diagnostics.",
        "header": "simple",
    },
    {
        "filename": "table_sugarcrepe_compositional.tex",
        "latex_label": "tab:sugarcrepe",
        "position": "t",
        "resize": False,
        "rows": COCO_INTERVENTION_ROWS,
        "columns": [
            metric("Replace-Rel", "coco", "summary/sugarcrepe/replace_rel"),
            metric("Swap-Att", "coco", "summary/sugarcrepe/swap_att"),
            metric("Swap-Obj", "coco", "summary/sugarcrepe/swap_obj"),
        ],
        "caption_prefix": "SugarCrepe compositional results.",
        "short_caption": "SugarCrepe compositional diagnostics.",
        "header": "simple",
    },
    {
        "filename": "table_appendix_full_retrieval.tex",
        "latex_label": "tab:appB_full_retrieval",
        "position": "p",
        "resize": True,
        "rows": APPENDIX_ROWS,
        "columns": FLICKR_I2T_FULL + FLICKR_T2I_FULL + COCO_I2T + COCO_T2I,
        "caption_prefix": "Full retrieval results.",
        "short_caption": "Full retrieval results on Flickr30K and MS-COCO.",
        "header": "appendix",
    },
]


def usage():
    print("Usage: python generate_result_tables.py [runs_summary.csv] [out_dir/] [--bold-multiseed-only]", file=sys.stderr)


def parse_args(argv):
    csv_path = "runs_summary.csv"
    out_dir = "tables"
    bold_multiseed_only = False
    positional = []
    for arg in argv[1:]:
        if arg == "--bold-multiseed-only":
            bold_multiseed_only = True
        elif arg.startswith("-"):
            usage()
            raise SystemExit(f"Unknown option: {arg}")
        else:
            positional.append(arg)
    if len(positional) > 2:
        usage()
        raise SystemExit("Too many positional arguments.")
    if positional:
        csv_path = positional[0]
    if len(positional) == 2:
        out_dir = positional[1]
    return csv_path, out_dir, bold_multiseed_only


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def scale_value(value):
    return value * 100.0 if abs(value) <= 1.5 else value


def normalize_dataset(value):
    return DATASET_ALIASES.get(str(value).strip(), str(value).strip())


def load_summary(csv_path):
    # data[label][dataset][metric_key][seed] = [scaled values]
    data = {}
    skipped = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=2):
            run_id = (row.get("config/run_id") or "").strip()
            if not run_id or run_id in IGNORED_RUN_IDS or run_id not in RUN_LABELS:
                if run_id:
                    skipped[run_id] = skipped.get(run_id, 0) + 1
                continue
            label = RUN_LABELS[run_id]
            dataset = normalize_dataset(row.get("config/dataset") or "")
            if dataset not in {"flickr30k", "coco"}:
                continue
            seed = (row.get("config/seed") or "").strip() or f"row-{row_index}"
            label_data = data.setdefault(label, {})
            dataset_data = label_data.setdefault(dataset, {})
            for key, raw in row.items():
                if not key.startswith("summary/"):
                    continue
                value = parse_float(raw)
                if value is None:
                    continue
                metric_data = dataset_data.setdefault(key, {})
                metric_data.setdefault(seed, []).append(scale_value(value))
    return data, skipped


def seed_values(data, label, dataset, metric_key):
    metric_data = data.get(label, {}).get(dataset, {}).get(metric_key, {})
    values = []
    for per_seed_values in metric_data.values():
        if per_seed_values:
            values.append(statistics.mean(per_seed_values))
    return values


def stats_for(data, label, column):
    if (label, column["dataset"]) in FORCED_MISSING_DATASETS:
        return None
    values = seed_values(data, label, column["dataset"], column["key"])
    if not values:
        return None
    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values) if len(values) >= 2 else None
    return {"mean": mean_value, "std": std_value, "n": len(values)}


def display_number(value):
    return f"{value:.1f}"


def displayed_mean_value(stats):
    return float(display_number(stats["mean"]))


def format_cell(stats, bold=False):
    if stats is None:
        return "--"
    if stats["n"] == 1:
        text = f"{display_number(stats['mean'])}*"
    else:
        text = f"{display_number(stats['mean'])} $\\pm$ {display_number(stats['std'])}"
    return f"\\textbf{{{text}}}" if bold else text


def best_cells(data, rows, columns, bold_multiseed_only):
    best = set()
    for col_index, column in enumerate(columns):
        if not column["metric"]:
            continue
        candidates = []
        for row_index, label in enumerate(rows):
            stats = stats_for(data, label, column)
            if stats is None:
                continue
            if bold_multiseed_only and stats["n"] < 2:
                continue
            candidates.append((displayed_mean_value(stats), row_index))
        if not candidates:
            continue
        best_value = max(value for value, _row_index in candidates)
        for value, row_index in candidates:
            if value == best_value:
                best.add((row_index, col_index))
    return best


def latex_row_label(label, table_filename):
    return f"\\textsc{{{label}}}"


def caption_text(table, bold_multiseed_only):
    if bold_multiseed_only:
        best_sentence = "Per column, the best multi-seed value is in bold."
    else:
        best_sentence = "Per column, the best value is in bold."
    return (
        f"{table['caption_prefix']} {best_sentence} "
        "Multi-seed entries are mean $\\pm$ std; single-seed entries marked with an asterisk."
    )


def alignment(num_columns):
    return "|" + "|".join(["l"] + ["c"] * num_columns) + "|"


def simple_header(columns):
    titles = ["Config"] + [column["title"] for column in columns]
    return [" & ".join(titles) + " \\\\", "\\hline"]


def primary_header():
    return [
        "\\multirow{2}{*}{Config} & \\multicolumn{2}{c|}{Flickr30K} & \\multicolumn{3}{c|}{COCO 5K I2T} & \\multicolumn{3}{c|}{COCO 5K T2I} \\\\",
        "\\cline{2-9}",
        " & I2T R@1 & T2I R@1 & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 \\\\",
        "\\hline",
    ]


def appendix_header():
    return [
        "\\multirow{2}{*}{Config} & \\multicolumn{3}{c|}{Flickr30K I2T} & \\multicolumn{3}{c|}{Flickr30K T2I} & \\multicolumn{3}{c|}{COCO 5K I2T} & \\multicolumn{3}{c|}{COCO 5K T2I} \\\\",
        "\\cline{2-13}",
        " & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 \\\\",
        "\\hline",
    ]


def header_lines(table):
    if table["header"] == "primary":
        return primary_header()
    if table["header"] == "appendix":
        return appendix_header()
    return simple_header(table["columns"])


def body_lines(data, table, best):
    rows = []
    for row_index, label in enumerate(table["rows"]):
        cells = [latex_row_label(label, table["filename"])]
        for col_index, column in enumerate(table["columns"]):
            if column["metric"]:
                stats = stats_for(data, label, column)
                cells.append(format_cell(stats, (row_index, col_index) in best))
            else:
                cells.append(column["values"].get(label, "--"))
        rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\hline")
    return rows


def table_lines(data, table, bold_multiseed_only):
    columns = table["columns"]
    best = best_cells(data, table["rows"], columns, bold_multiseed_only)
    lines = [
        "% Auto-generated by generate_result_tables.py; re-run after wandb sync + CSV re-export.",
        "% Requires \\usepackage{multirow} and \\usepackage{makecell}. Tables using \\resizebox also require \\usepackage{graphicx}.",
        f"\\begin{{table}}[{table['position']}]",
        "\\centering",
        f"\\caption[{table['short_caption']}]{{{caption_text(table, bold_multiseed_only)}}}",
        f"\\label{{{table['latex_label']}}}",
        "\\vspace{0.5\\baselineskip}",
    ]
    tabular = [f"\\begin{{tabular}}{{{alignment(len(columns))}}}", "\\hline"]
    tabular.extend(header_lines(table))
    tabular.extend(body_lines(data, table, best))
    tabular.append("\\end{tabular}")
    if table["resize"]:
        lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.extend(tabular)
        lines.append("}%")
    else:
        lines.extend(tabular)
    lines.append("\\end{table}")
    lines.append("")
    return lines


def write_table(data, table, out_dir, bold_multiseed_only):
    path = os.path.join(out_dir, table["filename"])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(table_lines(data, table, bold_multiseed_only)))
    return path


def coverage_for_table(data, table):
    coverage = {}
    for label in table["rows"]:
        for column in table["columns"]:
            if not column["metric"]:
                continue
            stats = stats_for(data, label, column)
            if stats is None:
                continue
            key = (label, column["dataset"])
            coverage.setdefault(key, set()).add(stats["n"])
    return coverage


def print_coverage(data, table):
    print(f"{table['filename']}:")
    coverage = coverage_for_table(data, table)
    if not coverage:
        print("  (no metric cells emitted)")
        return
    row_order = {label: index for index, label in enumerate(table["rows"])}
    dataset_order = {"flickr30k": 0, "coco": 1}
    for (label, dataset), counts in sorted(
        coverage.items(), key=lambda item: (row_order.get(item[0][0], 999), dataset_order.get(item[0][1], 999))
    ):
        n_text = "/".join(str(n) for n in sorted(counts))
        print(f"  ({label}, {dataset}, {n_text})")


def main(argv):
    csv_path, out_dir, bold_multiseed_only = parse_args(argv)
    data, skipped = load_summary(csv_path)
    os.makedirs(out_dir, exist_ok=True)
    for filename in DEPRECATED_OUTPUT_FILES:
        path = os.path.join(out_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed deprecated output {path}")
    for table in TABLES:
        path = write_table(data, table, out_dir, bold_multiseed_only)
        print_coverage(data, table)
        print(f"  wrote {path}")
    if skipped:
        skipped_names = ", ".join(f"{run_id}={count}" for run_id, count in sorted(skipped.items()))
        print(f"Skipped unmapped/deprecated run_ids: {skipped_names}")


if __name__ == "__main__":
    main(sys.argv)

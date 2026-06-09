from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


DEFAULT_RETRIEVAL_ROOT = Path("/Volumes/T7/Research/figures/retrieval_examples")
DEFAULT_FLIPS_JSON = DEFAULT_RETRIEVAL_ROOT / "basemin_vs_hnsyntactic_27697" / "flips.json"
DEFAULT_COCO_JSON = (
    Path("/Volumes/T7/Research/experiments/datasets/coco")
    / "caption_datasets"
    / "dataset_coco.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RETRIEVAL_ROOT / "qualitative_examples_basemin_vs_hnsyntactic"
NOT_FOUND_RANKS = {None, "", -1}


def latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def normalized_caption(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_coco_index(coco_json: Path) -> tuple[dict[int, dict], dict[str, dict]]:
    data = json.loads(coco_json.read_text())
    by_cocoid: dict[int, dict] = {}
    by_caption: dict[str, dict] = {}
    for image in data["images"]:
        by_cocoid[int(image["cocoid"])] = image
        for sent in image.get("sentences", []):
            by_caption[normalized_caption(sent["raw"])] = image
    return by_cocoid, by_caption


def copy_original_image(coco_root: Path, image: dict, output_dir: Path) -> str:
    src = coco_root / image.get("filepath", "") / image["filename"]
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = output_dir / image["filename"]
    if not dst.exists():
        shutil.copy2(src, dst)
    return image["filename"]


def normalize_image_name(value: object) -> str:
    return Path(str(value)).name.strip().lower()


def display_rank(rank: object) -> str:
    if rank in NOT_FOUND_RANKS:
        return "--"
    try:
        numeric_rank = int(rank)
    except (TypeError, ValueError):
        return "--"
    if numeric_rank < 0:
        return "--"
    return str(numeric_rank + 1)


def case_candidates(flips: dict, max_cases_per_kind: int) -> list[tuple[str, str, dict, str]]:
    chosen: list[tuple[str, str, dict, str]] = []
    for kind, description in (
        ("broken", "baseline correct; intervention misses"),
        ("fixed", "intervention correct; baseline misses"),
    ):
        for case in flips["cases"].get("t2i", {}).get(kind, [])[:max_cases_per_kind]:
            chosen.append((kind, "t2i", case, description))
    return chosen


def topk_image_names(
    topk: list[dict],
    by_cocoid: dict[int, dict],
    coco_root: Path,
    output_dir: Path,
    top_n: int,
) -> list[dict[str, object]]:
    rows = []
    for rank, item in enumerate(topk[:top_n], start=1):
        image = by_cocoid[int(item["image_id"])]
        rows.append(
            {
                "rank": rank,
                "cocoid": int(image["cocoid"]),
                "filename": copy_original_image(coco_root, image, output_dir),
            }
        )
    return rows


def image_matches_ground_truth(item: dict[str, object], gt_cocoid: int, gt_filename: str) -> bool:
    item_cocoid = item.get("cocoid")
    if item_cocoid is not None and int(item_cocoid) == int(gt_cocoid):
        return True
    return normalize_image_name(item.get("filename")) == normalize_image_name(gt_filename)


def image_grid(items: list[dict[str, object]], gt_cocoid: int, gt_filename: str, folder_name: str) -> str:
    cells = []
    for item in items:
        status = r"\qualretrievalcorrect" if image_matches_ground_truth(item, gt_cocoid, gt_filename) else r"\qualretrievalwrong"
        cells.append(
            r"\qualretrievalthumb{"
            + folder_name
            + "/"
            + str(item["filename"])
            + "}{"
            + str(item["rank"])
            + "}{"
            + status
            + "}"
        )
    return "\n".join(cells)


def example_block(record: dict, folder_name: str) -> str:
    gt_cocoid = int(record["ground_truth_cocoid"])
    gt_filename = str(record["ground_truth"])
    return (
        r"\qualretrievalexample{"
        + folder_name
        + "}{"
        + latex_escape(record["comparison"])
        + "}{"
        + latex_escape(record["query"])
        + "}{"
        + record["ground_truth"]
        + "}{"
        + display_rank(record["baseline_rank"])
        + "}{"
        + display_rank(record["intervention_rank"])
        + "}{%\n"
        + image_grid(record["baseline_topk"], gt_cocoid, gt_filename, folder_name)
        + "\n}{%\n"
        + image_grid(record["intervention_topk"], gt_cocoid, gt_filename, folder_name)
        + "\n}% "
        + record["id"]
        + ": "
        + record["description"]
    )


def tex_preamble(top_n: int) -> str:
    thumb_width = "0.135\\textwidth" if top_n <= 5 else "0.067\\textwidth"
    return (
        r"""% Auto-generated qualitative retrieval examples.
% Requires graphicx. If xcolor is loaded, correctness labels are colored.
\providecommand{\textcolor}[2]{#2}

"""
        + rf"\newcommand{{\qualretrievalthumbwidth}}{{{thumb_width}}}"
        + r"""
\newcommand{\qualretrievalgtwidth}{0.22\textwidth}
\newcommand{\qualretrievalcorrect}{\textcolor{green!45!black}{\bfseries correct}}
\newcommand{\qualretrievalwrong}{\textcolor{red!65!black}{\bfseries wrong}}
\newcommand{\qualretrievalimage}[2]{%
  {\setlength{\fboxsep}{1pt}\fbox{\includegraphics[width=#1,keepaspectratio]{#2}}}%
}
\newcommand{\qualretrievalthumb}[3]{%
  \begin{minipage}[t]{\qualretrievalthumbwidth}\centering
  \qualretrievalimage{0.96\linewidth}{#1}\\[-0.2ex]{\scriptsize #2.\ #3}
  \end{minipage}\hfill%
}
\newcommand{\qualretrievalrow}[2]{%
  \noindent{\scriptsize\bfseries #1}\par\vspace{0.2em}%
  \noindent #2\par\vspace{0.45em}%
}

% \qualretrievalexample{dir}{comparison}{query}{gt}{base rank}{intervention rank}{base grid}{intervention grid}
\newcommand{\qualretrievalexample}[8]{%
  \noindent{\bfseries #2}\hfill{\scriptsize Base rank: #5; intervention rank: #6}\par
  \noindent{\scriptsize\ttfamily #3}\par\vspace{0.45em}
  \qualretrievalrow{Ground truth}{\centering\qualretrievalimage{\qualretrievalgtwidth}{#1/#4}}%
  \qualretrievalrow{Baseline top-k}{#7}%
  \qualretrievalrow{Intervention top-k}{#8}%
  \vspace{0.35em}\hrule\vspace{0.55em}
}
"""
    )


def figure_tex(
    records: list[dict],
    folder_name: str,
    top_n: int,
    short_caption: str,
    caption: str,
    label: str,
) -> str:
    rows = "\n\n".join(example_block(record, folder_name) for record in records)
    return (
        r"""
\begin{figure}[p]
\centering
"""
        + rows
        + "\n"
        + rf"""\caption[{short_caption}]{{{caption}}}
\label{{{label}}}
\end{{figure}}
"""
    )


def appendix_label(index: int) -> str:
    suffixes = "abcdefghijklmnopqrstuvwxyz"
    suffix = suffixes[index - 1] if index <= len(suffixes) else str(index)
    return f"fig:app-qualitative-retrieval-topk-{suffix}"


def split_main_appendix(records: list[dict], main_count: int) -> tuple[list[dict], list[dict]]:
    main_records: list[dict] = []
    used_indices: set[int] = set()
    comparison_prefixes = ("Base-min vs HN-Syntactic", "Intra-Reg vs")
    for prefix in comparison_prefixes:
        for index, record in enumerate(records):
            if index in used_indices:
                continue
            if str(record["comparison"]).startswith(prefix):
                main_records.append(record)
                used_indices.add(index)
                break
    for index, record in enumerate(records):
        if len(main_records) >= main_count:
            break
        if index not in used_indices:
            main_records.append(record)
            used_indices.add(index)
    appendix_records = [record for index, record in enumerate(records) if index not in used_indices]
    return main_records, appendix_records


def build_tex(records: list[dict], folder_name: str, top_n: int, main_count: int) -> tuple[str, str | None]:
    main_records, appendix_records = split_main_appendix(records, main_count)
    main_caption = (
        "Qualitative top-5 COCO text-to-image retrieval examples. Each example shows the query caption, "
        "the ground-truth image, and the top-5 retrieved images from the reference model and the intervention. "
        "Green labels mark retrieved images that match the ground-truth COCO image."
    )
    main_tex = tex_preamble(top_n) + figure_tex(
        main_records,
        folder_name,
        top_n,
        "Qualitative top-5 COCO text-to-image retrieval examples",
        main_caption,
        "fig:qualitative-retrieval-topk",
    )
    if not appendix_records:
        return main_tex, None

    appendix_chunks = [appendix_records[index : index + 2] for index in range(0, len(appendix_records), 2)]
    appendix_figures = []
    for index, chunk in enumerate(appendix_chunks, start=1):
        appendix_figures.append(
            figure_tex(
                chunk,
                folder_name,
                top_n,
                f"Appendix qualitative top-{top_n} retrieval examples",
                (
                    f"Appendix qualitative top-{top_n} COCO text-to-image retrieval examples. Each example shows "
                    "the query caption, the ground-truth image, and the top-5 retrieved images from the reference "
                    "model and the intervention."
                ),
                appendix_label(index),
            )
        )
    return main_tex, tex_preamble(top_n) + "\n".join(appendix_figures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build thesis-ready qualitative retrieval example assets.")
    parser.add_argument("--flips-json", type=Path, action="append", default=None)
    parser.add_argument("--coco-json", type=Path, default=DEFAULT_COCO_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--latex-image-dir",
        default=None,
        help="Directory prefix to use inside \\includegraphics paths. Defaults to the output folder name.",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-cases-per-kind", type=int, default=1)
    parser.add_argument("--main-count", type=int, default=2)
    args = parser.parse_args()

    flips_paths = args.flips_json or [DEFAULT_FLIPS_JSON]
    coco_root = args.coco_json.parents[1]
    by_cocoid, by_caption = load_coco_index(args.coco_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for old in args.output_dir.glob("._*"):
        old.unlink()

    manifest = {
        "sources": [str(path) for path in flips_paths],
        "dataset_metadata": str(args.coco_json),
        "top_n_requested": args.top_n,
        "note": "Self-contained qualitative retrieval examples regenerated from original COCO image files with original filenames preserved.",
        "examples": [],
    }
    folder_name = args.latex_image_dir or args.output_dir.name

    for source_idx, flips_path in enumerate(flips_paths, start=1):
        flips = json.loads(flips_path.read_text())
        comparison = f"{flips['baseline']} vs {flips['intervention']}"
        for case_idx, (kind, direction, case, description) in enumerate(
            case_candidates(flips, args.max_cases_per_kind),
            start=1,
        ):
            gt_image = by_caption[normalized_caption(case["query"])]
            available_top_n = min(args.top_n, len(case["baseline_topk"]), len(case["intervention_topk"]))
            record = {
                "id": f"ex{source_idx:02d}_{case_idx:02d}",
                "source": str(flips_path),
                "comparison": comparison,
                "case_type": kind,
                "direction": direction,
                "description": description,
                "query_type": case["query_type"],
                "query": case["query"],
                "ground_truth": copy_original_image(coco_root, gt_image, args.output_dir),
                "ground_truth_cocoid": int(gt_image["cocoid"]),
                "baseline_rank": case["baseline_rank"],
                "intervention_rank": case["intervention_rank"],
                "top_n_available": available_top_n,
                "baseline_topk": topk_image_names(
                    case["baseline_topk"], by_cocoid, coco_root, args.output_dir, available_top_n
                ),
                "intervention_topk": topk_image_names(
                    case["intervention_topk"], by_cocoid, coco_root, args.output_dir, available_top_n
                ),
            }
            manifest["examples"].append(record)

    (args.output_dir / "examples.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    top_n_available = max((int(row["top_n_available"]) for row in manifest["examples"]), default=args.top_n)
    main_tex, appendix_tex = build_tex(manifest["examples"], folder_name, top_n_available, args.main_count)
    main_path = args.output_dir / "qualitative_retrieval_topk.tex"
    main_path.write_text(main_tex)
    if appendix_tex is not None:
        (args.output_dir / "qualitative_retrieval_topk_appendix.tex").write_text(appendix_tex)
    for old in args.output_dir.glob("._*"):
        old.unlink()
    print(main_path)
    if appendix_tex is not None:
        print(args.output_dir / "qualitative_retrieval_topk_appendix.tex")


if __name__ == "__main__":
    main()

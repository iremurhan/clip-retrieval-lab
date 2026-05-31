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
        status = "correct" if image_matches_ground_truth(item, gt_cocoid, gt_filename) else "wrong"
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


def build_tex(records: list[dict], folder_name: str, top_n: int) -> str:
    rows = "\n\n".join(example_block(record, folder_name) for record in records)
    thumb_width = "0.178\\textwidth" if top_n <= 5 else "0.087\\textwidth"
    return (
        r"""% Auto-generated qualitative retrieval examples.
% Requires graphicx. If xcolor is loaded, correctness labels are colored.
\makeatletter
\@ifundefined{textcolor}{\newcommand{\textcolor}[2]{#2}}{}
\makeatother

"""
        + rf"\newcommand{{\qualretrievalthumbwidth}}{{{thumb_width}}}"
        + r"""
\newcommand{\qualretrievalgtwidth}{0.18\textwidth}
\newcommand{\qualretrievalrowlabelwidth}{0.14\textwidth}
\newcommand{\qualretrievalrowgridwidth}{0.84\textwidth}
\newcommand{\qualretrievalcorrecttoken}{correct}
\newcommand{\qualretrievalstatus}[1]{%
  \begingroup\scriptsize\bfseries
  \def\qrtemp{#1}%
  \ifx\qrtemp\qualretrievalcorrecttoken
    \textcolor{green!45!black}{correct}%
  \else
    \textcolor{red!65!black}{wrong}%
  \fi
  \endgroup
}
\newcommand{\qualretrievalimage}[2]{%
  {\setlength{\fboxsep}{1pt}\fbox{\includegraphics[width=#1,keepaspectratio]{#2}}}%
}
\newcommand{\qualretrievalthumb}[3]{%
  \begin{minipage}[t]{\qualretrievalthumbwidth}\centering
  \qualretrievalimage{0.96\linewidth}{#1}\\[-0.2ex]{\scriptsize #2.\ \qualretrievalstatus{#3}}
  \end{minipage}\hfill%
}
\newcommand{\qualretrievalrow}[2]{%
  \begin{minipage}[t]{\qualretrievalrowlabelwidth}\raggedright\scriptsize\bfseries #1\end{minipage}\hfill
  \begin{minipage}[t]{\qualretrievalrowgridwidth}#2\end{minipage}\par\vspace{0.35em}%
}

% \qualretrievalexample{dir}{comparison}{query}{gt}{base rank}{intervention rank}{base grid}{intervention grid}
\newcommand{\qualretrievalexample}[8]{%
  \noindent{\bfseries #2}\hfill{\scriptsize Base rank: #5; intervention rank: #6}\par
  \noindent{\scriptsize\ttfamily #3}\par\vspace{0.45em}
  \qualretrievalrow{Ground truth}{\qualretrievalimage{\qualretrievalgtwidth}{#1/#4}}%
  \qualretrievalrow{Baseline top-k}{#7}%
  \qualretrievalrow{Intervention top-k}{#8}%
  \vspace{0.55em}\hrule\vspace{0.65em}
}

\newcommand{\qualretrievalfigure}[3][]{%
\begin{figure}[p]
\centering
"""
        + rows
        + "\n"
        + rf"""\caption[#1]{{#2}}
\label{{#3}}
\end{{figure}}
}}

\qualretrievalfigure[Qualitative top-{top_n} retrieval examples]{{Qualitative COCO text-to-image retrieval examples. Each row shows the query caption, ground-truth image, and the top-{top_n} retrieved images from the baseline and intervention; green labels mark retrieved images that match the ground-truth COCO image.}}{{fig:qualitative-retrieval-topk}}
"""
    )


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
    tex_name = "qualitative_retrieval_topk.tex"
    top_n_available = max((int(row["top_n_available"]) for row in manifest["examples"]), default=args.top_n)
    (args.output_dir / tex_name).write_text(build_tex(manifest["examples"], folder_name, top_n_available))
    for old in args.output_dir.glob("._*"):
        old.unlink()
    print(args.output_dir / tex_name)


if __name__ == "__main__":
    main()

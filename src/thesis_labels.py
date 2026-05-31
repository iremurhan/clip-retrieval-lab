from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_MAP_PATH = REPO_ROOT / "configs" / "thesis_label_map.yaml"

LABEL_COLUMNS = [
    "internal_run_id",
    "canonical_run_id",
    "thesis_label",
    "display_label",
    "latex_label",
    "intervention_group",
    "reference_label",
    "label_kind",
    "include_in_main_results",
    "include_in_sweep_figures",
    "sweep_display_label",
    "sweep_index",
    "is_excluded",
    "exclude_reason",
    "is_label_alias",
    "alias_of",
    "superseded_by",
    "is_superseded",
]


@dataclass(frozen=True)
class ThesisLabelInfo:
    internal_run_id: str
    canonical_run_id: str
    thesis_label: str
    display_label: str
    latex_label: str
    intervention_group: str
    reference_label: str
    label_kind: str
    include_in_main_results: bool
    include_in_sweep_figures: bool
    sweep_display_label: str
    sweep_index: float | None = None
    is_excluded: bool = False
    exclude_reason: str | None = None
    is_label_alias: bool = False
    alias_of: str | None = None
    superseded_by: str | None = None
    is_superseded: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "internal_run_id": self.internal_run_id,
            "canonical_run_id": self.canonical_run_id,
            "thesis_label": self.thesis_label,
            "display_label": self.display_label,
            "latex_label": self.latex_label,
            "intervention_group": self.intervention_group,
            "reference_label": self.reference_label,
            "label_kind": self.label_kind,
            "include_in_main_results": self.include_in_main_results,
            "include_in_sweep_figures": self.include_in_sweep_figures,
            "sweep_display_label": self.sweep_display_label,
            "sweep_index": self.sweep_index,
            "is_excluded": self.is_excluded,
            "exclude_reason": self.exclude_reason,
            "is_label_alias": self.is_label_alias,
            "alias_of": self.alias_of,
            "superseded_by": self.superseded_by,
            "is_superseded": self.is_superseded,
        }


class ThesisLabelMap:
    def __init__(self, path: str | Path = DEFAULT_LABEL_MAP_PATH):
        self.path = Path(path)
        with self.path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        self.runs: dict[str, dict[str, object]] = dict(payload.get("runs") or {})
        if not self.runs:
            raise ValueError(f"No runs found in thesis label map: {self.path}")

    def _entry(self, internal_run_id: str) -> dict[str, object] | None:
        return self.runs.get(str(internal_run_id))

    def known_ids(self) -> set[str]:
        return set(self.runs)

    def _merged_entry(self, internal: str, entry: dict[str, object]) -> tuple[str, dict[str, object], bool]:
        alias_of = entry.get("alias_of")
        if not alias_of:
            return internal, dict(entry), False
        canonical_run_id = str(alias_of)
        base_entry = self._entry(canonical_run_id)
        if base_entry is None:
            raise KeyError(f"{internal} aliases missing thesis label map entry {canonical_run_id}")
        merged = dict(base_entry)
        for key, value in entry.items():
            if key != "alias_of":
                merged[key] = value
        return canonical_run_id, merged, True

    def resolve(
        self,
        internal_run_id: object,
        *,
        present_ids: Iterable[object] | None = None,
    ) -> ThesisLabelInfo | None:
        if internal_run_id is None or pd.isna(internal_run_id):
            return None
        internal = str(internal_run_id)
        entry = self._entry(internal)
        if entry is None:
            return None

        alias_of = entry.get("alias_of")
        canonical_run_id, merged_entry, is_alias = self._merged_entry(internal, entry)

        superseded_by_raw = merged_entry.get("superseded_by")
        if isinstance(superseded_by_raw, list):
            superseders = [str(value) for value in superseded_by_raw]
        elif superseded_by_raw:
            superseders = [str(superseded_by_raw)]
        else:
            superseders = []
        present = {str(value) for value in (present_ids or []) if value is not None and not pd.isna(value)}
        matched_superseder = next((value for value in superseders if value in present), None)
        is_superseded = bool(merged_entry.get("always_superseded")) or matched_superseder is not None
        superseded_by = matched_superseder or ",".join(superseders) or None

        thesis_label = str(merged_entry["thesis_label"])
        display_label = str(merged_entry.get("display_label") or thesis_label)
        latex_label = str(merged_entry.get("latex_label") or display_label)
        label_kind = str(merged_entry.get("label_kind") or "active_main")
        include_in_main = bool(merged_entry.get("include_in_main_results", label_kind == "active_main"))
        include_in_sweep = bool(merged_entry.get("include_in_sweep_figures", False))
        sweep_display_label = str(merged_entry.get("sweep_display_label") or "")
        sweep_index_value = merged_entry.get("sweep_index")
        try:
            sweep_index = float(sweep_index_value) if sweep_index_value is not None and not pd.isna(sweep_index_value) else None
        except (TypeError, ValueError):
            sweep_index = None
        is_excluded = bool(merged_entry.get("always_excluded")) or label_kind in {"debug_only", "prose_only"}
        return ThesisLabelInfo(
            internal_run_id=internal,
            canonical_run_id=canonical_run_id,
            thesis_label=thesis_label,
            display_label=display_label,
            latex_label=latex_label,
            intervention_group=str(merged_entry.get("intervention_group") or ""),
            reference_label=str(merged_entry.get("reference_label") or ""),
            label_kind=label_kind,
            include_in_main_results=include_in_main,
            include_in_sweep_figures=include_in_sweep,
            sweep_display_label=sweep_display_label,
            sweep_index=sweep_index,
            is_excluded=is_excluded,
            exclude_reason=str(merged_entry.get("exclude_reason")) if merged_entry.get("exclude_reason") else None,
            is_label_alias=is_alias,
            alias_of=str(alias_of) if alias_of else None,
            superseded_by=superseded_by,
            is_superseded=is_superseded,
        )

    def apply_to_frame(
        self,
        df: pd.DataFrame,
        *,
        id_col: str,
        context: str,
        fail_on_unmapped: bool = True,
        drop_superseded: bool = False,
        warn_superseded: bool = True,
    ) -> pd.DataFrame:
        if id_col not in df.columns:
            raise KeyError(f"{context}: missing run-id column {id_col!r}")

        out = df.copy()
        present_ids = set(out[id_col].dropna().astype(str))
        infos = [self.resolve(value, present_ids=present_ids) for value in out[id_col]]
        unmapped = sorted(
            {
                str(value)
                for value, info in zip(out[id_col], infos, strict=True)
                if value is not None and not pd.isna(value) and info is None
            }
        )
        if unmapped and fail_on_unmapped:
            raise ValueError(f"{context}: unmapped internal run IDs: {', '.join(unmapped)}")

        label_rows = [info.as_dict() if info else _empty_label_row(value) for value, info in zip(out[id_col], infos, strict=True)]
        labels = pd.DataFrame(label_rows, index=out.index)
        for col in LABEL_COLUMNS:
            out[col] = labels[col]

        self.validate_no_duplicate_active_labels(out, context=context)
        superseded = sorted(out.loc[out["is_superseded"], "internal_run_id"].dropna().astype(str).unique())
        if superseded and warn_superseded:
            warnings.warn(
                f"{context}: superseded mapped run IDs present and should not be plotted: {', '.join(superseded)}",
                RuntimeWarning,
                stacklevel=2,
            )
        if drop_superseded:
            out = out.loc[~out["is_superseded"]].copy()
        return out

    def validate_no_duplicate_active_labels(self, df: pd.DataFrame, *, context: str) -> None:
        required = {
            "internal_run_id",
            "canonical_run_id",
            "thesis_label",
            "is_superseded",
            "is_excluded",
            "include_in_main_results",
            "include_in_sweep_figures",
        }
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{context}: missing label columns: {sorted(missing)}")
        active = df.loc[
            df["thesis_label"].notna()
            & ~df["is_superseded"].astype(bool)
            & ~df["is_excluded"].astype(bool)
            & (df["include_in_main_results"].astype(bool) | df["include_in_sweep_figures"].astype(bool))
        ]
        if active.empty:
            return
        collisions: list[str] = []
        for thesis_label, group in active.groupby("thesis_label", dropna=True):
            canonical_ids = set(group["canonical_run_id"].dropna().astype(str))
            if len(canonical_ids) <= 1:
                continue
            internal_ids = sorted(set(group["internal_run_id"].dropna().astype(str)))
            collisions.append(f"{thesis_label}: {', '.join(internal_ids)}")
        if collisions:
            raise ValueError(
                f"{context}: multiple non-alias active run IDs map to the same thesis label: "
                + "; ".join(collisions)
            )


def _empty_label_row(value: object) -> dict[str, object]:
    internal = "" if value is None or pd.isna(value) else str(value)
    return {
        "internal_run_id": internal,
        "canonical_run_id": "",
        "thesis_label": "",
        "display_label": "",
        "latex_label": "",
        "intervention_group": "",
        "reference_label": "",
        "label_kind": "",
        "include_in_main_results": False,
        "include_in_sweep_figures": False,
        "sweep_display_label": "",
        "sweep_index": None,
        "is_excluded": False,
        "exclude_reason": None,
        "is_label_alias": False,
        "alias_of": None,
        "superseded_by": None,
        "is_superseded": False,
    }


def load_thesis_label_map(path: str | Path = DEFAULT_LABEL_MAP_PATH) -> ThesisLabelMap:
    return ThesisLabelMap(path)


def apply_thesis_labels(
    df: pd.DataFrame,
    *,
    id_col: str,
    context: str,
    fail_on_unmapped: bool = True,
    drop_superseded: bool = False,
    warn_superseded: bool = True,
) -> pd.DataFrame:
    return load_thesis_label_map().apply_to_frame(
        df,
        id_col=id_col,
        context=context,
        fail_on_unmapped=fail_on_unmapped,
        drop_superseded=drop_superseded,
        warn_superseded=warn_superseded,
    )

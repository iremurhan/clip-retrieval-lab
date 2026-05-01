import argparse
import csv
import json
from pathlib import Path
from typing import Any

import wandb


DEFAULT_ENTITY = "iremurhan-bogazici-university"
DEFAULT_PROJECT = "clip-retrieval"
DEFAULT_OUT_DIR = Path("/Volumes/T7/Research/wandb")


def _serialise(value: Any) -> Any:
    """Keep scalar CSV cells readable while preserving nested values as JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def _flatten(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for key, value in values.items():
        if key.startswith("_"):
            continue
        column = f"{prefix}/{key}"
        if isinstance(value, dict):
            rows.update(_flatten(column, value))
        else:
            rows[column] = _serialise(value)
    return rows


def export_runs(entity: str, project: str, out_dir: Path) -> Path:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "runs_summary.csv"

    rows = []
    fieldnames = ["name", "id", "state", "created_at", "url"]

    for run in runs:
        summary = dict(run.summary._json_dict)
        config = dict(run.config)
        row = {
            "name": run.name,
            "id": run.id,
            "state": run.state,
            "created_at": run.created_at,
            "url": run.url,
            **_flatten("config", config),
            **_flatten("summary", summary),
        }
        rows.append(row)

        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export W&B run summaries to CSV.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = export_runs(args.entity, args.project, args.out_dir)
    print(f"Saved W&B run summaries to {out_path}")


if __name__ == "__main__":
    main()
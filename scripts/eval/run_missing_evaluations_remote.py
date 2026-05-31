#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import posixpath
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_ALIASES = {"coco": "coco", "flickr": "flickr30k", "flickr30k": "flickr30k"}
OPPOSITE_DATASET = {"coco": "flickr30k", "flickr30k": "coco"}
DEFAULT_T7_CLEAN_DIR = Path("/Volumes/T7/Research/figures/clean")
DEFAULT_T7_RESULTS_ROOT = Path("/Volumes/T7/Research/experiments/results")

EVAL_ALIASES = {
    "sugarcrepe": "sugarcrepe",
    "sugar_crepe": "sugarcrepe",
    "mmvp": "mmvp_vlm",
    "mmvp-vlm": "mmvp_vlm",
    "mmvp_vlm": "mmvp_vlm",
    "flickr30k_retrieval_ood": "flickr30k_retrieval_ood",
    "ood_flickr": "flickr30k_retrieval_ood",
    "ood_flickr30k": "flickr30k_retrieval_ood",
    "ood_coco_to_flickr": "flickr30k_retrieval_ood",
    "ood_coco_to_flickr30k": "flickr30k_retrieval_ood",
    "coco_retrieval_ood": "coco_retrieval_ood",
    "ood_coco": "coco_retrieval_ood",
    "ood_flickr_to_coco": "coco_retrieval_ood",
    "ood_flickr30k_to_coco": "coco_retrieval_ood",
    "ood_cxc": "ood_cxc",
    "ood_eccv": "ood_eccv_captions",
    "ood_eccv_captions": "ood_eccv_captions",
    "flickr30k_retrieval": "flickr30k_retrieval",
    "flickr_retrieval": "flickr30k_retrieval",
    "in_domain_flickr": "flickr30k_retrieval",
    "in_domain_flickr30k": "flickr30k_retrieval",
    "coco_5k_retrieval": "coco_5k_retrieval",
    "coco_1k_retrieval": "coco_1k_retrieval",
    "cxc": "cxc",
    "eccv": "eccv_captions",
    "eccv_captions": "eccv_captions",
    "in_domain_coco": "coco_5k_retrieval",
}


@dataclass(frozen=True)
class EvalSpec:
    slurm_eval_type: str
    train_dataset: str | None
    eval_dataset: str | None
    expected_summary_keys: tuple[str, ...]
    expected_remote_outputs: tuple[str, ...]


@dataclass
class PlannedEvaluation:
    row: dict[str, str]
    source_checkpoint: str
    checkpoint_for_slurm: str
    created_temp_copy: bool
    remote_temp_host_dir: str | None
    remote_temp_container_dir: str | None
    eval_name: str
    eval_spec: EvalSpec | None
    commands: list[list[str]]
    sbatch_command: list[str] | None
    unsupported_reason: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_dataset(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return DATASET_ALIASES.get(text, text)


def normalize_eval_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return EVAL_ALIASES.get(text, text)


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def remote_shell_cmd(server: str, remote_command: str) -> list[str]:
    return ["ssh", server, remote_command]


def run_cmd(cmd: list[str], *, dry_run: bool, capture: bool = False) -> subprocess.CompletedProcess[str] | None:
    if dry_run:
        print("[DRY RUN]", quote_cmd(cmd))
        return None
    if capture:
        return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(cmd, check=True)
    return None


def run_copy_cmd(cmd: list[str], args: argparse.Namespace) -> None:
    attempts = max(1, args.rsync_retries + 1) if cmd and cmd[0] == "rsync" else 1
    last_exc: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            run_cmd(cmd, dry_run=args.dry_run)
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            print(
                f"rsync failed with exit code {exc.returncode}; "
                f"retrying {attempt}/{attempts - 1} after {args.rsync_retry_sleep}s...",
                file=sys.stderr,
            )
            time.sleep(args.rsync_retry_sleep)
    assert last_exc is not None
    raise last_exc


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def derive_spec(eval_name: str, row_dataset: str | None) -> EvalSpec | None:
    dataset = normalize_dataset(row_dataset)
    if eval_name == "sugarcrepe":
        return EvalSpec(
            "sugarcrepe",
            dataset,
            dataset,
            ("sugarcrepe/overall", "sugarcrepe/macro_avg"),
            (),
        )
    if eval_name == "mmvp_vlm":
        return EvalSpec(
            "mmvp_vlm",
            dataset,
            dataset,
            ("mmvp_vlm/overall",),
            ("mmvp_vlm.json",),
        )
    if eval_name in {"flickr30k_retrieval_ood", "coco_retrieval_ood", "ood_cxc", "ood_eccv_captions"}:
        train_dataset = dataset
        eval_dataset = OPPOSITE_DATASET.get(train_dataset or "")
        if eval_name == "flickr30k_retrieval_ood":
            train_dataset, eval_dataset = "coco", "flickr30k"
        elif eval_name in {"coco_retrieval_ood", "ood_cxc", "ood_eccv_captions"}:
            train_dataset, eval_dataset = "flickr30k", "coco"
        if train_dataset is None or eval_dataset is None:
            return None
        if eval_dataset == "coco" and eval_name == "ood_cxc":
            keys = (f"ood/{train_dataset}_to_coco/cxc_r1_i2t", f"ood/{train_dataset}_to_coco/cxc_r1_t2i")
        elif eval_dataset == "coco" and eval_name == "ood_eccv_captions":
            keys = (
                f"ood/{train_dataset}_to_coco/eccv_map_at_r_i2t",
                f"ood/{train_dataset}_to_coco/eccv_map_at_r_t2i",
            )
        else:
            keys = (
                f"ood/{train_dataset}_to_{eval_dataset}/r1_i2t",
                f"ood/{train_dataset}_to_{eval_dataset}/r1_t2i",
            )
        return EvalSpec("ood_retrieval", train_dataset, eval_dataset, keys, ("ood_retrieval.json",))
    if eval_name in {"flickr30k_retrieval", "coco_5k_retrieval", "coco_1k_retrieval", "cxc", "eccv_captions"}:
        eval_dataset = "flickr30k" if eval_name == "flickr30k_retrieval" else "coco"
        if eval_name == "flickr30k_retrieval":
            keys = ("test/r1_i2t", "test/r1_t2i")
        elif eval_name == "coco_1k_retrieval":
            keys = ("test/coco_1k_r1_i2t", "test/coco_1k_r1_t2i")
        elif eval_name == "cxc":
            keys = ("test/cxc_r1_i2t", "test/cxc_r1_t2i")
        elif eval_name == "eccv_captions":
            keys = ("test/eccv_map_at_r_i2t", "test/eccv_map_at_r_t2i")
        else:
            keys = ("test/coco_5k_r1_i2t", "test/coco_5k_r1_t2i")
        return EvalSpec("in_domain_retrieval", eval_dataset, eval_dataset, keys, ("in_domain_retrieval.json",))
    return None


def row_matches_filters(row: dict[str, str], args: argparse.Namespace) -> bool:
    eval_name = normalize_eval_name(field(row, "missing_evaluation", "eval_type", "evaluation_type"))
    if args.eval_type and eval_name not in {normalize_eval_name(v) for v in args.eval_type}:
        return False
    if args.wandb_run_id and field(row, "wandb_run_id") not in set(args.wandb_run_id):
        return False
    if args.thesis_label and field(row, "thesis_label") not in set(args.thesis_label):
        return False
    if args.internal_run_id and field(row, "internal_run_id", "registry_id") not in set(args.internal_run_id):
        return False
    if args.canonical_run_id and field(row, "canonical_run_id") not in set(args.canonical_run_id):
        return False
    if args.dataset:
        row_dataset = normalize_dataset(field(row, "dataset"))
        allowed = {normalize_dataset(v) for v in args.dataset}
        if row_dataset not in allowed:
            return False
    if args.seed and field(row, "seed") not in {str(v) for v in args.seed}:
        return False
    return True


def is_local_checkpoint(checkpoint_path: str, local_root: Path) -> bool:
    path = Path(checkpoint_path).expanduser()
    try:
        path.resolve().relative_to(local_root.expanduser().resolve())
        return True
    except (OSError, ValueError):
        return False


def build_copy_commands(
    *,
    checkpoint_path: str,
    server: str,
    server_root: str,
    container_server_root: str,
    wandb_run_id: str,
    eval_name: str,
) -> tuple[list[list[str]], str, str, str]:
    run_dir = Path(checkpoint_path).expanduser().parent
    safe_run = wandb_run_id or "unknown_run"
    unique = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    rel_dir = posixpath.join("tmp", safe_run, eval_name, unique)
    remote_host_dir = posixpath.join(server_root.rstrip("/"), rel_dir)
    remote_container_dir = posixpath.join(container_server_root.rstrip("/"), rel_dir)
    remote_ckpt = posixpath.join(remote_container_dir, "best_model.pth")
    mkdir_cmd = remote_shell_cmd(server, "mkdir -p " + shlex.quote(remote_host_dir))
    rsync_cmd = [
        "rsync",
        "-av",
        "--partial",
        "--inplace",
        "--include",
        "best_model.pth",
        "--include",
        "config*.yaml",
        "--include",
        "config*.yml",
        "--include",
        "training.log",
        "--exclude",
        "*",
        str(run_dir) + "/",
        f"{server}:{remote_host_dir}/",
    ]
    return [mkdir_cmd, rsync_cmd], remote_ckpt, remote_host_dir, remote_container_dir


def build_sbatch_command(
    *,
    server: str,
    repo_root_remote: str,
    spec: EvalSpec,
    checkpoint_for_slurm: str,
    wandb_run_id: str,
    wandb_run_name: str,
) -> list[str]:
    args = [
        "env",
        "-u",
        "SLURM_JOB_ID",
        "sbatch",
        "--parsable",
        "scripts/eval/eval_missing_single.slurm",
        spec.slurm_eval_type,
        checkpoint_for_slurm,
        wandb_run_id,
        wandb_run_name,
        spec.train_dataset or "",
        spec.eval_dataset or "",
    ]
    remote_command = "cd " + shlex.quote(repo_root_remote) + " && " + quote_cmd(args)
    return remote_shell_cmd(server, remote_command)


def plan_evaluation(row: dict[str, str], args: argparse.Namespace) -> PlannedEvaluation:
    source_checkpoint = field(row, "checkpoint_path")
    wandb_run_id = field(row, "wandb_run_id")
    eval_name = normalize_eval_name(field(row, "missing_evaluation", "eval_type", "evaluation_type")) or ""
    spec = derive_spec(eval_name, field(row, "dataset"))
    commands: list[list[str]] = []
    remote_host_dir = None
    remote_container_dir = None
    created_temp_copy = False
    checkpoint_for_slurm = source_checkpoint

    if spec is None:
        return PlannedEvaluation(
            row=row,
            source_checkpoint=source_checkpoint,
            checkpoint_for_slurm=checkpoint_for_slurm,
            created_temp_copy=False,
            remote_temp_host_dir=None,
            remote_temp_container_dir=None,
            eval_name=eval_name,
            eval_spec=None,
            commands=[],
            sbatch_command=None,
            unsupported_reason=f"Unsupported or under-specified evaluation type: {eval_name!r}",
        )

    if not wandb_run_id:
        return PlannedEvaluation(
            row=row,
            source_checkpoint=source_checkpoint,
            checkpoint_for_slurm=checkpoint_for_slurm,
            created_temp_copy=False,
            remote_temp_host_dir=None,
            remote_temp_container_dir=None,
            eval_name=eval_name,
            eval_spec=spec,
            commands=[],
            sbatch_command=None,
            unsupported_reason="Missing wandb_run_id; refusing to run an eval that cannot log to the original W&B run.",
        )

    if is_local_checkpoint(source_checkpoint, args.local_root):
        copy_commands, checkpoint_for_slurm, remote_host_dir, remote_container_dir = build_copy_commands(
            checkpoint_path=source_checkpoint,
            server=args.server,
            server_root=args.server_root,
            container_server_root=args.container_server_root,
            wandb_run_id=wandb_run_id,
            eval_name=eval_name,
        )
        commands.extend(copy_commands)
        created_temp_copy = True

    sbatch_command = build_sbatch_command(
        server=args.server,
        repo_root_remote=args.repo_root_remote,
        spec=spec,
        checkpoint_for_slurm=checkpoint_for_slurm,
        wandb_run_id=wandb_run_id,
        wandb_run_name=field(row, "wandb_run_name"),
    )
    return PlannedEvaluation(
        row=row,
        source_checkpoint=source_checkpoint,
        checkpoint_for_slurm=checkpoint_for_slurm,
        created_temp_copy=created_temp_copy,
        remote_temp_host_dir=remote_host_dir,
        remote_temp_container_dir=remote_container_dir,
        eval_name=eval_name,
        eval_spec=spec,
        commands=commands,
        sbatch_command=sbatch_command,
    )


def append_audit(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def base_audit_record(plan: PlannedEvaluation, status: str) -> dict[str, Any]:
    row = plan.row
    return {
        "timestamp": now_iso(),
        "wandb_run_id": field(row, "wandb_run_id"),
        "wandb_run_name": field(row, "wandb_run_name"),
        "thesis_label": field(row, "thesis_label"),
        "dataset": field(row, "dataset"),
        "seed": field(row, "seed"),
        "checkpoint_source_path": plan.source_checkpoint,
        "server_temp_path": plan.remote_temp_host_dir or "",
        "evaluation_type": plan.eval_name,
        "command_submitted": quote_cmd(plan.sbatch_command) if plan.sbatch_command else "",
        "job_id": "",
        "status": status,
        "cleanup_performed": False,
        "created_temp_copy": plan.created_temp_copy,
        "internal_run_id": field(row, "internal_run_id", "registry_id"),
        "canonical_run_id": field(row, "canonical_run_id"),
    }


def parse_job_id(output: str) -> str:
    first = output.strip().splitlines()[0] if output.strip() else ""
    return first.split(";", 1)[0].strip()


def poll_slurm_job(server: str, job_id: str, poll_interval: int, timeout_seconds: int | None) -> tuple[bool, str]:
    start = time.time()
    while True:
        sacct_cmd = remote_shell_cmd(
            server,
            "sacct -j "
            + shlex.quote(job_id)
            + " --format=State,ExitCode --noheader --parsable2 | head -n 1",
        )
        try:
            result = subprocess.run(sacct_cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        except subprocess.CalledProcessError as exc:
            line = exc.stderr.strip()
        if line:
            state = line.split("|", 1)[0].strip()
            if state in {"COMPLETED"}:
                return True, line
            if state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}:
                return False, line
        if timeout_seconds is not None and time.time() - start > timeout_seconds:
            return False, f"timeout_after_{timeout_seconds}s"
        time.sleep(poll_interval)


def verify_remote_outputs(server: str, job_id: str, spec: EvalSpec, dry_run: bool) -> bool:
    if not spec.expected_remote_outputs:
        return True
    checks = []
    for rel in spec.expected_remote_outputs:
        path = posixpath.join("/output/results/missing_eval_backfill/jobs", f"{job_id}_{spec.slurm_eval_type}", rel)
        checks.append("[ -f " + shlex.quote(path) + " ]")
    cmd = remote_shell_cmd(server, " && ".join(checks))
    if dry_run:
        print("[DRY RUN]", quote_cmd(cmd))
        return True
    return subprocess.run(cmd).returncode == 0


def verify_wandb_summary(
    *,
    entity: str | None,
    project: str,
    run_id: str,
    expected_keys: tuple[str, ...],
    dry_run: bool,
) -> tuple[bool, str]:
    if dry_run:
        return True, "dry_run"
    if not expected_keys:
        return True, "no_expected_keys"
    if not entity:
        return False, "wandb_entity_required_for_verification"
    import wandb

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    summary = dict(run.summary._json_dict)
    present = [key for key in expected_keys if key in summary and summary[key] is not None]
    if present:
        return True, "present:" + ",".join(present)
    return False, "missing:" + ",".join(expected_keys)


def cleanup_temp_copy(plan: PlannedEvaluation, args: argparse.Namespace, *, dry_run: bool) -> bool:
    if not plan.created_temp_copy or not plan.remote_temp_host_dir:
        return False
    allowed_prefix = posixpath.join(args.server_root.rstrip("/"), "tmp") + "/"
    if not plan.remote_temp_host_dir.startswith(allowed_prefix):
        raise RuntimeError(
            f"Refusing cleanup outside orchestrator temp root: {plan.remote_temp_host_dir} "
            f"(allowed prefix: {allowed_prefix})"
        )
    cmd = remote_shell_cmd(args.server, "rm -rf -- " + shlex.quote(plan.remote_temp_host_dir))
    run_cmd(cmd, dry_run=dry_run)
    return not dry_run


def process_plan(plan: PlannedEvaluation, args: argparse.Namespace) -> str:
    audit = base_audit_record(plan, "planned")
    if plan.unsupported_reason:
        audit["status"] = "unsupported"
        audit["error"] = plan.unsupported_reason
        append_audit(args.output_log, audit)
        print(f"UNSUPPORTED {plan.eval_name}: {plan.unsupported_reason}")
        return "unsupported"

    assert plan.eval_spec is not None
    assert plan.sbatch_command is not None

    if args.dry_run:
        print()
        print(f"Plan: {field(plan.row, 'wandb_run_name') or field(plan.row, 'wandb_run_id')} :: {plan.eval_name}")
        for cmd in plan.commands:
            run_cmd(cmd, dry_run=True)
        run_cmd(plan.sbatch_command, dry_run=True)
        if plan.created_temp_copy and plan.remote_temp_host_dir:
            cleanup_cmd = remote_shell_cmd(args.server, "rm -rf -- " + shlex.quote(plan.remote_temp_host_dir))
            print("[DRY RUN] cleanup after verified success:", quote_cmd(cleanup_cmd))
        audit["status"] = "dry_run_planned"
        append_audit(args.output_log, audit)
        return "dry_run_planned"

    try:
        for cmd in plan.commands:
            run_copy_cmd(cmd, args)
        result = run_cmd(plan.sbatch_command, dry_run=False, capture=True)
        assert result is not None
        job_id = parse_job_id(result.stdout)
        audit["job_id"] = job_id
        audit["status"] = "submitted"
        append_audit(args.output_log, audit)
        if not args.wait:
            return "submitted"

        ok, slurm_state = poll_slurm_job(args.server, job_id, args.poll_interval, args.job_timeout_seconds)
        audit["slurm_state"] = slurm_state
        if not ok:
            audit["status"] = "failed"
            if args.cleanup_failed:
                audit["cleanup_performed"] = cleanup_temp_copy(plan, args, dry_run=False)
            append_audit(args.output_log, audit)
            return "failed"

        outputs_ok = verify_remote_outputs(args.server, job_id, plan.eval_spec, dry_run=False)
        audit["remote_outputs_verified"] = outputs_ok
        if not outputs_ok:
            audit["status"] = "failed_output_verification"
            if args.cleanup_failed:
                audit["cleanup_performed"] = cleanup_temp_copy(plan, args, dry_run=False)
            append_audit(args.output_log, audit)
            return "failed"

        if args.no_wandb_verify:
            wandb_ok, wandb_note = True, "disabled"
        else:
            wandb_ok, wandb_note = verify_wandb_summary(
                entity=args.wandb_entity,
                project=args.wandb_project,
                run_id=field(plan.row, "wandb_run_id"),
                expected_keys=plan.eval_spec.expected_summary_keys,
                dry_run=False,
            )
        audit["wandb_verification"] = wandb_note
        if not wandb_ok:
            audit["status"] = "failed_wandb_verification"
            if args.cleanup_failed:
                audit["cleanup_performed"] = cleanup_temp_copy(plan, args, dry_run=False)
            append_audit(args.output_log, audit)
            return "failed"

        audit["cleanup_performed"] = cleanup_temp_copy(plan, args, dry_run=False)
        audit["status"] = "succeeded"
        append_audit(args.output_log, audit)
        return "succeeded"
    except Exception as exc:
        audit["status"] = "error"
        audit["error"] = str(exc)
        append_audit(args.output_log, audit)
        if not args.continue_on_error:
            raise
        print(f"ERROR {plan.eval_name}: {exc}", file=sys.stderr)
        return "error"


def submit_export(args: argparse.Namespace) -> None:
    cmd = remote_shell_cmd(
        args.server,
        "cd "
        + shlex.quote(args.repo_root_remote)
        + " && "
        + quote_cmd(["sbatch", "scripts/util/export_clean_results.slurm"]),
    )
    run_cmd(cmd, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely backfill missing CLIP retrieval evaluations on a SLURM server."
    )
    parser.add_argument(
        "--missing-csv",
        type=Path,
        default=DEFAULT_T7_CLEAN_DIR / "missing_evaluations.csv",
        help="Clean-results missing-evaluation CSV. Defaults to the T7 W&B extraction output.",
    )
    parser.add_argument("--server", required=True, help="SSH target, for example user@cluster.")
    parser.add_argument("--server-root", required=True, help="Host-side temp root on the server.")
    parser.add_argument(
        "--container-server-root",
        default="/output/results/missing_eval_backfill",
        help="Container-visible path corresponding to --server-root.",
    )
    parser.add_argument("--repo-root-remote", required=True, help="Remote repository path used as sbatch cwd.")
    parser.add_argument(
        "--local-root",
        type=Path,
        default=DEFAULT_T7_RESULTS_ROOT,
        help="Local results root used to identify checkpoints that must be copied to the server.",
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        default=DEFAULT_T7_CLEAN_DIR / "missing_eval_backfill_audit.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-type", action="append", default=[])
    parser.add_argument("--wandb-run-id", action="append", default=[])
    parser.add_argument("--thesis-label", action="append", default=[])
    parser.add_argument("--internal-run-id", action="append", default=[])
    parser.add_argument("--canonical-run-id", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--seed", action="append", default=[])
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "clip-retrieval"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--no-wandb-verify", action="store_true")
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    parser.set_defaults(wait=True)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--job-timeout-seconds", type=int, default=None)
    parser.add_argument("--rsync-retries", type=int, default=2)
    parser.add_argument("--rsync-retry-sleep", type=int, default=15)
    parser.add_argument("--cleanup-failed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--export-clean-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [row for row in read_rows(args.missing_csv) if row_matches_filters(row, args)]
    if args.limit is not None:
        rows = rows[: args.limit]

    counts: dict[str, int] = {}
    for row in rows:
        plan = plan_evaluation(row, args)
        status = process_plan(plan, args)
        counts[status] = counts.get(status, 0) + 1

    print()
    print("Summary")
    print(f"  Rows selected: {len(rows)}")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    print(f"  Audit log: {args.output_log}")
    if args.export_clean_results:
        submit_export(args)
    else:
        print("  Next: rerun scripts/util/export_clean_results.py after successful jobs to refresh clean outputs.")


if __name__ == "__main__":
    main()

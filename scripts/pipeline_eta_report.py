#!/usr/bin/env python3
"""Write combined download/sharding/training pipeline ETA and status reports."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUPERVISOR_STATE_CANDIDATES = (
    "artifacts/reports/train_supervisor_phase1_talk",
    "artifacts/reports/train_supervisor_350bt",
)


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, text=True)
    return out.strip()


def _count_find(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    try:
        out = _run(["find", str(root), "-type", "f", "-name", pattern])
    except subprocess.CalledProcessError:
        return 0
    if not out:
        return 0
    return len(out.splitlines())


def _du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        out = _run(["du", "-sb", str(path)])
    except subprocess.CalledProcessError:
        return 0
    return int(out.split()[0])


def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    unique = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                unique.add(value)
    return len(unique)


def _file_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _dir_latest_mtime(path: Path) -> float:
    latest = _file_mtime(path) or 0.0
    for pattern in (
        "supervisor_*.log",
        "train_*.log",
        "generation_trend.tsv",
        "holdout_trend.tsv",
        "eval_trend.tsv",
    ):
        for candidate in path.glob(pattern):
            mtime = _file_mtime(candidate)
            if mtime is not None and mtime > latest:
                latest = mtime
    return latest


def _resolve_supervisor_state_dir(raw_value: str) -> Path:
    requested = raw_value.strip()
    if requested:
        requested_path = Path(requested)
        if requested_path.exists():
            return requested_path

    existing_candidates = [Path(p) for p in SUPERVISOR_STATE_CANDIDATES if Path(p).exists()]
    if not existing_candidates:
        if requested:
            return Path(requested)
        return Path(SUPERVISOR_STATE_CANDIDATES[-1])
    return max(existing_candidates, key=_dir_latest_mtime)


def _pgrep_root_count(pattern: str) -> int:
    proc = subprocess.run(
        ["pgrep", "-af", "--", pattern],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return 0
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if parts and parts[0].isdigit():
            pids.append(int(parts[0]))
    if not pids:
        return 0

    pid_set = set(pids)
    tree = subprocess.run(
        ["ps", "-o", "pid=,ppid=", "-p", ",".join(str(pid) for pid in pids)],
        text=True,
        capture_output=True,
        check=False,
    )
    if tree.returncode != 0 or not tree.stdout:
        return len(pids)

    ppids: dict[int, int] = {}
    for raw in tree.stdout.splitlines():
        parts = raw.split()
        if len(parts) != 2:
            continue
        if not parts[0].isdigit() or not parts[1].isdigit():
            continue
        ppids[int(parts[0])] = int(parts[1])

    root_pids = [pid for pid in pids if ppids.get(pid) not in pid_set]
    if not root_pids:
        return len(pids)
    return len(root_pids)


def _capture_command(
    cmd: list[str],
    *,
    timeout_seconds: int = 20,
    max_lines: int = 200,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        output = (proc.stdout or proc.stderr or "").strip()
        lines = output.splitlines()
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines.append(f"... truncated to {max_lines} lines ...")
            truncated = True
        return {
            "command": " ".join(cmd),
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "truncated": truncated,
            "output": "\n".join(lines),
        }
    except FileNotFoundError:
        return {
            "command": " ".join(cmd),
            "ok": False,
            "returncode": 127,
            "truncated": False,
            "output": "command not found",
        }
    except subprocess.TimeoutExpired:
        return {
            "command": " ".join(cmd),
            "ok": False,
            "returncode": 124,
            "truncated": False,
            "output": f"timed out after {timeout_seconds}s",
        }


STEP_RE = re.compile(r"step=(\d+)\b")
TARGET_STEP_RE = re.compile(r"target_step=(\d+)\b")
MAX_STEPS_RE = re.compile(r"--max-steps(?:=|\s+)(\d+)\b")
WAIT_MANIFESTS_RE = re.compile(r"waiting_for_manifests have=(\d+) need=(\d+)")
WAIT_UNIQUE_RE = re.compile(r"waiting_for_unique_inputs have=(\d+) need=(\d+)")
WAIT_TRAIN_TOKENS_RE = re.compile(r"waiting_for_train_tokens have_tokens=(\d+) need_tokens=(\d+)")


def _latest_train_step(supervisor_state_dir: Path) -> int:
    if not supervisor_state_dir.exists():
        return 0
    logs = sorted(
        supervisor_state_dir.glob("train_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for log_path in logs[:5]:
        proc = subprocess.run(
            ["tail", "-n", "400", str(log_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        text = proc.stdout if proc.returncode == 0 else ""
        steps = [int(m.group(1)) for m in STEP_RE.finditer(text)]
        if steps:
            return max(steps)
    return 0


def _latest_supervisor_target_step(supervisor_state_dir: Path) -> int | None:
    if not supervisor_state_dir.exists():
        return None
    logs = sorted(
        supervisor_state_dir.glob("supervisor_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for log_path in logs[:4]:
        proc = subprocess.run(
            ["tail", "-n", "400", str(log_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        text = proc.stdout if proc.returncode == 0 else ""
        if not text:
            continue
        for line in reversed(text.splitlines()):
            if "train_launch " not in line:
                continue
            match = TARGET_STEP_RE.search(line)
            if match:
                return int(match.group(1))
    return None


def _active_trainer_target_step() -> int | None:
    proc = subprocess.run(
        ["pgrep", "-af", "--", r"llm\.cli train"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    targets: list[int] = []
    for line in proc.stdout.splitlines():
        match = MAX_STEPS_RE.search(line)
        if match:
            targets.append(int(match.group(1)))
    if not targets:
        return None
    return max(targets)


def _effective_train_target_step(
    configured_target_step: int,
    supervisor_state_dir: Path,
    train_step: int,
) -> int | None:
    if configured_target_step > 0:
        return max(configured_target_step, train_step)
    active_target = _active_trainer_target_step()
    if active_target is not None:
        return max(active_target, train_step)
    supervisor_target = _latest_supervisor_target_step(supervisor_state_dir)
    if supervisor_target is not None:
        return max(supervisor_target, train_step)
    return None


def _latest_generation_summary(supervisor_state_dir: Path) -> dict[str, Any]:
    trend_path = supervisor_state_dir / "generation_trend.tsv"
    if not trend_path.exists():
        return {"step": None, "generation_rc": None, "pass_rate": None, "regression_pass": None}
    latest: str | None = None
    with trend_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            row = line.strip()
            if row and not row.startswith("run_tag\t"):
                latest = row
    if latest is None:
        return {"step": None, "generation_rc": None, "pass_rate": None, "regression_pass": None}
    parts = latest.split("\t")
    # run_tag,step,generation_rc,pass_rate,check_pass_rate,avg_case_score,cases_passed,cases_total,regression_pass,baseline_report,report_json
    if len(parts) < 9:
        return {"step": None, "generation_rc": None, "pass_rate": None, "regression_pass": None}
    step = int(parts[1]) if parts[1].isdigit() else None
    return {
        "step": step,
        "generation_rc": parts[2],
        "pass_rate": parts[3],
        "regression_pass": parts[8],
    }


def _parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _trend_rows(path: Path, min_cols: int) -> list[list[str]]:
    if not path.exists():
        return []
    rows: list[list[str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                row = line.strip()
                if not row or row.startswith("run_tag\t"):
                    continue
                parts = row.split("\t")
                if len(parts) >= min_cols:
                    rows.append(parts)
    except OSError:
        return []
    return rows


def _trend_metric_state(
    *,
    latest_rc: str,
    latest_regression_pass: str | None,
    latest_pass: float | None,
    latest_check: float | None,
    latest_score: float | None,
    prev_pass: float | None,
    prev_check: float | None,
    prev_score: float | None,
    pass_eps: float,
    check_eps: float,
    score_eps: float,
) -> str:
    if latest_rc not in {"0", "NA", ""}:
        return "regressed"
    if latest_regression_pass in {"False", "0"}:
        return "regressed"
    if latest_pass is None or latest_check is None or latest_score is None:
        return "warming"
    if prev_pass is None or prev_check is None or prev_score is None:
        return "warming"
    d_pass = latest_pass - prev_pass
    d_check = latest_check - prev_check
    d_score = latest_score - prev_score
    if d_pass < -pass_eps or d_check < -check_eps or d_score < -score_eps:
        return "regressed"
    if d_pass > pass_eps or d_check > check_eps or d_score > score_eps:
        return "improving"
    return "flat"


def _quality_heartbeat(supervisor_state_dir: Path) -> dict[str, str]:
    eval_rows = _trend_rows(supervisor_state_dir / "eval_trend.tsv", min_cols=8)
    gen_rows = _trend_rows(supervisor_state_dir / "generation_trend.tsv", min_cols=9)
    holdout_rows = _trend_rows(supervisor_state_dir / "holdout_trend.tsv", min_cols=9)

    eval_state = "unknown"
    if eval_rows:
        latest = eval_rows[-1]
        prev = eval_rows[-2] if len(eval_rows) > 1 else None
        eval_regression = latest[8].strip() if len(latest) > 8 else ""
        if eval_regression not in {"True", "False", "1", "0"}:
            eval_regression = ""
        eval_state = _trend_metric_state(
            latest_rc=latest[2].strip(),
            latest_regression_pass=(eval_regression or None),
            latest_pass=_parse_float(latest[3]),
            latest_check=_parse_float(latest[4]),
            latest_score=_parse_float(latest[5]),
            prev_pass=(_parse_float(prev[3]) if prev is not None else None),
            prev_check=(_parse_float(prev[4]) if prev is not None else None),
            prev_score=(_parse_float(prev[5]) if prev is not None else None),
            pass_eps=0.005,
            check_eps=0.005,
            score_eps=0.002,
        )

    gen_state = "unknown"
    if gen_rows:
        latest = gen_rows[-1]
        prev = gen_rows[-2] if len(gen_rows) > 1 else None
        gen_regression = latest[8].strip() if len(latest) > 8 else ""
        if gen_regression not in {"True", "False", "1", "0"}:
            gen_regression = ""
        gen_state = _trend_metric_state(
            latest_rc=latest[2].strip(),
            latest_regression_pass=(gen_regression or None),
            latest_pass=_parse_float(latest[3]),
            latest_check=_parse_float(latest[4]),
            latest_score=_parse_float(latest[5]),
            prev_pass=(_parse_float(prev[3]) if prev is not None else None),
            prev_check=(_parse_float(prev[4]) if prev is not None else None),
            prev_score=(_parse_float(prev[5]) if prev is not None else None),
            pass_eps=0.005,
            check_eps=0.005,
            score_eps=0.002,
        )

    holdout_state = "unknown"
    if holdout_rows:
        latest = holdout_rows[-1]
        prev = holdout_rows[-2] if len(holdout_rows) > 1 else None
        holdout_regression = latest[8].strip() if len(latest) > 8 else ""
        if holdout_regression not in {"True", "False", "1", "0"}:
            holdout_regression = ""
        holdout_state = _trend_metric_state(
            latest_rc=latest[2].strip(),
            latest_regression_pass=(holdout_regression or None),
            latest_pass=_parse_float(latest[3]),
            latest_check=_parse_float(latest[4]),
            latest_score=_parse_float(latest[5]),
            prev_pass=(_parse_float(prev[3]) if prev is not None else None),
            prev_check=(_parse_float(prev[4]) if prev is not None else None),
            prev_score=(_parse_float(prev[5]) if prev is not None else None),
            pass_eps=0.005,
            check_eps=0.005,
            score_eps=0.002,
        )

    states = {eval_state, gen_state, holdout_state}
    if "regressed" in states:
        overall = "regressed"
    elif "improving" in states:
        overall = "improving"
    elif states <= {"unknown"}:
        overall = "unknown"
    elif "warming" in states:
        overall = "warming"
    else:
        overall = "flat"
    return {
        "overall": overall,
        "eval_state": eval_state,
        "gen_state": gen_state,
        "holdout_state": holdout_state,
    }


def _confidence_score(level: str) -> float:
    if level == "high":
        return 1.0
    if level == "medium":
        return 0.65
    return 0.30


def _status_confidence(
    *,
    manifest_overlap_inputs: int,
    manifest_unique_inputs: int,
    expected_parquet_files: int,
    coverage_rate: float | None,
    train_rate: float | None,
    trainer_active: bool,
    quality_overall: str,
) -> dict[str, Any]:
    if manifest_unique_inputs <= 0:
        coverage_level = "low"
    elif manifest_overlap_inputs == 0 and coverage_rate is not None and coverage_rate > 0:
        coverage_level = "high"
    elif manifest_overlap_inputs <= max(2, expected_parquet_files // 200):
        coverage_level = "medium"
    else:
        coverage_level = "low"

    if train_rate is not None and train_rate > 0:
        train_level = "high"
    elif trainer_active:
        train_level = "medium"
    else:
        train_level = "low"

    if quality_overall in {"improving", "flat"}:
        quality_level = "high"
    elif quality_overall in {"warming", "unknown"}:
        quality_level = "medium"
    else:
        quality_level = "low"

    overall = (
        _confidence_score(coverage_level)
        + _confidence_score(train_level)
        + _confidence_score(quality_level)
    ) / 3.0
    return {
        "coverage": coverage_level,
        "train_eta": train_level,
        "quality": quality_level,
        "overall_score": round(overall, 4),
    }


def _manifest_input_coverage(shards_root: Path) -> dict[str, int]:
    if not shards_root.exists():
        return {
            "manifest_count": 0,
            "unique_input_files": 0,
            "overlap_input_files": 0,
            "overlap_manifests": 0,
            "manifest_parse_errors": 0,
        }

    manifests = list(shards_root.rglob("manifest.json"))
    manifests.extend(shards_root.rglob("manifest.offloaded.json"))
    manifests = sorted(manifests)
    file_counts: dict[str, int] = {}
    per_manifest_files: list[set[str]] = []
    parse_errors = 0
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parse_errors += 1
            continue
        raw_files = payload.get("input_files", [])
        if not isinstance(raw_files, list):
            parse_errors += 1
            continue
        names = {Path(str(raw)).name for raw in raw_files if str(raw).strip()}
        if not names:
            continue
        per_manifest_files.append(names)
        for name in names:
            file_counts[name] = file_counts.get(name, 0) + 1

    overlap_input_files = sum(1 for count in file_counts.values() if count > 1)
    overlap_manifests = sum(
        1 for names in per_manifest_files if any(file_counts.get(name, 0) > 1 for name in names)
    )
    return {
        "manifest_count": len(manifests),
        "unique_input_files": len(file_counts),
        "overlap_input_files": overlap_input_files,
        "overlap_manifests": overlap_manifests,
        "manifest_parse_errors": parse_errors,
    }


def _manifest_hot_state(shards_root: Path) -> dict[str, int]:
    if not shards_root.exists():
        return {
            "active_manifests": 0,
            "offloaded_manifests": 0,
            "active_manifests_with_symlink_bins": 0,
        }
    active = sorted(shards_root.rglob("manifest.json"))
    offloaded = list(shards_root.rglob("manifest.offloaded.json"))
    active_with_symlink = 0
    for manifest_path in active:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rels: list[str] = []
        for split in ("train", "val"):
            split_meta = payload.get(split, {})
            if not isinstance(split_meta, dict):
                continue
            shards = split_meta.get("shards", [])
            if not isinstance(shards, list):
                continue
            for row in shards:
                if not isinstance(row, dict):
                    continue
                rel = row.get("path")
                if isinstance(rel, str) and rel.strip():
                    rels.append(rel)
        if any((manifest_path.parent / rel).is_symlink() for rel in rels):
            active_with_symlink += 1
    return {
        "active_manifests": len(active),
        "offloaded_manifests": len(offloaded),
        "active_manifests_with_symlink_bins": active_with_symlink,
    }


def _offload_eligibility(
    shards_root: Path,
    trained_batches_file: Path,
    *,
    keep_local_batches: int,
    min_active_manifests: int,
) -> dict[str, Any]:
    if not shards_root.exists():
        return {
            "eligible_batches": 0,
            "raw_eligible_batches": 0,
            "max_offloadable_batches": 0,
            "trained_registry_present": False,
        }

    manifests = sorted(
        shards_root.rglob("manifest.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    batch_names = [path.parent.name for path in manifests]
    keep_n = max(0, keep_local_batches)
    keep_set = set(batch_names[:keep_n])
    candidates = [name for name in batch_names if name not in keep_set]

    trained_registry_present = trained_batches_file.exists()
    trained_batches: set[str] = set()
    if trained_registry_present:
        with trained_batches_file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                value = line.strip()
                if value and not value.startswith("#"):
                    trained_batches.add(value)

    raw_eligible = (
        sum(1 for name in candidates if name in trained_batches) if trained_registry_present else 0
    )
    max_offloadable = max(0, len(batch_names) - max(0, min_active_manifests))
    eligible = min(raw_eligible, max_offloadable)
    return {
        "eligible_batches": eligible,
        "raw_eligible_batches": raw_eligible,
        "max_offloadable_batches": max_offloadable,
        "trained_registry_present": trained_registry_present,
    }


def _default_offload_trained_file(
    supervisor_state_dir: Path,
    configured_path: str,
) -> Path:
    if configured_path:
        return Path(configured_path)
    primary = supervisor_state_dir / "trained_batch_names.txt"
    if primary.exists():
        return primary
    phase1 = Path("artifacts/reports/train_supervisor_phase1_talk/trained_batch_names.txt")
    if phase1.exists():
        return phase1
    standard = Path("artifacts/reports/train_supervisor_350bt/trained_batch_names.txt")
    if standard.exists():
        return standard
    return primary


def _latest_supervisor_gate(supervisor_state_dir: Path) -> str:
    if not supervisor_state_dir.exists():
        return "unknown"
    logs = sorted(
        supervisor_state_dir.glob("supervisor_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for log_path in logs[:3]:
        proc = subprocess.run(
            ["tail", "-n", "200", str(log_path)],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            continue
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        for line in reversed(lines):
            match = WAIT_TRAIN_TOKENS_RE.search(line)
            if match:
                return f"waiting_train_tokens {match.group(1)}/{match.group(2)}"
            match = WAIT_UNIQUE_RE.search(line)
            if match:
                return f"waiting_unique_inputs {match.group(1)}/{match.group(2)}"
            match = WAIT_MANIFESTS_RE.search(line)
            if match:
                return f"waiting_manifests {match.group(1)}/{match.group(2)}"
            if "train_launch " in line:
                return "train_chunk_launching"
    return "unknown"


def _task_stop_reason(
    task_name: str,
    *,
    active: dict[str, int],
    coverage_complete: bool,
    warm_parquet: int,
    expected_parquet_files: int,
    train_step: int,
    train_target_step: int | None,
    supervisor_gate: str,
    offload_eligible_batches: int,
    trained_registry_present: bool,
) -> str:
    if task_name == "hf_watchdog":
        if warm_parquet >= expected_parquet_files:
            return "download complete"
        return "not started"
    if task_name == "download_worker":
        if warm_parquet >= expected_parquet_files:
            return "download complete"
        if active.get("hf_watchdog", 0) > 0:
            return "watchdog-managed; worker idle/restarting"
        return "not started"
    if task_name == "stage_watchdog":
        if coverage_complete:
            return "coverage complete"
        if active.get("stage_loop", 0) > 0:
            return "stage-loop running directly (no watchdog)"
        return "not started"
    if task_name == "stage_loop":
        if coverage_complete:
            return "coverage complete"
        if active.get("stage_watchdog", 0) > 0:
            return "waiting for watchdog restart"
        return "not started"
    if task_name == "shard_builder":
        if coverage_complete:
            return "all expected parquet processed"
        if active.get("stage_loop", 0) > 0:
            return "idle between shard batches"
        return "not started"
    if task_name == "train_supervisor":
        if train_target_step is not None and train_step >= train_target_step:
            return "target step reached"
        if supervisor_gate != "unknown":
            return f"blocked: {supervisor_gate}"
        return "not started"
    if task_name == "trainer":
        if train_target_step is not None and train_step >= train_target_step:
            return "target step reached"
        if active.get("train_supervisor", 0) > 0:
            if supervisor_gate.startswith("waiting_"):
                return f"waiting on supervisor gate ({supervisor_gate})"
            return "idle between chunks/eval"
        return "no active supervisor"
    if task_name == "eval_runner":
        return "runs only during eval windows"
    if task_name == "generation_gate_runner":
        return "runs only on scheduled gate interval"
    if task_name == "shard_offload":
        if not trained_registry_present:
            return "trained-batch registry missing"
        if offload_eligible_batches <= 0:
            return "no eligible trained batches"
        return f"awaiting timer trigger (eligible={offload_eligible_batches})"
    return "not started"


def _eta_seconds(remaining: float | None, rate_per_sec: float | None) -> float | None:
    if remaining is None:
        return None
    if rate_per_sec is None or rate_per_sec <= 0:
        return None
    if remaining <= 0:
        return 0.0
    return remaining / rate_per_sec


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds <= 0:
        return "done"
    minutes = int(seconds // 60)
    hours = minutes // 60
    mins = minutes % 60
    if hours > 0:
        return f"{hours}h{mins:02d}m"
    return f"{mins}m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-dir", default="/mnt/ceph/llm/data/fineweb/sample-350BT")
    parser.add_argument("--shards-root", default="data/shards_global/fineweb-global-bpe-v1")
    parser.add_argument("--stage-state-dir", default="artifacts/reports/fineweb_stage_shard_loop")
    parser.add_argument(
        "--supervisor-state-dir",
        default="",
        help=(
            "Supervisor state dir. Default: auto-detect newest existing path from "
            "artifacts/reports/train_supervisor_phase1_talk and "
            "artifacts/reports/train_supervisor_350bt."
        ),
    )
    parser.add_argument("--expected-parquet-files", type=int, default=510)
    parser.add_argument("--expected-bytes", type=int, default=1061360917731)
    parser.add_argument(
        "--train-target-step",
        type=int,
        default=0,
        help="Training target step. 0 means auto-detect from active trainer/supervisor logs.",
    )
    parser.add_argument("--output-json", default="artifacts/reports/pipeline_status.json")
    parser.add_argument("--output-text", default="artifacts/reports/pipeline_status.txt")
    parser.add_argument("--state-file", default="artifacts/reports/pipeline_status_state.json")
    parser.add_argument("--command-timeout-seconds", type=int, default=20)
    parser.add_argument("--command-max-lines", type=int, default=200)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument(
        "--offload-trained-batches-file",
        default="",
        help=(
            "Optional trained-batches registry for shard offload eligibility. "
            "Default: <supervisor-state-dir>/trained_batch_names.txt"
        ),
    )
    parser.add_argument("--offload-keep-local-batches", type=int, default=24)
    parser.add_argument("--offload-min-active-manifests", type=int, default=48)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one snapshot and exit (same behavior as default when --loop is unset)",
    )
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def _read_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def collect_status(args: argparse.Namespace) -> dict[str, Any]:
    warm_dir = Path(args.warm_dir)
    shards_root = Path(args.shards_root)
    stage_state_dir = Path(args.stage_state_dir)
    sup_dir = _resolve_supervisor_state_dir(args.supervisor_state_dir)

    now = time.time()
    warm_parquet = _count_find(warm_dir, "*.parquet")
    warm_incomplete = _count_find(warm_dir, "*.incomplete")
    warm_bytes = _du_bytes(warm_dir)
    manifests = _count_find(shards_root, "manifest.json")
    manifest_coverage = _manifest_input_coverage(shards_root)
    manifest_hot_state = _manifest_hot_state(shards_root)
    manifest_unique_inputs = int(manifest_coverage["unique_input_files"])
    manifest_overlap_inputs = int(manifest_coverage["overlap_input_files"])
    manifest_overlap_manifests = int(manifest_coverage["overlap_manifests"])
    sharded_parquet = _count_nonempty_lines(stage_state_dir / "processed_parquet_files.txt")
    trained_batch_count = _count_nonempty_lines(sup_dir / "trained_batch_names.txt")
    offload_trained_file = _default_offload_trained_file(
        sup_dir,
        args.offload_trained_batches_file,
    )
    offload_eligibility = _offload_eligibility(
        shards_root,
        offload_trained_file,
        keep_local_batches=int(args.offload_keep_local_batches),
        min_active_manifests=int(args.offload_min_active_manifests),
    )
    train_step = _latest_train_step(sup_dir)
    train_target_step = _effective_train_target_step(args.train_target_step, sup_dir, train_step)
    supervisor_gate = _latest_supervisor_gate(sup_dir)
    generation_gate_latest = _latest_generation_summary(sup_dir)
    quality_heartbeat = _quality_heartbeat(sup_dir)

    active = {
        "hf_watchdog": _pgrep_root_count(r"hf_download_watchdog\.sh"),
        "download_worker": _pgrep_root_count(r"hf_download_resumable\.sh"),
        "stage_watchdog": _pgrep_root_count(r"fineweb_stage_shard_watchdog\.sh"),
        "stage_loop": _pgrep_root_count(r"fineweb_stage_shard_loop\.sh"),
        "shard_builder": _pgrep_root_count(r"scripts/fineweb_parquet_to_shards\.py"),
        "train_supervisor": _pgrep_root_count(r"train_supervisor_rtx5070_350bt\.sh"),
        "trainer": _pgrep_root_count(r"llm\.cli train"),
        "eval_runner": _pgrep_root_count(r"scripts/eval_checkpoint_prompts\.py"),
        "generation_gate_runner": _pgrep_root_count(
            r"scripts/eval_checkpoint_prompts\.py .*generation_smoke_suite_v1\.json"
        ),
        "shard_offload": _pgrep_root_count(r"scripts/offload_shard_bins_to_warm\.py"),
    }
    system_commands = {
        "top": _capture_command(
            ["top", "-b", "-n", "1"],
            timeout_seconds=args.command_timeout_seconds,
            max_lines=args.command_max_lines,
        ),
        "free_h": _capture_command(
            ["free", "-h"],
            timeout_seconds=args.command_timeout_seconds,
            max_lines=args.command_max_lines,
        ),
        "nvidia_smi": _capture_command(
            ["nvidia-smi"],
            timeout_seconds=args.command_timeout_seconds,
            max_lines=args.command_max_lines,
        ),
        "df_h": _capture_command(
            ["df", "-h"],
            timeout_seconds=args.command_timeout_seconds,
            max_lines=args.command_max_lines,
        ),
    }

    prev = _read_state(Path(args.state_file))
    dt = None
    bytes_rate = None
    warm_parquet_rate = None
    sharded_parquet_rate = None
    manifest_rate = None
    unique_inputs_rate = None
    step_rate = None
    if prev is not None:
        dt = now - float(prev.get("ts", now))
        if dt > 0:
            prev_bytes = int(prev.get("warm_bytes", warm_bytes))
            prev_warm_parquet = int(prev.get("warm_parquet_count", warm_parquet))
            prev_sharded = int(prev.get("sharded_parquet_count", sharded_parquet))
            prev_manifests = int(prev.get("manifest_count", manifests))
            prev_unique_inputs = int(prev.get("manifest_unique_input_files", manifest_unique_inputs))
            prev_step = int(prev.get("train_step", train_step))
            d_bytes = warm_bytes - prev_bytes
            d_warm_parquet = warm_parquet - prev_warm_parquet
            d_sharded = sharded_parquet - prev_sharded
            d_manifests = manifests - prev_manifests
            d_unique_inputs = manifest_unique_inputs - prev_unique_inputs
            d_steps = train_step - prev_step
            if d_bytes >= 0:
                bytes_rate = d_bytes / dt
            if d_warm_parquet >= 0:
                warm_parquet_rate = d_warm_parquet / dt
            if d_sharded >= 0:
                sharded_parquet_rate = d_sharded / dt
            if d_manifests >= 0:
                manifest_rate = d_manifests / dt
            if d_unique_inputs >= 0:
                unique_inputs_rate = d_unique_inputs / dt
            if d_steps > 0:
                step_rate = d_steps / dt

    trainer_stall_seconds: int | None = None
    train_step_change_ts = now
    train_step_change_value = train_step
    if prev is not None:
        prev_change_ts = prev.get("train_step_change_ts")
        prev_change_value = prev.get("train_step_change_value")
        if isinstance(prev_change_ts, (int, float)) and isinstance(prev_change_value, int):
            train_step_change_ts = float(prev_change_ts)
            train_step_change_value = int(prev_change_value)
    if active.get("trainer", 0) > 0:
        if train_step != train_step_change_value:
            train_step_change_value = train_step
            train_step_change_ts = now
        trainer_stall_seconds = int(max(0.0, now - train_step_change_ts))
    else:
        train_step_change_value = train_step
        train_step_change_ts = now

    rem_bytes = max(0, int(args.expected_bytes) - warm_bytes)
    rem_download_parquet = max(0, int(args.expected_parquet_files) - warm_parquet)
    rem_sharded_parquet = max(0, int(args.expected_parquet_files) - sharded_parquet)
    rem_unique_inputs = max(0, int(args.expected_parquet_files) - manifest_unique_inputs)
    rem_steps = None if train_target_step is None else max(0, int(train_target_step) - train_step)
    coverage_complete = (
        warm_parquet >= int(args.expected_parquet_files)
        and sharded_parquet >= int(args.expected_parquet_files)
        and manifest_unique_inputs >= int(args.expected_parquet_files)
    )

    task_order = [
        "hf_watchdog",
        "download_worker",
        "stage_watchdog",
        "stage_loop",
        "shard_builder",
        "train_supervisor",
        "trainer",
        "eval_runner",
        "generation_gate_runner",
        "shard_offload",
    ]
    task_status: dict[str, dict[str, Any]] = {}
    for task_name in task_order:
        count = int(active.get(task_name, 0))
        if count > 0:
            task_status[task_name] = {"state": "RUN", "count": count, "reason": ""}
            continue
        reason = _task_stop_reason(
            task_name,
            active=active,
            coverage_complete=coverage_complete,
            warm_parquet=warm_parquet,
            expected_parquet_files=int(args.expected_parquet_files),
            train_step=train_step,
            train_target_step=train_target_step,
            supervisor_gate=supervisor_gate,
            offload_eligible_batches=int(offload_eligibility["eligible_batches"]),
            trained_registry_present=bool(offload_eligibility["trained_registry_present"]),
        )
        task_status[task_name] = {"state": "STOP", "count": 0, "reason": reason}

    coverage_rate_source = "manifest_unique_inputs"
    coverage_rate = unique_inputs_rate
    if (
        (coverage_rate is None or coverage_rate <= 0)
        and manifest_overlap_inputs == 0
        and sharded_parquet_rate is not None
        and sharded_parquet_rate > 0
        and sharded_parquet >= manifest_unique_inputs
    ):
        coverage_rate = sharded_parquet_rate
        coverage_rate_source = "sharding_fallback_no_overlap"

    eta = {
        "download_seconds": _eta_seconds(rem_bytes, bytes_rate),
        "download_parquet_seconds": _eta_seconds(rem_download_parquet, warm_parquet_rate),
        "sharding_seconds": _eta_seconds(rem_sharded_parquet, sharded_parquet_rate),
        "manifests_seconds": _eta_seconds(rem_sharded_parquet, manifest_rate),
        "manifest_unique_inputs_seconds": _eta_seconds(rem_unique_inputs, coverage_rate),
        "train_seconds": _eta_seconds(rem_steps, step_rate),
    }

    status = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "warm_parquet_count": warm_parquet,
            "warm_incomplete_count": warm_incomplete,
            "warm_bytes": warm_bytes,
            "manifest_count": manifests,
            "manifest_unique_input_files": manifest_unique_inputs,
            "manifest_overlap_input_files": manifest_overlap_inputs,
            "manifest_overlap_manifests": manifest_overlap_manifests,
            "manifest_parse_errors": int(manifest_coverage["manifest_parse_errors"]),
            "active_manifests": int(manifest_hot_state["active_manifests"]),
            "offloaded_manifests": int(manifest_hot_state["offloaded_manifests"]),
            "active_manifests_with_symlink_bins": int(
                manifest_hot_state["active_manifests_with_symlink_bins"]
            ),
            "trained_batch_names_count": trained_batch_count,
            "offload_eligible_batches": int(offload_eligibility["eligible_batches"]),
            "offload_raw_eligible_batches": int(offload_eligibility["raw_eligible_batches"]),
            "offload_max_offloadable_batches": int(offload_eligibility["max_offloadable_batches"]),
            "offload_trained_registry_present": bool(
                offload_eligibility["trained_registry_present"]
            ),
            "sharded_parquet_count": sharded_parquet,
            "train_step": train_step,
            "trainer_stall_seconds": trainer_stall_seconds,
            "coverage_complete": coverage_complete,
        },
        "expected": {
            "parquet_files": int(args.expected_parquet_files),
            "bytes": int(args.expected_bytes),
            "train_target_step": train_target_step,
            "train_target_step_configured": int(args.train_target_step),
        },
        "rates": {
            "download_bytes_per_sec": bytes_rate,
            "download_mib_per_sec": (bytes_rate / 1024 / 1024) if bytes_rate else None,
            "download_parquet_per_sec": warm_parquet_rate,
            "sharding_parquet_per_sec": sharded_parquet_rate,
            "manifest_per_sec": manifest_rate,
            "manifest_unique_inputs_per_sec": coverage_rate,
            "manifest_unique_inputs_rate_source": coverage_rate_source,
            "train_steps_per_sec": step_rate,
            "sample_window_seconds": dt,
        },
        "remaining": {
            "download_parquet_files": rem_download_parquet,
            "sharded_parquet_files": rem_sharded_parquet,
            "manifest_unique_input_files": rem_unique_inputs,
            "bytes": rem_bytes,
            "train_steps": rem_steps,
        },
        "eta": eta,
        "eta_human": {
            "download": _fmt_eta(eta["download_seconds"]),
            "download_parquet": _fmt_eta(eta["download_parquet_seconds"]),
            "sharding": _fmt_eta(eta["sharding_seconds"]),
            "manifest_unique_inputs": _fmt_eta(eta["manifest_unique_inputs_seconds"]),
            "train": _fmt_eta(eta["train_seconds"]),
        },
        "active_processes": active,
        "task_status": task_status,
        "supervisor_gate": supervisor_gate,
        "generation_gate_latest": generation_gate_latest,
        "quality_heartbeat": quality_heartbeat,
        "status_confidence": _status_confidence(
            manifest_overlap_inputs=manifest_overlap_inputs,
            manifest_unique_inputs=manifest_unique_inputs,
            expected_parquet_files=int(args.expected_parquet_files),
            coverage_rate=coverage_rate,
            train_rate=step_rate,
            trainer_active=(active.get("trainer", 0) > 0),
            quality_overall=str(quality_heartbeat.get("overall", "unknown")),
        ),
        "system_commands": system_commands,
    }
    return status


def write_reports(status: dict[str, Any], output_json: Path, output_text: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(status, indent=2), encoding="utf-8")

    m = status["metrics"]
    r = status["rates"]
    p = status["active_processes"]
    c = status["system_commands"]
    g = status.get("generation_gate_latest", {})
    q = status.get("quality_heartbeat", {})
    confidence = status.get("status_confidence", {})
    supervisor_gate = status.get("supervisor_gate", "unknown")
    task_status = status.get("task_status", {})
    task_order = [
        "hf_watchdog",
        "download_worker",
        "stage_watchdog",
        "stage_loop",
        "shard_builder",
        "train_supervisor",
        "trainer",
        "eval_runner",
        "generation_gate_runner",
        "shard_offload",
    ]
    task_lines: list[str] = []
    for name in task_order:
        row = task_status.get(name, {})
        state = str(row.get("state", "STOP"))
        count = int(row.get("count", 0) or 0)
        if state == "RUN":
            task_lines.append(f"  {name}=RUN x{count}")
            continue
        reason = str(row.get("reason", "not started"))
        task_lines.append(f"  {name}=STOP reason={reason}")
    text = "\n".join(
        [
            f"time_utc={status['timestamp_utc']}",
            f"warm_parquet={m['warm_parquet_count']} warm_incomplete={m['warm_incomplete_count']}",
            f"warm_bytes={m['warm_bytes']} sharded_parquet={m['sharded_parquet_count']} manifests={m['manifest_count']} manifest_unique_inputs={m['manifest_unique_input_files']} train_step={m['train_step']}",
            "rates:"
            f" download_mib_per_sec={r['download_mib_per_sec']} download_parquet_per_sec={r['download_parquet_per_sec']}"
            f" sharding_parquet_per_sec={r['sharding_parquet_per_sec']} manifest_per_sec={r['manifest_per_sec']}"
            f" manifest_unique_inputs_per_sec={r['manifest_unique_inputs_per_sec']}"
            f" manifest_unique_inputs_rate_source={r.get('manifest_unique_inputs_rate_source')}"
            f" train_steps_per_sec={r['train_steps_per_sec']}",
            f"eta: download_bytes={status['eta_human']['download']} download_parquet={status['eta_human']['download_parquet']}"
            f" sharding={status['eta_human']['sharding']} manifest_unique_inputs={status['eta_human']['manifest_unique_inputs']} train={status['eta_human']['train']}",
            "coverage:"
            f" complete={int(m['coverage_complete'])} overlap_input_files={m['manifest_overlap_input_files']} overlap_manifests={m['manifest_overlap_manifests']}",
            "hotset:"
            f" active_manifests={m['active_manifests']} offloaded_manifests={m['offloaded_manifests']}"
            f" active_manifests_with_symlink_bins={m['active_manifests_with_symlink_bins']}"
            f" trained_batch_names_count={m['trained_batch_names_count']}"
            f" offload_eligible_batches={m['offload_eligible_batches']}"
            f" offload_raw_eligible_batches={m['offload_raw_eligible_batches']}"
            f" offload_max_offloadable_batches={m['offload_max_offloadable_batches']}",
            f"trainer_stall_seconds={m['trainer_stall_seconds']}",
            "active:"
            f" hf_watchdog={p['hf_watchdog']} download_worker={p['download_worker']} stage_watchdog={p['stage_watchdog']} stage_loop={p['stage_loop']}"
            f" shard_builder={p['shard_builder']} train_supervisor={p['train_supervisor']}"
            f" trainer={p['trainer']} eval_runner={p['eval_runner']} generation_gate_runner={p['generation_gate_runner']}"
            f" shard_offload={p['shard_offload']}",
            f"supervisor_gate={supervisor_gate}",
            "tasks:",
            *task_lines,
            "generation_gate_latest:"
            f" step={g.get('step')} rc={g.get('generation_rc')} pass_rate={g.get('pass_rate')} regression_pass={g.get('regression_pass')}",
            "quality_heartbeat:"
            f" overall={q.get('overall')} eval_state={q.get('eval_state')} gen_state={q.get('gen_state')} holdout_state={q.get('holdout_state')}",
            "status_confidence:"
            f" coverage={confidence.get('coverage')} train_eta={confidence.get('train_eta')} quality={confidence.get('quality')} overall_score={confidence.get('overall_score')}",
            "",
            "--- top -b -n 1 ---",
            c["top"]["output"],
            "",
            "--- free -h ---",
            c["free_h"]["output"],
            "",
            "--- nvidia-smi ---",
            c["nvidia_smi"]["output"],
            "",
            "--- df -h ---",
            c["df_h"]["output"],
        ]
    )
    output_text.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.once:
        args.loop = False
    output_json = Path(args.output_json)
    output_text = Path(args.output_text)
    state_path = Path(args.state_file)

    while True:
        status = collect_status(args)
        write_reports(status, output_json, output_text)
        state_snapshot = {
            "ts": time.time(),
            "warm_bytes": status["metrics"]["warm_bytes"],
            "warm_parquet_count": status["metrics"]["warm_parquet_count"],
            "sharded_parquet_count": status["metrics"]["sharded_parquet_count"],
            "manifest_count": status["metrics"]["manifest_count"],
            "manifest_unique_input_files": status["metrics"]["manifest_unique_input_files"],
            "train_step": status["metrics"]["train_step"],
            "train_step_change_ts": (
                time.time() - float(status["metrics"]["trainer_stall_seconds"])
                if status["metrics"]["trainer_stall_seconds"] is not None
                else time.time()
            ),
            "train_step_change_value": status["metrics"]["train_step"],
        }
        _write_state(state_path, state_snapshot)
        print(
            "status_written",
            f"download_eta={status['eta_human']['download']}",
            f"sharding_eta={status['eta_human']['sharding']}",
            f"train_eta={status['eta_human']['train']}",
        )
        if not args.loop:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

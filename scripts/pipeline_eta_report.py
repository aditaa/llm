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


def _pgrep_count(pattern: str) -> int:
    proc = subprocess.run(
        ["pgrep", "-af", "--", pattern],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return 0
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return len(lines)


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


def _eta_seconds(remaining: float, rate_per_sec: float | None) -> float | None:
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
        "--supervisor-state-dir", default="artifacts/reports/train_supervisor_350bt"
    )
    parser.add_argument("--expected-parquet-files", type=int, default=510)
    parser.add_argument("--expected-bytes", type=int, default=1061360917731)
    parser.add_argument("--train-target-step", type=int, default=100000)
    parser.add_argument("--output-json", default="artifacts/reports/pipeline_status.json")
    parser.add_argument("--output-text", default="artifacts/reports/pipeline_status.txt")
    parser.add_argument("--state-file", default="artifacts/reports/pipeline_status_state.json")
    parser.add_argument("--command-timeout-seconds", type=int, default=20)
    parser.add_argument("--command-max-lines", type=int, default=200)
    parser.add_argument("--interval-seconds", type=int, default=60)
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
    sup_dir = Path(args.supervisor_state_dir)

    now = time.time()
    warm_parquet = _count_find(warm_dir, "*.parquet")
    warm_incomplete = _count_find(warm_dir, "*.incomplete")
    warm_bytes = _du_bytes(warm_dir)
    manifests = _count_find(shards_root, "manifest.json")
    sharded_parquet = _count_nonempty_lines(stage_state_dir / "processed_parquet_files.txt")
    train_step = _latest_train_step(sup_dir)
    generation_gate_latest = _latest_generation_summary(sup_dir)

    active = {
        "hf_watchdog": _pgrep_count(r"hf_download_watchdog\.sh"),
        "download_worker": _pgrep_count(r"hf_download_resumable\.sh"),
        "prefetch_worker": _pgrep_count(r"fineweb_prefetch_hot_queue\.sh"),
        "stage_watchdog": _pgrep_count(r"fineweb_stage_shard_watchdog\.sh"),
        "stage_loop": _pgrep_count(r"fineweb_stage_shard_loop\.sh"),
        "shard_builder": _pgrep_count(r"scripts/fineweb_parquet_to_shards\.py"),
        "train_supervisor": _pgrep_count(r"train_supervisor_rtx5070_350bt\.sh"),
        "trainer": _pgrep_count(r"llm\.cli train"),
        "eval_runner": _pgrep_count(r"scripts/eval_checkpoint_prompts\.py"),
        "generation_gate_runner": _pgrep_count(
            r"scripts/eval_checkpoint_prompts\.py .*generation_smoke_suite_v1\.json"
        ),
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
    step_rate = None
    if prev is not None:
        dt = now - float(prev.get("ts", now))
        if dt > 0:
            prev_bytes = int(prev.get("warm_bytes", warm_bytes))
            prev_warm_parquet = int(prev.get("warm_parquet_count", warm_parquet))
            prev_sharded = int(prev.get("sharded_parquet_count", sharded_parquet))
            prev_manifests = int(prev.get("manifest_count", manifests))
            prev_step = int(prev.get("train_step", train_step))
            d_bytes = warm_bytes - prev_bytes
            d_warm_parquet = warm_parquet - prev_warm_parquet
            d_sharded = sharded_parquet - prev_sharded
            d_manifests = manifests - prev_manifests
            d_steps = train_step - prev_step
            if d_bytes >= 0:
                bytes_rate = d_bytes / dt
            if d_warm_parquet >= 0:
                warm_parquet_rate = d_warm_parquet / dt
            if d_sharded >= 0:
                sharded_parquet_rate = d_sharded / dt
            if d_manifests >= 0:
                manifest_rate = d_manifests / dt
            if d_steps >= 0:
                step_rate = d_steps / dt

    rem_bytes = max(0, int(args.expected_bytes) - warm_bytes)
    rem_download_parquet = max(0, int(args.expected_parquet_files) - warm_parquet)
    rem_sharded_parquet = max(0, int(args.expected_parquet_files) - sharded_parquet)
    rem_steps = max(0, int(args.train_target_step) - train_step)

    eta = {
        "download_seconds": _eta_seconds(rem_bytes, bytes_rate),
        "download_parquet_seconds": _eta_seconds(rem_download_parquet, warm_parquet_rate),
        "sharding_seconds": _eta_seconds(rem_sharded_parquet, sharded_parquet_rate),
        "manifests_seconds": _eta_seconds(rem_sharded_parquet, manifest_rate),
        "train_seconds": _eta_seconds(rem_steps, step_rate),
    }

    status = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "warm_parquet_count": warm_parquet,
            "warm_incomplete_count": warm_incomplete,
            "warm_bytes": warm_bytes,
            "manifest_count": manifests,
            "sharded_parquet_count": sharded_parquet,
            "train_step": train_step,
        },
        "expected": {
            "parquet_files": int(args.expected_parquet_files),
            "bytes": int(args.expected_bytes),
            "train_target_step": int(args.train_target_step),
        },
        "rates": {
            "download_bytes_per_sec": bytes_rate,
            "download_mib_per_sec": (bytes_rate / 1024 / 1024) if bytes_rate else None,
            "download_parquet_per_sec": warm_parquet_rate,
            "sharding_parquet_per_sec": sharded_parquet_rate,
            "manifest_per_sec": manifest_rate,
            "train_steps_per_sec": step_rate,
            "sample_window_seconds": dt,
        },
        "remaining": {
            "download_parquet_files": rem_download_parquet,
            "sharded_parquet_files": rem_sharded_parquet,
            "bytes": rem_bytes,
            "train_steps": rem_steps,
        },
        "eta": eta,
        "eta_human": {
            "download": _fmt_eta(eta["download_seconds"]),
            "download_parquet": _fmt_eta(eta["download_parquet_seconds"]),
            "sharding": _fmt_eta(eta["sharding_seconds"]),
            "train": _fmt_eta(eta["train_seconds"]),
        },
        "active_processes": active,
        "generation_gate_latest": generation_gate_latest,
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
    text = "\n".join(
        [
            f"time_utc={status['timestamp_utc']}",
            f"warm_parquet={m['warm_parquet_count']} warm_incomplete={m['warm_incomplete_count']}",
            f"warm_bytes={m['warm_bytes']} sharded_parquet={m['sharded_parquet_count']} manifests={m['manifest_count']} train_step={m['train_step']}",
            "rates:"
            f" download_mib_per_sec={r['download_mib_per_sec']} download_parquet_per_sec={r['download_parquet_per_sec']}"
            f" sharding_parquet_per_sec={r['sharding_parquet_per_sec']} manifest_per_sec={r['manifest_per_sec']}"
            f" train_steps_per_sec={r['train_steps_per_sec']}",
            f"eta: download_bytes={status['eta_human']['download']} download_parquet={status['eta_human']['download_parquet']}"
            f" sharding={status['eta_human']['sharding']} train={status['eta_human']['train']}",
            "active:"
            f" hf_watchdog={p['hf_watchdog']} download_worker={p['download_worker']} prefetch_worker={p['prefetch_worker']} stage_watchdog={p['stage_watchdog']} stage_loop={p['stage_loop']}"
            f" shard_builder={p['shard_builder']} train_supervisor={p['train_supervisor']}"
            f" trainer={p['trainer']} eval_runner={p['eval_runner']} generation_gate_runner={p['generation_gate_runner']}",
            "generation_gate_latest:"
            f" step={g.get('step')} rc={g.get('generation_rc')} pass_rate={g.get('pass_rate')} regression_pass={g.get('regression_pass')}",
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
            "train_step": status["metrics"]["train_step"],
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

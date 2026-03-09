#!/usr/bin/env python3
"""Live terminal monitor for system + project pipeline status (no report files)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class SampleState:
    ts: float | None = None
    warm_bytes: int | None = None
    warm_parquet: int | None = None
    processed_parquet: int | None = None
    manifest_unique_inputs: int | None = None
    manifest_count: int | None = None
    train_step: int | None = None
    train_step_change_ts: float | None = None
    train_step_change_value: int | None = None
    train_sps_estimate: float | None = None


STEP_RE = re.compile(r"step=(\d+)\b")
TARGET_STEP_RE = re.compile(r"target_step=(\d+)\b")
MAX_STEPS_RE = re.compile(r"--max-steps(?:=|\s+)(\d+)\b")
WAIT_MANIFESTS_RE = re.compile(r"waiting_for_manifests have=(\d+) need=(\d+)")
WAIT_UNIQUE_RE = re.compile(r"waiting_for_unique_inputs have=(\d+) need=(\d+)")
WAIT_TRAIN_TOKENS_RE = re.compile(r"waiting_for_train_tokens have_tokens=(\d+) need_tokens=(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument("--warm-dir", default="/mnt/ceph/llm/data/fineweb/sample-350BT")
    parser.add_argument("--hot-dir", default="data/fineweb/sample-350BT/sample/350BT")
    parser.add_argument("--shards-root", default="data/shards_global/fineweb-global-bpe-v1")
    parser.add_argument("--stage-state-dir", default="artifacts/reports/fineweb_stage_shard_loop")
    parser.add_argument(
        "--supervisor-state-dir", default="artifacts/reports/train_supervisor_350bt"
    )
    parser.add_argument("--expected-parquet-files", type=int, default=510)
    parser.add_argument("--expected-bytes", type=int, default=1061360917731)
    parser.add_argument(
        "--train-target-step",
        type=int,
        default=0,
        help="Training target step. 0 means auto-detect from active trainer/supervisor logs.",
    )
    parser.add_argument(
        "--manifest-stall-seconds",
        type=int,
        default=1200,
        help="Warn when manifest count is unchanged for this many seconds and sharding is idle",
    )
    parser.add_argument(
        "--mounts",
        default="/,/mnt/ceph,/mnt/ceph/llm/data",
        help="Comma-separated mount paths for disk usage rows",
    )
    parser.add_argument("--top-procs", type=int, default=5)
    parser.add_argument("--no-alt-screen", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _supports_ansi() -> bool:
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    return term not in ("", "dumb")


def _enter_fullscreen() -> None:
    sys.stdout.write("\x1b[?1049h\x1b[?25l")
    sys.stdout.flush()


def _exit_fullscreen() -> None:
    sys.stdout.write("\x1b[?25h\x1b[?1049l")
    sys.stdout.flush()


def _clear_home() -> None:
    sys.stdout.write("\x1b[H\x1b[2J")
    sys.stdout.flush()


def _trim_line(text: str, width: int) -> str:
    if width <= 4:
        return text[:width]
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _human_bytes(value: float | int) -> str:
    v = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    idx = 0
    while v >= 1024.0 and idx < len(units) - 1:
        v /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(v)}{units[idx]}"
    return f"{v:.2f}{units[idx]}"


def _run_capture(cmd: list[str], timeout: int = 10) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        text = (proc.stdout or proc.stderr or "").strip()
        return proc.returncode, text
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 1, ""


def _count_find(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    rc, text = _run_capture(["find", str(path), "-type", "f", "-name", pattern], timeout=20)
    if rc != 0 or not text:
        return 0
    return len(text.splitlines())


def _du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    rc, text = _run_capture(["du", "-sb", str(path)], timeout=20)
    if rc != 0 or not text:
        return 0
    first = text.split()[0]
    return int(first) if first.isdigit() else 0


def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    unique: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                unique.add(value)
    return len(unique)


def _latest_manifest_mtime(shards_root: Path) -> float | None:
    if not shards_root.exists():
        return None
    latest: float | None = None
    for manifest_path in shards_root.rglob("manifest.json"):
        try:
            mtime = manifest_path.stat().st_mtime
        except OSError:
            continue
        if latest is None or mtime > latest:
            latest = mtime
    return latest


def _file_mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _latest_train_step(supervisor_state_dir: Path) -> int:
    if not supervisor_state_dir.exists():
        return 0
    logs = sorted(
        supervisor_state_dir.glob("train_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for log_path in logs[:5]:
        rc, text = _run_capture(["tail", "-n", "500", str(log_path)], timeout=5)
        if rc != 0:
            continue
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
        rc, text = _run_capture(["tail", "-n", "400", str(log_path)], timeout=5)
        if rc != 0 or not text:
            continue
        for line in reversed(text.splitlines()):
            if "train_launch " not in line:
                continue
            match = TARGET_STEP_RE.search(line)
            if match:
                return int(match.group(1))
    return None


def _active_trainer_target_step() -> int | None:
    rc, text = _run_capture(["pgrep", "-af", r"llm\.cli train"], timeout=5)
    if rc != 0 or not text:
        return None
    targets: list[int] = []
    for line in text.splitlines():
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


def _latest_generation_summary(supervisor_state_dir: Path) -> tuple[str, str, str]:
    trend_path = supervisor_state_dir / "generation_trend.tsv"
    if not trend_path.exists():
        return ("NA", "NA", "NA")
    latest: str | None = None
    with trend_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            row = line.strip()
            if row and not row.startswith("run_tag\t"):
                latest = row
    if latest is None:
        return ("NA", "NA", "NA")
    parts = latest.split("\t")
    # run_tag, step, generation_rc, pass_rate, check_pass_rate, avg_case_score,
    # cases_passed, cases_total, regression_pass, baseline_report, report_json
    if len(parts) < 9:
        return ("NA", "NA", "NA")
    return (parts[2], parts[3], parts[8])


def _latest_supervisor_gate(supervisor_state_dir: Path) -> str:
    if not supervisor_state_dir.exists():
        return "unknown"
    logs = sorted(
        supervisor_state_dir.glob("supervisor_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for log_path in logs[:3]:
        rc, text = _run_capture(["tail", "-n", "200", str(log_path)], timeout=5)
        if rc != 0 or not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            m = WAIT_TRAIN_TOKENS_RE.search(line)
            if m:
                return f"waiting_train_tokens {m.group(1)}/{m.group(2)}"
            m = WAIT_UNIQUE_RE.search(line)
            if m:
                return f"waiting_unique_inputs {m.group(1)}/{m.group(2)}"
            m = WAIT_MANIFESTS_RE.search(line)
            if m:
                return f"waiting_manifests {m.group(1)}/{m.group(2)}"
            if "train_launch " in line:
                return "train_chunk_launching"
    return "unknown"


def _manifest_input_coverage(shards_root: Path) -> tuple[int, int, int]:
    if not shards_root.exists():
        return (0, 0, 0)
    manifests = sorted(shards_root.rglob("manifest.json"))
    file_counts: dict[str, int] = {}
    per_manifest: list[set[str]] = []
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_files = payload.get("input_files", [])
        if not isinstance(raw_files, list):
            continue
        names = {Path(str(raw)).name for raw in raw_files if str(raw).strip()}
        if not names:
            continue
        per_manifest.append(names)
        for name in names:
            file_counts[name] = file_counts.get(name, 0) + 1
    overlap_inputs = sum(1 for count in file_counts.values() if count > 1)
    overlap_manifests = sum(
        1 for names in per_manifest if any(file_counts.get(name, 0) > 1 for name in names)
    )
    return (len(file_counts), overlap_inputs, overlap_manifests)


def _cpu_snapshot() -> tuple[int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        line = handle.readline().strip()
    parts = line.split()
    vals = [int(x) for x in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    total = sum(vals)
    return total, idle


def _cpu_usage(prev: tuple[int, int] | None, curr: tuple[int, int]) -> float | None:
    if prev is None:
        return None
    total_delta = curr[0] - prev[0]
    idle_delta = curr[1] - prev[1]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, (100.0 * (total_delta - idle_delta) / total_delta)))


def _loadavg() -> str:
    try:
        with open("/proc/loadavg", "r", encoding="utf-8") as handle:
            values = handle.read().split()
        return " ".join(values[:3])
    except OSError:
        return "n/a"


def _mem_snapshot() -> tuple[int, int, int, int]:
    total = avail = swap_total = swap_free = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                number = int(value.strip().split()[0]) * 1024
                if key == "MemTotal":
                    total = number
                elif key == "MemAvailable":
                    avail = number
                elif key == "SwapTotal":
                    swap_total = number
                elif key == "SwapFree":
                    swap_free = number
    except OSError:
        pass
    return total, avail, swap_total, swap_free


def _disk_snapshot(paths: list[str]) -> list[str]:
    lines: list[str] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            lines.append(f"{path}: missing")
            continue
        stats = os.statvfs(str(p))
        total = stats.f_blocks * stats.f_frsize
        free = stats.f_bavail * stats.f_frsize
        used = max(0, total - free)
        pct = (100.0 * used / total) if total > 0 else 0.0
        lines.append(
            f"{path}: used={_human_bytes(used)} free={_human_bytes(free)} "
            f"total={_human_bytes(total)} ({pct:.1f}%)"
        )
    return lines


def _gpu_snapshot() -> list[str]:
    rc, text = _run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    if rc != 0 or not text:
        return ["nvidia-smi: unavailable"]
    lines: list[str] = []
    for row in text.splitlines():
        parts = [item.strip() for item in row.split(",")]
        if len(parts) < 8:
            continue
        idx, name, util, mem_used, mem_total, temp, power_draw, power_limit = parts[:8]
        mem_pct = (100.0 * float(mem_used) / float(mem_total)) if float(mem_total) > 0 else 0.0
        lines.append(
            f"GPU{idx} {name}: util={util}% mem={mem_used}/{mem_total}MiB ({mem_pct:.1f}%) "
            f"temp={temp}C power={power_draw}/{power_limit}W"
        )
    rc_apps, apps_text = _run_capture(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    if rc_apps == 0 and apps_text:
        app_rows = [line.strip() for line in apps_text.splitlines() if line.strip()]
        for app in app_rows[:4]:
            lines.append(f"  app {app}")
        if len(app_rows) > 4:
            lines.append(f"  app ... +{len(app_rows) - 4} more")
    return lines if lines else ["nvidia-smi: no data"]


def _task_status(pattern: str) -> tuple[int, list[str]]:
    rc, text = _run_capture(["pgrep", "-af", "--", pattern], timeout=5)
    if rc != 0 or not text:
        return 0, []
    pids: list[int] = []
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if parts and parts[0].isdigit():
            pids.append(int(parts[0]))
    if not pids:
        return 0, []

    pid_set = set(pids)
    root_pids: list[int] = []
    ppids: dict[int, int] = {}
    rc_tree, tree_text = _run_capture(
        ["ps", "-o", "pid=,ppid=", "-p", ",".join(str(pid) for pid in pids)],
        timeout=5,
    )
    if rc_tree == 0 and tree_text:
        for raw in tree_text.splitlines():
            parts = raw.split()
            if len(parts) != 2:
                continue
            if not parts[0].isdigit() or not parts[1].isdigit():
                continue
            pid = int(parts[0])
            ppid = int(parts[1])
            ppids[pid] = ppid
    if ppids:
        for pid in pids:
            ppid = ppids.get(pid)
            if ppid is None or ppid not in pid_set:
                root_pids.append(pid)
    else:
        root_pids = pids[:]
    if not root_pids:
        root_pids = pids[:]

    pid_csv = ",".join(str(pid) for pid in root_pids)
    rc_ps, ps_text = _run_capture(
        ["ps", "-o", "pid=,etime=,pcpu=,pmem=,comm=", "-p", pid_csv],
        timeout=5,
    )
    rows: list[str] = []
    if rc_ps == 0 and ps_text:
        for raw in ps_text.splitlines():
            entry = " ".join(raw.split())
            rows.append(entry)
    return len(root_pids), rows


def _stop_reason(
    task_name: str,
    *,
    coverage_complete: bool,
    warm_parquet: int,
    expected_parquet_files: int,
    hot_parquet: int,
    train_step: int,
    train_target_step: int | None,
    supervisor_gate: str,
    task_counts: dict[str, int],
) -> str:
    if task_name == "hf-watchdog":
        if warm_parquet >= expected_parquet_files:
            return "download complete"
        return "not started"
    if task_name == "download-worker":
        if warm_parquet >= expected_parquet_files:
            return "download complete"
        if task_counts.get("hf-watchdog", 0) > 0:
            return "watchdog-managed; worker idle/restarting"
        return "not started"
    if task_name == "hf-download":
        if warm_parquet >= expected_parquet_files:
            return "download complete"
        if task_counts.get("download-worker", 0) > 0 or task_counts.get("hf-watchdog", 0) > 0:
            return "managed by resumable worker/watchdog"
        return "not started"
    if task_name == "prefetch-worker":
        if coverage_complete:
            return "coverage complete"
        if task_counts.get("stage-loop", 0) > 0:
            return "staging handled by stage-loop"
        return "not started"
    if task_name == "stage-watchdog":
        if coverage_complete:
            return "coverage complete"
        if task_counts.get("stage-loop", 0) > 0:
            return "stage-loop running directly (no watchdog)"
        return "not started"
    if task_name == "stage-loop":
        if coverage_complete:
            return "coverage complete"
        if task_counts.get("stage-watchdog", 0) > 0:
            return "waiting for watchdog restart"
        return "not started"
    if task_name == "shard-builder":
        if coverage_complete:
            return "all expected parquet processed"
        if task_counts.get("stage-loop", 0) > 0:
            if hot_parquet <= 0:
                return "waiting for staged hot parquet"
            return "idle between shard batches"
        return "not started"
    if task_name == "train-supervisor":
        if train_target_step is not None and train_step >= train_target_step:
            return "target step reached"
        if supervisor_gate != "unknown":
            return f"blocked: {supervisor_gate}"
        return "not started"
    if task_name == "trainer":
        if train_target_step is not None and train_step >= train_target_step:
            return "target step reached"
        if task_counts.get("train-supervisor", 0) > 0:
            if supervisor_gate.startswith("waiting_"):
                return f"waiting on supervisor gate ({supervisor_gate})"
            return "idle between chunks/eval"
        return "no active supervisor"
    if task_name == "eval-runner":
        return "runs only during eval windows"
    if task_name == "generation-gate":
        return "runs only on scheduled gate interval"
    if task_name == "zim-offload":
        return "offload worker not started"
    return "not started"


def _top_cpu_processes(limit: int) -> list[str]:
    rc, text = _run_capture(
        ["ps", "-eo", "pid=,pcpu=,pmem=,etime=,comm=", "--sort=-pcpu"],
        timeout=8,
    )
    if rc != 0 or not text:
        return ["unavailable"]
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    return lines[: max(1, limit)]


def _rate(current: int, previous: int | None, dt: float | None) -> float | None:
    if previous is None or dt is None or dt <= 0:
        return None
    delta = current - previous
    if delta <= 0:
        return None
    return delta / dt


def _eta(remaining: float | None, rate_per_sec: float | None) -> str:
    if remaining is None:
        return "unknown"
    if remaining <= 0:
        return "done"
    if rate_per_sec is None or rate_per_sec <= 0:
        return "unknown"
    seconds = remaining / rate_per_sec
    mins = int(seconds // 60)
    hours = mins // 60
    mins = mins % 60
    if hours > 0:
        return f"{hours}h{mins:02d}m"
    return f"{mins}m"


def _render(
    args: argparse.Namespace,
    state: SampleState,
    cpu_prev: tuple[int, int] | None,
    cpu_curr: tuple[int, int],
) -> str:
    now = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    warm_dir = Path(args.warm_dir)
    hot_dir = Path(args.hot_dir)
    shards_root = Path(args.shards_root)
    stage_state = Path(args.stage_state_dir) / "processed_parquet_files.txt"
    sup_dir = Path(args.supervisor_state_dir)

    warm_parquet = _count_find(warm_dir, "*.parquet")
    warm_incomplete = _count_find(warm_dir, "*.incomplete")
    warm_bytes = _du_bytes(warm_dir)
    hot_parquet = _count_find(hot_dir, "*.parquet")
    hot_bytes = _du_bytes(hot_dir)
    manifest_count = _count_find(shards_root, "manifest.json")
    manifest_unique_inputs, manifest_overlap_inputs, manifest_overlap_manifests = (
        _manifest_input_coverage(shards_root)
    )
    processed_parquet = _count_nonempty_lines(stage_state)
    train_step = _latest_train_step(sup_dir)
    train_target_step = _effective_train_target_step(args.train_target_step, sup_dir, train_step)
    supervisor_gate = _latest_supervisor_gate(sup_dir)
    gen_rc, gen_pass_rate, gen_regression_pass = _latest_generation_summary(sup_dir)

    dt = (now - state.ts) if state.ts is not None else None
    download_bps = _rate(warm_bytes, state.warm_bytes, dt)
    download_pps = _rate(warm_parquet, state.warm_parquet, dt)
    shard_pps = _rate(processed_parquet, state.processed_parquet, dt)
    coverage_pps = _rate(manifest_unique_inputs, state.manifest_unique_inputs, dt)
    train_sps = _rate(train_step, state.train_step, dt)
    if train_sps is None:
        if state.train_step_change_value is None:
            state.train_step_change_value = train_step
            state.train_step_change_ts = now
        elif train_step > state.train_step_change_value:
            if state.train_step_change_ts is not None:
                d_steps = train_step - state.train_step_change_value
                d_secs = now - state.train_step_change_ts
                if d_steps > 0 and d_secs > 0:
                    state.train_sps_estimate = d_steps / d_secs
            state.train_step_change_value = train_step
            state.train_step_change_ts = now
        elif train_step < state.train_step_change_value:
            state.train_step_change_value = train_step
            state.train_step_change_ts = now
            state.train_sps_estimate = None
        train_sps = state.train_sps_estimate

    latest_manifest_mtime = _latest_manifest_mtime(shards_root)
    manifest_stall_age = (
        None if latest_manifest_mtime is None else max(0.0, now - latest_manifest_mtime)
    )
    processed_mtime = _file_mtime(stage_state)
    processed_stall_age = None if processed_mtime is None else max(0.0, now - processed_mtime)

    state.ts = now
    state.warm_bytes = warm_bytes
    state.warm_parquet = warm_parquet
    state.processed_parquet = processed_parquet
    state.manifest_unique_inputs = manifest_unique_inputs
    state.manifest_count = manifest_count
    state.train_step = train_step

    cpu_usage = _cpu_usage(cpu_prev, cpu_curr)
    mem_total, mem_avail, swap_total, swap_free = _mem_snapshot()
    mem_used = max(0, mem_total - mem_avail)
    swap_used = max(0, swap_total - swap_free)
    load = _loadavg()
    gpu_lines = _gpu_snapshot()
    disk_lines = _disk_snapshot([p.strip() for p in args.mounts.split(",") if p.strip()])
    top_procs = _top_cpu_processes(args.top_procs)

    tasks = [
        ("hf-watchdog", r"hf_download_watchdog\.sh"),
        ("download-worker", r"hf_download_resumable\.sh"),
        ("prefetch-worker", r"fineweb_prefetch_hot_queue\.sh"),
        ("stage-watchdog", r"fineweb_stage_shard_watchdog\.sh"),
        ("hf-download", r"\.venv/bin/hf download HuggingFaceFW/fineweb"),
        ("stage-loop", r"fineweb_stage_shard_loop\.sh"),
        ("shard-builder", r"scripts/fineweb_parquet_to_shards\.py"),
        ("train-supervisor", r"bash scripts/train_supervisor_rtx5070_350bt\.sh"),
        ("trainer", r"llm\.cli train"),
        ("eval-runner", r"eval_checkpoint_prompts\.py"),
        ("generation-gate", r"eval_checkpoint_prompts\.py .*generation_smoke_suite_v1\.json"),
        ("zim-offload", r"zim_offload_worker\.sh"),
    ]

    coverage_complete = (
        warm_parquet >= int(args.expected_parquet_files)
        and processed_parquet >= int(args.expected_parquet_files)
        and manifest_unique_inputs >= int(args.expected_parquet_files)
    )

    task_lines: list[str] = []
    task_counts: dict[str, int] = {}
    task_rows: dict[str, list[str]] = {}
    for name, pattern in tasks:
        count, rows = _task_status(pattern)
        task_counts[name] = count
        task_rows[name] = rows
    for name, _ in tasks:
        count = task_counts.get(name, 0)
        if count <= 0:
            reason = _stop_reason(
                name,
                coverage_complete=coverage_complete,
                warm_parquet=warm_parquet,
                expected_parquet_files=int(args.expected_parquet_files),
                hot_parquet=hot_parquet,
                train_step=train_step,
                train_target_step=train_target_step,
                supervisor_gate=supervisor_gate,
                task_counts=task_counts,
            )
            task_lines.append(f"{name:16} STOP | {reason}")
            continue
        rows = task_rows.get(name, [])
        summary = rows[0] if rows else "running"
        task_lines.append(f"{name:16} RUN x{count} | {summary}")

    rem_bytes = max(0, int(args.expected_bytes) - warm_bytes)
    rem_files = max(0, int(args.expected_parquet_files) - warm_parquet)
    rem_manifest_unique = max(0, int(args.expected_parquet_files) - manifest_unique_inputs)
    rem_steps: int | None
    if train_target_step is None:
        rem_steps = None
    else:
        rem_steps = max(0, int(train_target_step) - train_step)
    rate_mib = (download_bps / 1024 / 1024) if download_bps is not None else None
    cpu_text = f"{cpu_usage:.1f}%" if cpu_usage is not None else "warming"
    mem_pct = (100.0 * mem_used / mem_total) if mem_total > 0 else 0.0
    swap_pct = (100.0 * swap_used / swap_total) if swap_total > 0 else 0.0
    alerts: list[str] = []
    if task_counts.get("stage-watchdog", 0) > 1:
        alerts.append(f"multiple stage watchdogs detected ({task_counts.get('stage-watchdog', 0)})")
    if task_counts.get("stage-loop", 0) > 1:
        alerts.append(f"multiple stage loops detected ({task_counts.get('stage-loop', 0)})")
    if task_counts.get("stage-watchdog", 0) == 0 and task_counts.get("stage-loop", 0) == 0:
        alerts.append("stage pipeline controller is not running")
    if (
        not coverage_complete
        and task_counts.get("stage-loop", 0) > 0
        and task_counts.get("shard-builder", 0) == 0
        and manifest_stall_age is not None
        and manifest_stall_age >= float(args.manifest_stall_seconds)
    ):
        mins = int(manifest_stall_age // 60)
        alerts.append(f"manifest count stalled for {mins}m with no active shard-builder")
    if (
        not coverage_complete
        and task_counts.get("stage-loop", 0) > 0
        and task_counts.get("shard-builder", 0) == 0
        and processed_stall_age is not None
        and processed_stall_age >= float(args.manifest_stall_seconds)
    ):
        mins = int(processed_stall_age // 60)
        alerts.append(f"processed parquet count stalled for {mins}m with no active shard-builder")

    lines: list[str] = []
    lines.append(f"LLM Live Monitor | {ts} | refresh={args.refresh_seconds:.1f}s | ctrl+c to exit")
    lines.append("-" * 120)
    lines.append(f"CPU: {cpu_text}  load(1/5/15): {load}")
    lines.append(
        "MEM: "
        f"{_human_bytes(mem_used)}/{_human_bytes(mem_total)} ({mem_pct:.1f}%)  "
        f"SWAP: {_human_bytes(swap_used)}/{_human_bytes(swap_total)} ({swap_pct:.1f}%)"
    )
    lines.extend([f"DISK: {line}" for line in disk_lines])
    lines.extend([f"GPU:  {line}" for line in gpu_lines])

    lines.append("")
    lines.append("Pipeline Progress")
    lines.append(
        f"  Download: warm_parquet={warm_parquet}/{args.expected_parquet_files} "
        f"incomplete={warm_incomplete} "
        f"warm_size={_human_bytes(warm_bytes)}/{_human_bytes(args.expected_bytes)} "
        f"rate={f'{rate_mib:.2f}MiB/s' if rate_mib is not None else 'n/a'} "
        f"eta={_eta(rem_bytes, download_bps)}"
    )
    lines.append(
        f"  Staging:  hot_parquet={hot_parquet} hot_size={_human_bytes(hot_bytes)} "
        f"processed={processed_parquet}/{args.expected_parquet_files} "
        f"rate={f'{shard_pps:.3f} files/s' if shard_pps is not None else 'n/a'}"
    )
    lines.append(
        f"  Shards:   manifests={manifest_count} "
        f"download_file_eta={_eta(rem_files, download_pps)}"
    )
    lines.append(
        f"  Coverage: manifest_unique={manifest_unique_inputs}/{args.expected_parquet_files} "
        f"remaining={rem_manifest_unique} overlap_inputs={manifest_overlap_inputs} "
        f"overlap_manifests={manifest_overlap_manifests} "
        f"rate={f'{coverage_pps:.3f} files/s' if coverage_pps is not None else 'n/a'} "
        f"eta={_eta(rem_manifest_unique, coverage_pps)} "
        f"complete={int(coverage_complete)}"
    )
    train_target_text = str(train_target_step) if train_target_step is not None else "?"
    lines.append(
        f"  Training: step={train_step}/{train_target_text} "
        f"rate={f'{train_sps:.3f} step/s' if train_sps is not None else 'n/a'} "
        f"eta={_eta(rem_steps, train_sps)}"
    )
    if manifest_stall_age is not None:
        processed_stall_secs = int(processed_stall_age or 0)
        lines.append(
            f"  StallAge: manifest={int(manifest_stall_age)}s "
            f"processed={processed_stall_secs}s "
            f"threshold={args.manifest_stall_seconds}s"
        )
    lines.append(f"  Supervisor: gate={supervisor_gate}")
    lines.append(
        f"  GenGate:  latest_rc={gen_rc} latest_pass_rate={gen_pass_rate} "
        f"latest_regression_pass={gen_regression_pass}"
    )

    lines.append("")
    lines.append("Alerts")
    if alerts:
        lines.extend([f"  ALERT: {line}" for line in alerts])
    else:
        lines.append("  none")

    lines.append("")
    lines.append("Task Status")
    lines.extend([f"  {line}" for line in task_lines])

    lines.append("")
    lines.append("Top CPU Processes")
    lines.extend([f"  {line}" for line in top_procs])

    width, height = shutil.get_terminal_size(fallback=(140, 45))
    width = max(80, width - 1)
    trimmed = [_trim_line(line, width) for line in lines]
    if len(trimmed) > height:
        trimmed = trimmed[: max(1, height - 1)]
        trimmed.append(_trim_line(f"... clipped ({len(lines) - len(trimmed)} more lines)", width))
    return "\n".join(trimmed) + "\n"


def main() -> int:
    args = parse_args()
    ansi = _supports_ansi()
    use_alt = ansi and not args.no_alt_screen and not args.once

    state = SampleState()
    cpu_prev: tuple[int, int] | None = None
    try:
        if use_alt:
            _enter_fullscreen()
        while True:
            cpu_curr = _cpu_snapshot()
            output = _render(args, state, cpu_prev, cpu_curr)
            cpu_prev = cpu_curr

            if ansi:
                _clear_home()
            print(output, end="")
            sys.stdout.flush()
            if args.once:
                break
            time.sleep(max(0.5, args.refresh_seconds))
    except KeyboardInterrupt:
        return 0
    finally:
        if use_alt:
            _exit_fullscreen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

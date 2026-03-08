#!/usr/bin/env python3
"""Live terminal monitor for system + project pipeline status (no report files)."""

from __future__ import annotations

import argparse
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
    train_step: int | None = None


STEP_RE = re.compile(r"step=(\d+)\b")


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
    parser.add_argument("--train-target-step", type=int, default=100000)
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
    # run_tag,step,generation_rc,pass_rate,check_pass_rate,avg_case_score,cases_passed,cases_total,regression_pass,baseline_report,report_json
    if len(parts) < 9:
        return ("NA", "NA", "NA")
    return (parts[2], parts[3], parts[8])


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
            f"{path}: used={_human_bytes(used)} free={_human_bytes(free)} total={_human_bytes(total)} ({pct:.1f}%)"
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
            f"GPU{idx} {name}: util={util}% mem={mem_used}/{mem_total}MiB ({mem_pct:.1f}%) temp={temp}C power={power_draw}/{power_limit}W"
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

    pid_csv = ",".join(str(pid) for pid in pids)
    rc_ps, ps_text = _run_capture(["ps", "-o", "pid=,etime=,pcpu=,pmem=,comm=", "-p", pid_csv], timeout=5)
    rows: list[str] = []
    if rc_ps == 0 and ps_text:
        for raw in ps_text.splitlines():
            entry = " ".join(raw.split())
            rows.append(entry)
    return len(pids), rows


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
    if delta < 0:
        return None
    return delta / dt


def _eta(remaining: float, rate_per_sec: float | None) -> str:
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
    processed_parquet = _count_nonempty_lines(stage_state)
    train_step = _latest_train_step(sup_dir)
    gen_rc, gen_pass_rate, gen_regression_pass = _latest_generation_summary(sup_dir)

    dt = (now - state.ts) if state.ts is not None else None
    download_bps = _rate(warm_bytes, state.warm_bytes, dt)
    download_pps = _rate(warm_parquet, state.warm_parquet, dt)
    shard_pps = _rate(processed_parquet, state.processed_parquet, dt)
    train_sps = _rate(train_step, state.train_step, dt)

    state.ts = now
    state.warm_bytes = warm_bytes
    state.warm_parquet = warm_parquet
    state.processed_parquet = processed_parquet
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
        ("train-supervisor", r"train_supervisor_rtx5070_350bt\.sh"),
        ("trainer", r"llm\.cli train"),
        ("eval-runner", r"eval_checkpoint_prompts\.py"),
        ("generation-gate", r"eval_checkpoint_prompts\.py .*generation_smoke_suite_v1\.json"),
        ("zim-offload", r"zim_offload_worker\.sh"),
    ]
    task_lines: list[str] = []
    for name, pattern in tasks:
        count, rows = _task_status(pattern)
        if count <= 0:
            task_lines.append(f"{name:16} STOP")
            continue
        summary = rows[0] if rows else "running"
        task_lines.append(f"{name:16} RUN x{count} | {summary}")

    rem_bytes = max(0, int(args.expected_bytes) - warm_bytes)
    rem_files = max(0, int(args.expected_parquet_files) - warm_parquet)
    rem_steps = max(0, int(args.train_target_step) - train_step)

    rate_mib = (download_bps / 1024 / 1024) if download_bps is not None else None
    cpu_text = f"{cpu_usage:.1f}%" if cpu_usage is not None else "warming"
    mem_pct = (100.0 * mem_used / mem_total) if mem_total > 0 else 0.0
    swap_pct = (100.0 * swap_used / swap_total) if swap_total > 0 else 0.0

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
        f"incomplete={warm_incomplete} warm_size={_human_bytes(warm_bytes)}/{_human_bytes(args.expected_bytes)} "
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
        f"  Training: step={train_step}/{args.train_target_step} "
        f"rate={f'{train_sps:.3f} step/s' if train_sps is not None else 'n/a'} "
        f"eta={_eta(rem_steps, train_sps)}"
    )
    lines.append(
        f"  GenGate:  latest_rc={gen_rc} latest_pass_rate={gen_pass_rate} latest_regression_pass={gen_regression_pass}"
    )

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

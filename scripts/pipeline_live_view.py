#!/usr/bin/env python3
"""Render a live terminal view of pipeline status."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", default="artifacts/reports/pipeline_status.json")
    parser.add_argument("--status-text", default="artifacts/reports/pipeline_status.txt")
    parser.add_argument("--state-file", default="artifacts/reports/pipeline_status_state.json")
    parser.add_argument("--reporter-script", default="scripts/pipeline_eta_report.py")
    parser.add_argument("--refresh-seconds", type=int, default=5)
    parser.add_argument("--command-timeout-seconds", type=int, default=20)
    parser.add_argument("--command-max-lines", type=int, default=120)
    parser.add_argument("--no-reporter-refresh", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--top-lines", type=int, default=35)
    parser.add_argument("--free-lines", type=int, default=20)
    parser.add_argument("--nvidia-lines", type=int, default=25)
    parser.add_argument("--df-lines", type=int, default=25)
    return parser.parse_args()


def _run_reporter(args: argparse.Namespace) -> None:
    if args.no_reporter_refresh:
        return
    cmd = [
        sys.executable,
        args.reporter_script,
        "--output-json",
        args.status_json,
        "--output-text",
        args.status_text,
        "--state-file",
        args.state_file,
        "--command-timeout-seconds",
        str(args.command_timeout_seconds),
        "--command-max-lines",
        str(args.command_max_lines),
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _clip_block(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    clipped = lines[:max_lines]
    clipped.append(f"... ({len(lines) - max_lines} more lines)")
    return "\n".join(clipped)


def _read_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _render(status: dict[str, Any], args: argparse.Namespace) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = status.get("metrics", {})
    rates = status.get("rates", {})
    eta = status.get("eta_human", {})
    active = status.get("active_processes", {})
    sys_cmds = status.get("system_commands", {})

    top_out = _clip_block(sys_cmds.get("top", {}).get("output", ""), args.top_lines)
    free_out = _clip_block(sys_cmds.get("free_h", {}).get("output", ""), args.free_lines)
    nvidia_out = _clip_block(sys_cmds.get("nvidia_smi", {}).get("output", ""), args.nvidia_lines)
    df_out = _clip_block(sys_cmds.get("df_h", {}).get("output", ""), args.df_lines)

    lines = [
        f"Pipeline Live View  |  refreshed={ts}  |  ctrl+c to exit",
        "=" * 78,
        (
            f"metrics: warm_parquet={metrics.get('warm_parquet_count')} "
            f"warm_incomplete={metrics.get('warm_incomplete_count')} "
            f"warm_bytes={metrics.get('warm_bytes')} "
            f"sharded_parquet={metrics.get('sharded_parquet_count')} "
            f"manifests={metrics.get('manifest_count')} "
            f"train_step={metrics.get('train_step')}"
        ),
        (
            f"rates: download_mib_per_sec={rates.get('download_mib_per_sec')} "
            f"download_parquet_per_sec={rates.get('download_parquet_per_sec')} "
            f"sharding_parquet_per_sec={rates.get('sharding_parquet_per_sec')} "
            f"train_steps_per_sec={rates.get('train_steps_per_sec')}"
        ),
        (
            f"eta: download={eta.get('download')} "
            f"download_parquet={eta.get('download_parquet')} "
            f"sharding={eta.get('sharding')} "
            f"train={eta.get('train')}"
        ),
        (
            f"active: download_worker={active.get('download_worker')} "
            f"stage_loop={active.get('stage_loop')} "
            f"shard_builder={active.get('shard_builder')} "
            f"train_supervisor={active.get('train_supervisor')} "
            f"trainer={active.get('trainer')} "
            f"eval_runner={active.get('eval_runner')}"
        ),
        "",
        "--- top -b -n 1 ---",
        top_out,
        "",
        "--- free -h ---",
        free_out,
        "",
        "--- nvidia-smi ---",
        nvidia_out,
        "",
        "--- df -h ---",
        df_out,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    status_path = Path(args.status_json)

    try:
        while True:
            _run_reporter(args)
            status = _read_status(status_path)
            print("\033[2J\033[H", end="")
            if status is None:
                print(
                    f"waiting for valid status json at {status_path} "
                    f"(refresh={args.refresh_seconds}s)"
                )
            else:
                print(_render(status, args), end="")
            if args.once:
                break
            time.sleep(max(1, args.refresh_seconds))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

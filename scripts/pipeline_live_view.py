#!/usr/bin/env python3
"""Render a live terminal view of pipeline status."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    parser.add_argument(
        "--view-mode",
        choices=("rotate", "fit", "full"),
        default="rotate",
        help="rotate: one detailed section per refresh, fit: all sections clipped to screen, full: all sections by fixed limits",
    )
    parser.add_argument("--no-fit-terminal", action="store_true")
    parser.add_argument("--no-alt-screen", action="store_true")
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


def _trim_line(line: str, width: int) -> str:
    if width <= 4:
        return line[:width]
    if len(line) <= width:
        return line
    return line[: width - 3] + "..."


def _trim_block(text: str, width: int) -> str:
    return "\n".join(_trim_line(line, width) for line in text.splitlines())


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


def _terminal_size() -> tuple[int, int]:
    size = shutil.get_terminal_size(fallback=(120, 40))
    return max(40, size.columns), max(20, size.lines)


def _allocate_section_lines(total: int) -> tuple[int, int, int, int]:
    if total <= 0:
        return 0, 0, 0, 0
    top = max(3, int(total * 0.45))
    nvidia = max(3, int(total * 0.25))
    df = max(2, int(total * 0.20))
    free = total - top - nvidia - df
    while free < 2 and top > 3:
        top -= 1
        free += 1
    while free < 2 and nvidia > 3:
        nvidia -= 1
        free += 1
    while free < 2 and df > 2:
        df -= 1
        free += 1
    if free < 2:
        free = max(1, free)
    return top, free, nvidia, df


def _read_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _render(status: dict[str, Any], args: argparse.Namespace, frame_index: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics = status.get("metrics", {})
    rates = status.get("rates", {})
    eta = status.get("eta_human", {})
    active = status.get("active_processes", {})
    sys_cmds = status.get("system_commands", {})
    term_width, term_height = _terminal_size()
    usable_width = term_width if args.no_fit_terminal else max(40, term_width - 1)

    top_cap = args.top_lines
    free_cap = args.free_lines
    nvidia_cap = args.nvidia_lines
    df_cap = args.df_lines
    fit_mode = args.view_mode == "fit" and not args.no_fit_terminal
    if fit_mode:
        reserved = 10
        available = max(0, term_height - reserved)
        top_fit, free_fit, nvidia_fit, df_fit = _allocate_section_lines(available)
        top_cap = min(top_cap, top_fit)
        free_cap = min(free_cap, free_fit)
        nvidia_cap = min(nvidia_cap, nvidia_fit)
        df_cap = min(df_cap, df_fit)
    elif args.view_mode == "rotate":
        reserved = 9
        available = max(6, term_height - reserved)
        sections = ("top", "free_h", "nvidia_smi", "df_h")
        selected = sections[frame_index % len(sections)]
        top_cap = available if selected == "top" else 0
        free_cap = available if selected == "free_h" else 0
        nvidia_cap = available if selected == "nvidia_smi" else 0
        df_cap = available if selected == "df_h" else 0

    top_out = _trim_block(_clip_block(sys_cmds.get("top", {}).get("output", ""), top_cap), usable_width) if top_cap > 0 else ""
    free_out = _trim_block(_clip_block(sys_cmds.get("free_h", {}).get("output", ""), free_cap), usable_width) if free_cap > 0 else ""
    nvidia_out = _trim_block(
        _clip_block(sys_cmds.get("nvidia_smi", {}).get("output", ""), nvidia_cap), usable_width
    ) if nvidia_cap > 0 else ""
    df_out = _trim_block(_clip_block(sys_cmds.get("df_h", {}).get("output", ""), df_cap), usable_width) if df_cap > 0 else ""

    lines = [
        _trim_line(f"Pipeline Live View  |  refreshed={ts}  |  ctrl+c to exit", usable_width),
        _trim_line("=" * usable_width, usable_width),
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
        f"view_mode={args.view_mode} refresh={args.refresh_seconds}s",
        "",
    ]

    if args.view_mode == "rotate":
        rotate_order = ["TOP", "FREE -H", "NVIDIA-SMI", "DF -H"]
        selected_idx = frame_index % 4
        selected_name = rotate_order[selected_idx]
        lines.append(f"[ROTATE] showing={selected_name} next={rotate_order[(selected_idx + 1) % 4]}")
        lines.append("")
        if selected_idx == 0:
            lines.append("[TOP]")
            lines.append(top_out)
        elif selected_idx == 1:
            lines.append("[FREE -H]")
            lines.append(free_out)
        elif selected_idx == 2:
            lines.append("[NVIDIA-SMI]")
            lines.append(nvidia_out)
        else:
            lines.append("[DF -H]")
            lines.append(df_out)
    else:
        lines.extend(
            [
                "[TOP]",
                top_out,
                "",
                "[FREE -H]",
                free_out,
                "",
                "[NVIDIA-SMI]",
                nvidia_out,
                "",
                "[DF -H]",
                df_out,
            ]
        )
    return "\n".join(_trim_line(line, usable_width) for line in lines) + "\n"


def main() -> int:
    args = parse_args()
    status_path = Path(args.status_json)

    ansi = _supports_ansi()
    use_alt_screen = ansi and not args.no_alt_screen and not args.once
    frame_index = 0
    try:
        if use_alt_screen:
            _enter_fullscreen()
        while True:
            _run_reporter(args)
            status = _read_status(status_path)
            if ansi:
                _clear_home()
            if status is None:
                msg = (
                    f"waiting for valid status json at {status_path} "
                    f"(refresh={args.refresh_seconds}s)"
                )
                print(msg)
            else:
                print(_render(status, args, frame_index), end="")
            frame_index += 1
            sys.stdout.flush()
            if args.once:
                break
            time.sleep(max(1, args.refresh_seconds))
    except KeyboardInterrupt:
        return 0
    finally:
        if use_alt_screen:
            _exit_fullscreen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

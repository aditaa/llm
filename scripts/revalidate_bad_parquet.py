#!/usr/bin/env python3
"""Revalidate bad parquet entries against warm storage and optionally restage valid files."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    name: str
    status: str
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bad-list",
        default="artifacts/reports/fineweb_stage_shard_loop/bad_parquet_files.txt",
        help="Path to bad parquet basename list",
    )
    parser.add_argument(
        "--warm-dir",
        default="/mnt/ceph/llm/data/fineweb/sample-350BT/sample/350BT",
        help="Warm parquet directory",
    )
    parser.add_argument("--field", default="text", help="Expected parquet text field")
    parser.add_argument(
        "--report-output",
        default="",
        help=(
            "Report JSON path. Default: "
            "artifacts/reports/fineweb_stage_shard_loop/bad_parquet_revalidate_<ts>.json"
        ),
    )
    parser.add_argument(
        "--no-rewrite-bad-list",
        action="store_true",
        help="Do not rewrite bad-list with retained invalid/missing entries",
    )
    parser.add_argument(
        "--restage-valid",
        action="store_true",
        help="Copy valid reinstated parquet files from warm to hot",
    )
    parser.add_argument(
        "--hot-dir",
        default="data/fineweb/sample-350BT/sample/350BT",
        help="Hot parquet directory used with --restage-valid",
    )
    parser.add_argument(
        "--max-restage-files",
        type=int,
        default=0,
        help="Max valid files to restage (0 = all)",
    )
    parser.add_argument(
        "--min-free-gib",
        type=int,
        default=80,
        help="Destination free-space floor before restage copies",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing/copying",
    )
    parser.add_argument(
        "--quarantine-dir",
        default="artifacts/reports/fineweb_stage_shard_loop/quarantine_bad_parquet",
        help="Directory containing quarantined parquet copies (*.bad_<ts>)",
    )
    parser.add_argument(
        "--quarantine-keep-per-name",
        type=int,
        default=1,
        help=(
            "For entries still marked bad, keep at most this many newest quarantine "
            "copies per parquet basename (default: 1)"
        ),
    )
    parser.add_argument(
        "--no-prune-quarantine",
        action="store_true",
        help="Disable quarantine directory pruning",
    )
    return parser.parse_args()


def _validate_parquet(path: Path, field: str) -> tuple[bool, str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required: run make setup-train") from exc

    try:
        table = pq.ParquetFile(path)
        meta = table.metadata
        if meta is None or meta.num_row_groups <= 0:
            return False, "missing_row_groups"
        if meta.num_rows <= 0:
            return False, "no_rows"
        if field not in table.schema.names:
            return False, f"missing_field:{field}"
    except Exception as exc:  # noqa: BLE001
        return False, f"read_error:{exc}"
    return True, "ok"


def _read_bad_list(path: Path) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        name = raw.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _write_bad_list(path: Path, names: list[str], dry_run: bool) -> None:
    lines = "".join(f"{name}\n" for name in sorted(set(names)))
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lines, encoding="utf-8")


def _free_bytes(path: Path) -> int:
    stats = shutil.disk_usage(path)
    return int(stats.free)


def _rsync_atomic(src: Path, dest: Path, dry_run: bool) -> tuple[bool, str]:
    tmp = dest.with_name(dest.name + ".incomplete")
    if dry_run:
        return True, "dry_run"
    if tmp.exists():
        tmp.unlink()
    cmd = ["rsync", "-ah", "--partial", "--inplace", str(src), str(tmp)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        err = (proc.stderr or proc.stdout or "").strip()
        return False, f"rsync_failed:{err[:240]}"
    src_size = src.stat().st_size
    tmp_size = tmp.stat().st_size if tmp.exists() else -1
    if src_size != tmp_size:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False, f"size_mismatch:src={src_size}:tmp={tmp_size}"
    tmp.replace(dest)
    return True, "ok"


def _quarantine_base_name(path: Path) -> str:
    marker = ".bad_"
    name = path.name
    idx = name.rfind(marker)
    if idx == -1:
        return name
    return name[:idx]


def _prune_quarantine(
    quarantine_dir: Path,
    *,
    retained_bad: set[str],
    valid_names: set[str],
    keep_per_name: int,
    dry_run: bool,
) -> dict[str, int]:
    summary = {
        "quarantine_files_before": 0,
        "quarantine_files_removed": 0,
        "quarantine_files_after": 0,
        "quarantine_bytes_removed": 0,
    }
    if not quarantine_dir.exists():
        return summary

    keep_limit = max(0, keep_per_name)
    by_name: dict[str, list[Path]] = {}
    for path in sorted(quarantine_dir.iterdir()):
        if not path.is_file():
            continue
        base = _quarantine_base_name(path)
        by_name.setdefault(base, []).append(path)
        summary["quarantine_files_before"] += 1

    remove_paths: list[Path] = []
    for base, paths in by_name.items():
        # If source is now valid (or not currently tracked as bad), drop all stale quarantine copies.
        if base in valid_names or base not in retained_bad:
            remove_paths.extend(paths)
            continue

        # For currently bad files, keep only N newest quarantine copies.
        paths_sorted = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
        remove_paths.extend(paths_sorted[keep_limit:])

    for path in remove_paths:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if not dry_run:
            try:
                path.unlink()
            except OSError:
                continue
        summary["quarantine_files_removed"] += 1
        summary["quarantine_bytes_removed"] += int(size)

    summary["quarantine_files_after"] = (
        summary["quarantine_files_before"] - summary["quarantine_files_removed"]
    )
    return summary


def main() -> int:
    args = parse_args()
    bad_list = Path(args.bad_list)
    warm_dir = Path(args.warm_dir)
    hot_dir = Path(args.hot_dir)
    quarantine_dir = Path(args.quarantine_dir)

    if not bad_list.exists():
        raise FileNotFoundError(f"bad-list not found: {bad_list}")
    if not warm_dir.exists():
        raise FileNotFoundError(f"warm-dir not found: {warm_dir}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_output = (
        Path(args.report_output)
        if args.report_output
        else Path(
            f"artifacts/reports/fineweb_stage_shard_loop/bad_parquet_revalidate_{ts}.json"
        )
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)

    bad_names = _read_bad_list(bad_list)
    results: list[ValidationResult] = []

    valid_names: list[str] = []
    retained_bad_names: list[str] = []
    for name in bad_names:
        warm_path = warm_dir / name
        if not warm_path.exists():
            results.append(ValidationResult(name=name, status="missing_warm"))
            retained_bad_names.append(name)
            continue
        ok, detail = _validate_parquet(warm_path, args.field)
        if ok:
            results.append(ValidationResult(name=name, status="valid", detail=detail))
            valid_names.append(name)
        else:
            results.append(ValidationResult(name=name, status="invalid", detail=detail))
            retained_bad_names.append(name)

    rewritten = False
    if not args.no_rewrite_bad_list:
        _write_bad_list(bad_list, retained_bad_names, args.dry_run)
        rewritten = not args.dry_run

    restage_attempted = 0
    restage_ok = 0
    restage_skipped_existing = 0
    restage_skipped_space = 0
    restage_failures: list[dict[str, str]] = []

    if args.restage_valid:
        if not hot_dir.exists():
            raise FileNotFoundError(f"hot-dir not found: {hot_dir}")
        min_free_bytes = int(args.min_free_gib) * 1024 * 1024 * 1024
        limit = int(args.max_restage_files)
        candidates = valid_names[: limit if limit > 0 else len(valid_names)]
        for name in candidates:
            warm_path = warm_dir / name
            hot_path = hot_dir / name
            src_size = warm_path.stat().st_size
            if hot_path.exists() and hot_path.stat().st_size == src_size:
                restage_skipped_existing += 1
                continue
            free_now = _free_bytes(hot_dir)
            if free_now - src_size < min_free_bytes:
                restage_skipped_space += 1
                continue
            restage_attempted += 1
            ok, detail = _rsync_atomic(warm_path, hot_path, args.dry_run)
            if ok:
                restage_ok += 1
            else:
                restage_failures.append({"name": name, "error": detail})

    quarantine_summary = {
        "quarantine_files_before": 0,
        "quarantine_files_removed": 0,
        "quarantine_files_after": 0,
        "quarantine_bytes_removed": 0,
    }
    if not args.no_prune_quarantine:
        quarantine_summary = _prune_quarantine(
            quarantine_dir,
            retained_bad=set(retained_bad_names),
            valid_names=set(valid_names),
            keep_per_name=int(args.quarantine_keep_per_name),
            dry_run=bool(args.dry_run),
        )

    summary = {
        "bad_list": str(bad_list),
        "warm_dir": str(warm_dir),
        "hot_dir": str(hot_dir),
        "quarantine_dir": str(quarantine_dir),
        "field": args.field,
        "input_bad_entries": len(bad_names),
        "valid_reinstated": len(valid_names),
        "retained_bad_entries": len(retained_bad_names),
        "rewrote_bad_list": rewritten,
        "dry_run": bool(args.dry_run),
        "restage_valid": bool(args.restage_valid),
        "restage_attempted": restage_attempted,
        "restage_ok": restage_ok,
        "restage_skipped_existing": restage_skipped_existing,
        "restage_skipped_space": restage_skipped_space,
        "restage_failures": restage_failures,
        "pruned_quarantine": not bool(args.no_prune_quarantine),
        "quarantine_keep_per_name": int(args.quarantine_keep_per_name),
        **quarantine_summary,
        "report_output": str(report_output),
    }
    payload = {
        "summary": summary,
        "results": [result.__dict__ for result in results],
        "valid_names": valid_names,
        "retained_bad_names": retained_bad_names,
    }
    report_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        "revalidate_done",
        f"input={len(bad_names)}",
        f"reinstated={len(valid_names)}",
        f"retained={len(retained_bad_names)}",
        f"restage_ok={restage_ok}",
        f"report={report_output}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

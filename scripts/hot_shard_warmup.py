#!/usr/bin/env python3
"""Hydrate active shard binaries from warm storage into hot-local storage."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards-root",
        default="data/shards_global/fineweb-global-bpe-v1",
        help="Local shards root containing active manifest.json files",
    )
    parser.add_argument(
        "--warm-shards-root",
        default="/mnt/ceph/llm/data/shards_global/fineweb-global-bpe-v1",
        help="Warm shards root mirroring local batch directory names",
    )
    parser.add_argument(
        "--include-offloaded-manifests",
        action="store_true",
        help="Also inspect manifest.offloaded.json files (default: active manifests only)",
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="Comma-separated manifest splits to include (default: train,val)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel copy workers for warm->hot hydration (default: 4)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on hydration file count (0 = no cap)",
    )
    parser.add_argument(
        "--allow-missing-warm",
        action="store_true",
        help="Do not fail the command when warm copies are missing",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview hydration actions without copying files",
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _manifest_relpaths(payload: dict[str, Any], *, splits: set[str]) -> list[Path]:
    out: list[Path] = []
    for split in sorted(splits):
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
                out.append(Path(rel))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for rel in out:
        if rel in seen:
            continue
        seen.add(rel)
        uniq.append(rel)
    return uniq


def _copy_warm_to_hot(*, warm_file: Path, local_file: Path) -> tuple[bool, int, str]:
    if not warm_file.exists():
        return (False, 0, "missing_warm_file")
    try:
        size = int(warm_file.stat().st_size)
    except OSError as exc:
        return (False, 0, f"warm_stat_error:{exc}")

    try:
        local_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_copy = local_file.with_name(local_file.name + ".warmup_tmp")
        if tmp_copy.exists() or tmp_copy.is_symlink():
            tmp_copy.unlink()
        with warm_file.open("rb") as src, tmp_copy.open("wb") as dst:
            while True:
                chunk = src.read(4 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp_copy, local_file)
    except OSError as exc:
        return (False, 0, f"copy_error:{exc}")
    return (True, size, "ok")


def main() -> int:
    args = parse_args()
    shards_root = Path(args.shards_root)
    warm_root = Path(args.warm_shards_root)
    if not shards_root.exists():
        raise FileNotFoundError(f"shards-root not found: {shards_root}")
    if not warm_root.exists():
        raise FileNotFoundError(f"warm-shards-root not found: {warm_root}")

    requested_splits = {part.strip() for part in str(args.splits).split(",") if part.strip()}
    if not requested_splits:
        raise ValueError("at least one split is required in --splits")

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = (
        Path(args.report_output)
        if args.report_output
        else Path(f"artifacts/reports/hot_shard_warmup_{ts}.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    manifests = sorted(p for p in shards_root.rglob("manifest.json") if p.is_file())
    if args.include_offloaded_manifests:
        manifests.extend(
            sorted(p for p in shards_root.rglob("manifest.offloaded.json") if p.is_file())
        )
    manifest_total = len(manifests)

    parse_errors = 0
    total_refs = 0
    hot_ready = 0
    skipped_cap = 0
    missing_warm = 0
    failed = 0
    hydrated = 0
    hydrated_bytes = 0
    inspected: dict[str, dict[str, str]] = {}
    to_hydrate: list[tuple[Path, Path]] = []

    max_files = max(0, int(args.max_files))
    for manifest_path in manifests:
        payload = _load_manifest(manifest_path)
        if payload is None:
            parse_errors += 1
            continue
        batch_name = manifest_path.parent.name
        warm_batch = warm_root / batch_name
        for rel in _manifest_relpaths(payload, splits=requested_splits):
            total_refs += 1
            local_file = manifest_path.parent / rel
            key = str(local_file.absolute())
            if key in inspected:
                continue
            warm_file = warm_batch / rel
            local_ok = local_file.exists() and not local_file.is_symlink()
            if local_ok:
                hot_ready += 1
                inspected[key] = {"status": "hot_ready", "warm": str(warm_file)}
                continue
            if max_files > 0 and len(to_hydrate) >= max_files:
                skipped_cap += 1
                inspected[key] = {"status": "skipped_cap", "warm": str(warm_file)}
                continue
            inspected[key] = {"status": "queued", "warm": str(warm_file)}
            to_hydrate.append((local_file, warm_file))

    workers = max(1, int(args.workers))
    if args.dry_run:
        for local_file, warm_file in to_hydrate:
            key = str(local_file.absolute())
            if warm_file.exists():
                inspected[key]["status"] = "would_hydrate"
            else:
                inspected[key]["status"] = "missing_warm_file"
                missing_warm += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_copy_warm_to_hot, warm_file=warm_file, local_file=local_file): (
                    local_file,
                    warm_file,
                )
                for local_file, warm_file in to_hydrate
            }
            for future in as_completed(futures):
                local_file, _warm_file = futures[future]
                key = str(local_file.absolute())
                try:
                    ok, bytes_copied, detail = future.result()
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    bytes_copied = 0
                    detail = f"copy_exception:{exc}"
                if ok:
                    hydrated += 1
                    hydrated_bytes += int(bytes_copied)
                    inspected[key]["status"] = "hydrated"
                else:
                    if detail.startswith("missing_warm"):
                        missing_warm += 1
                    else:
                        failed += 1
                    inspected[key]["status"] = detail

    summary = {
        "shards_root": str(shards_root),
        "warm_shards_root": str(warm_root),
        "dry_run": bool(args.dry_run),
        "workers": workers,
        "splits": sorted(requested_splits),
        "manifest_count": manifest_total,
        "manifest_parse_errors": parse_errors,
        "shard_refs_total": total_refs,
        "unique_shard_files": len(inspected),
        "hot_ready_files": hot_ready,
        "queued_for_hydration": len(to_hydrate),
        "hydrated_files": hydrated,
        "hydrated_bytes": hydrated_bytes,
        "missing_warm_files": missing_warm,
        "failed_files": failed,
        "skipped_due_to_max_files_cap": skipped_cap,
        "max_files": max_files,
    }
    report_payload = {
        "summary": summary,
        "inspected_files": inspected,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(
        "hot_shard_warmup_done",
        f"manifests={manifest_total}",
        f"unique_files={summary['unique_shard_files']}",
        f"hot_ready={hot_ready}",
        f"queued={len(to_hydrate)}",
        f"hydrated={hydrated}",
        f"missing_warm={missing_warm}",
        f"failed={failed}",
        f"report={report_path}",
    )

    if failed > 0:
        return 1
    if missing_warm > 0 and not args.allow_missing_warm:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

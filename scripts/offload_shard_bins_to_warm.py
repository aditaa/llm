#!/usr/bin/env python3
"""Offload shard .bin files to warm storage by replacing local files with symlinks.

Keeps manifest files local so manifest discovery continues to work under shards root.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class BatchResult:
    batch: str
    status: str
    detail: str
    files_linked: int = 0
    bytes_freed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards-root",
        default="data/shards_global/fineweb-global-bpe-v1",
        help="Local shards root containing batch dirs with manifest.json",
    )
    parser.add_argument(
        "--warm-shards-root",
        default="/mnt/ceph/llm/data/shards_global/fineweb-global-bpe-v1",
        help="Warm shards root mirroring local batch directory names",
    )
    parser.add_argument(
        "--keep-local-batches",
        type=int,
        default=24,
        help="Keep newest N batch dirs fully local (default: 24)",
    )
    parser.add_argument(
        "--target-free-gib",
        type=int,
        default=0,
        help="Stop offloading when local free space reaches this GiB (0 = no target)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Max candidate batches to process this run (0 = all)",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help=(
            "Report JSON path. Default: "
            "artifacts/reports/shard_bin_offload_<ts>.json"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without changing files",
    )
    return parser.parse_args()


def _free_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return int(stats.f_bsize * stats.f_bavail)


def _iter_manifest_dirs(shards_root: Path) -> list[Path]:
    manifests = sorted(p for p in shards_root.rglob("manifest.json") if p.is_file())
    return [p.parent for p in manifests]


def _manifest_shard_relpaths(manifest_path: Path) -> list[Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[Path] = []
    for split in ("train", "val"):
        split_meta = payload.get(split, {})
        shards = split_meta.get("shards", []) if isinstance(split_meta, dict) else []
        if not isinstance(shards, list):
            continue
        for row in shards:
            if not isinstance(row, dict):
                continue
            rel = row.get("path")
            if isinstance(rel, str) and rel.strip():
                out.append(Path(rel))
    # Preserve order but drop duplicates.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for rel in out:
        if rel in seen:
            continue
        seen.add(rel)
        uniq.append(rel)
    return uniq


def _batch_sort_key(batch_dir: Path) -> float:
    manifest = batch_dir / "manifest.json"
    try:
        return manifest.stat().st_mtime
    except OSError:
        return 0.0


def _validate_batch_paths(
    local_batch: Path,
    warm_batch: Path,
    shard_relpaths: Iterable[Path],
) -> tuple[bool, str]:
    if not warm_batch.exists():
        return False, f"missing_warm_batch:{warm_batch}"
    for rel in shard_relpaths:
        local_file = local_batch / rel
        warm_file = warm_batch / rel
        if local_file.is_symlink():
            if not warm_file.exists():
                return False, f"missing_warm_for_symlink:{warm_file}"
            continue
        if not local_file.exists():
            return False, f"missing_local_file:{local_file}"
        if not warm_file.exists():
            return False, f"missing_warm_file:{warm_file}"
        try:
            local_size = local_file.stat().st_size
            warm_size = warm_file.stat().st_size
        except OSError as exc:
            return False, f"stat_error:{exc}"
        if local_size != warm_size:
            return False, f"size_mismatch:{rel}:{local_size}!={warm_size}"
    return True, "ok"


def _replace_with_symlink(local_file: Path, warm_file: Path, dry_run: bool) -> int:
    if local_file.is_symlink():
        try:
            target = local_file.resolve(strict=False)
        except OSError:
            target = Path("")
        if target == warm_file:
            return 0
    try:
        local_size = local_file.stat().st_size if local_file.exists() and not local_file.is_symlink() else 0
    except OSError:
        local_size = 0
    if dry_run:
        return local_size
    local_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = local_file.with_name(local_file.name + ".offload_link_tmp")
    try:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        os.symlink(str(warm_file), str(tmp_link))
        os.replace(tmp_link, local_file)
    finally:
        if tmp_link.exists() or tmp_link.is_symlink():
            try:
                tmp_link.unlink()
            except OSError:
                pass
    return local_size


def main() -> int:
    args = parse_args()
    shards_root = Path(args.shards_root)
    warm_root = Path(args.warm_shards_root)
    if not shards_root.exists():
        raise FileNotFoundError(f"shards-root not found: {shards_root}")
    if not warm_root.exists():
        raise FileNotFoundError(f"warm-shards-root not found: {warm_root}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = (
        Path(args.report_output)
        if args.report_output
        else Path(f"artifacts/reports/shard_bin_offload_{ts}.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    free_before = _free_bytes(shards_root)
    target_free_bytes = int(args.target_free_gib) * 1024 * 1024 * 1024

    batch_dirs = _iter_manifest_dirs(shards_root)
    batch_dirs = sorted(batch_dirs, key=_batch_sort_key, reverse=True)
    keep_n = max(0, int(args.keep_local_batches))
    keep_set = {p.name for p in batch_dirs[:keep_n]}
    candidates = [p for p in batch_dirs if p.name not in keep_set]
    if args.max_batches > 0:
        candidates = candidates[: int(args.max_batches)]

    results: list[BatchResult] = []
    total_linked = 0
    total_freed = 0

    for batch_dir in candidates:
        if target_free_bytes > 0 and _free_bytes(shards_root) >= target_free_bytes:
            results.append(
                BatchResult(
                    batch=batch_dir.name,
                    status="stopped_target_reached",
                    detail=f"target_free_gib={args.target_free_gib}",
                )
            )
            break

        manifest = batch_dir / "manifest.json"
        try:
            relpaths = _manifest_shard_relpaths(manifest)
        except Exception as exc:  # noqa: BLE001
            results.append(BatchResult(batch=batch_dir.name, status="skip_manifest_error", detail=str(exc)))
            continue
        if not relpaths:
            results.append(BatchResult(batch=batch_dir.name, status="skip_no_shards", detail="manifest had no shard paths"))
            continue

        warm_batch = warm_root / batch_dir.name
        ok, detail = _validate_batch_paths(batch_dir, warm_batch, relpaths)
        if not ok:
            results.append(BatchResult(batch=batch_dir.name, status="skip_validation_failed", detail=detail))
            continue

        batch_linked = 0
        batch_freed = 0
        for rel in relpaths:
            local_file = batch_dir / rel
            warm_file = warm_batch / rel
            freed = _replace_with_symlink(local_file, warm_file, args.dry_run)
            if freed >= 0:
                batch_linked += 1
                batch_freed += int(freed)

        total_linked += batch_linked
        total_freed += batch_freed
        results.append(
            BatchResult(
                batch=batch_dir.name,
                status="offloaded" if not args.dry_run else "would_offload",
                detail="ok",
                files_linked=batch_linked,
                bytes_freed=batch_freed,
            )
        )

    free_after = _free_bytes(shards_root)
    summary = {
        "shards_root": str(shards_root),
        "warm_shards_root": str(warm_root),
        "dry_run": bool(args.dry_run),
        "keep_local_batches": keep_n,
        "candidate_batches": len(candidates),
        "files_linked": total_linked,
        "bytes_freed_estimate": total_freed,
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "target_free_gib": int(args.target_free_gib),
        "report_output": str(report_path),
    }
    payload = {
        "summary": summary,
        "results": [r.__dict__ for r in results],
        "kept_batches": sorted(keep_set),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "shard_offload_done",
        f"candidates={len(candidates)}",
        f"linked={total_linked}",
        f"bytes_freed={total_freed}",
        f"free_before={free_before}",
        f"free_after={free_after}",
        f"report={report_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

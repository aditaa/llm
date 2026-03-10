#!/usr/bin/env python3
"""Reconcile offloaded manifests against training registry and coverage targets.

This script can restore `manifest.offloaded.json` back to `manifest.json` when:
- the batch is not present in the trained-batches registry, and/or
- active manifest unique-input coverage is below a configured threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards-root",
        default="data/shards_global/fineweb-global-bpe-v1",
        help="Root containing shard batch directories",
    )
    parser.add_argument(
        "--trained-batches-file",
        default="artifacts/reports/train_supervisor_phase1_talk/trained_batch_names.txt,artifacts/reports/train_supervisor_350bt/trained_batch_names.txt",
        help=(
            "Path or comma-separated fallback list of trained-batch registry files "
            "(one batch name per line)"
        ),
    )
    parser.add_argument(
        "--skip-if-trained-file-missing",
        action="store_true",
        help="Proceed without trained-registry gating if no trained-batches file exists",
    )
    parser.add_argument(
        "--manifest-disabled-suffix",
        default=".offloaded.json",
        help="Suffix used for offloaded manifests (default: .offloaded.json)",
    )
    parser.add_argument(
        "--min-active-unique-input-files",
        type=int,
        default=0,
        help=(
            "Restore offloaded manifests until active manifest unique input files "
            "reach this threshold (default: 0 disables)"
        ),
    )
    parser.add_argument(
        "--max-restore",
        type=int,
        default=0,
        help="Maximum manifests to restore this run (0 = unlimited)",
    )
    parser.add_argument(
        "--warm-shards-root",
        default="/mnt/ceph/llm/data/shards_global/fineweb-global-bpe-v1",
        help="Warm shards root mirroring local batch directories",
    )
    parser.add_argument(
        "--rehydrate-restored-bins",
        action="store_true",
        help="Copy shard bins back to local hot storage for manifests restored this run",
    )
    parser.add_argument(
        "--rehydrate-active-symlink-bins",
        action="store_true",
        help="Copy shard bins back to local hot storage for all active manifests with symlink bins",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help="Optional JSON report output path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    return parser.parse_args()


def _resolve_fallback_path(raw: str) -> Path | None:
    value = raw.strip()
    if not value:
        return None
    if "," not in value:
        return Path(value)
    candidates = [Path(part.strip()) for part in value.split(",") if part.strip()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _manifest_input_files(payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    raw_files = payload.get("input_files", [])
    if not isinstance(raw_files, list):
        return out
    for raw in raw_files:
        name = Path(str(raw)).name
        if name:
            out.add(name)
    return out


def _manifest_shard_relpaths(payload: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
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
                out.append(Path(rel))
    # preserve order / dedupe
    seen: set[Path] = set()
    uniq: list[Path] = []
    for rel in out:
        if rel in seen:
            continue
        seen.add(rel)
        uniq.append(rel)
    return uniq


def _active_manifest_paths(shards_root: Path) -> list[Path]:
    return sorted(p for p in shards_root.rglob("manifest.json") if p.is_file())


def _offloaded_manifest_paths(shards_root: Path, suffix: str) -> list[Path]:
    filename = f"manifest{suffix}"
    return sorted(
        (p for p in shards_root.rglob(filename) if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _rehydrate_manifest_bins(
    *,
    manifest_path: Path,
    payload: dict[str, Any],
    warm_root: Path,
    dry_run: bool,
) -> tuple[int, int, int]:
    batch_name = manifest_path.parent.name
    warm_batch = warm_root / batch_name
    if not warm_batch.exists():
        return (0, 0, 0)
    copied = 0
    bytes_copied = 0
    missing_warm = 0
    for rel in _manifest_shard_relpaths(payload):
        local_file = manifest_path.parent / rel
        warm_file = warm_batch / rel
        local_is_symlink = local_file.is_symlink()
        needs_copy = local_is_symlink or not local_file.exists()
        if not needs_copy:
            continue
        if not warm_file.exists():
            missing_warm += 1
            continue
        try:
            warm_size = warm_file.stat().st_size
        except OSError:
            missing_warm += 1
            continue
        copied += 1
        bytes_copied += int(warm_size)
        if dry_run:
            continue
        local_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_copy = local_file.with_name(local_file.name + ".rehydrate_tmp")
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
    return (copied, bytes_copied, missing_warm)


def _load_trained_batches(path: Path) -> set[str]:
    values: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            values.add(value)
    return values


def main() -> int:
    args = parse_args()
    shards_root = Path(args.shards_root)
    warm_root = Path(args.warm_shards_root)
    if not shards_root.exists():
        raise FileNotFoundError(f"shards-root not found: {shards_root}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = (
        Path(args.report_output)
        if args.report_output
        else Path(f"artifacts/reports/offload_reconcile_{ts}.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    trained_file = _resolve_fallback_path(args.trained_batches_file)
    trained_batches: set[str] | None = None
    trained_registry_present = False
    if args.trained_batches_file:
        if trained_file is None:
            if not args.skip_if_trained_file_missing:
                raise FileNotFoundError("no valid trained-batches file candidates provided")
        elif trained_file.exists():
            trained_batches = _load_trained_batches(trained_file)
            trained_registry_present = True
        elif not args.skip_if_trained_file_missing:
            raise FileNotFoundError(f"trained batches file not found: {trained_file}")

    active_paths = _active_manifest_paths(shards_root)
    active_payloads: dict[Path, dict[str, Any]] = {}
    active_unique_inputs: set[str] = set()
    active_parse_errors = 0
    for path in active_paths:
        payload = _load_manifest(path)
        if payload is None:
            active_parse_errors += 1
            continue
        active_payloads[path] = payload
        active_unique_inputs.update(_manifest_input_files(payload))

    offloaded_paths = _offloaded_manifest_paths(shards_root, args.manifest_disabled_suffix)
    restored = 0
    restored_untrained = 0
    restored_coverage = 0
    skipped_active_exists = 0
    skipped_parse_error = 0
    skipped_trained = 0
    rehydrated_files = 0
    rehydrated_bytes = 0
    rehydrate_missing_warm_files = 0
    changes: list[dict[str, Any]] = []

    target_active_unique = max(0, int(args.min_active_unique_input_files))
    max_restore = max(0, int(args.max_restore))

    for offloaded_path in offloaded_paths:
        if max_restore > 0 and restored >= max_restore:
            break
        active_path = offloaded_path.with_name("manifest.json")
        if active_path.exists():
            skipped_active_exists += 1
            changes.append(
                {
                    "batch": offloaded_path.parent.name,
                    "action": "skip_active_exists",
                    "offloaded_manifest": str(offloaded_path),
                    "active_manifest": str(active_path),
                }
            )
            continue

        payload = _load_manifest(offloaded_path)
        if payload is None:
            skipped_parse_error += 1
            changes.append(
                {
                    "batch": offloaded_path.parent.name,
                    "action": "skip_parse_error",
                    "offloaded_manifest": str(offloaded_path),
                }
            )
            continue

        batch_name = offloaded_path.parent.name
        inputs = _manifest_input_files(payload)

        restore_reason = ""
        if trained_batches is not None and batch_name not in trained_batches:
            restore_reason = "untrained_batch"
        elif target_active_unique > 0 and len(active_unique_inputs) < target_active_unique:
            projected = active_unique_inputs | inputs
            if len(projected) > len(active_unique_inputs):
                restore_reason = "active_coverage_floor"
        else:
            skipped_trained += 1
            changes.append(
                {
                    "batch": batch_name,
                    "action": "skip_trained_or_not_needed",
                    "offloaded_manifest": str(offloaded_path),
                }
            )
            continue

        changes.append(
            {
                "batch": batch_name,
                "action": "restore_manifest",
                "reason": restore_reason,
                "offloaded_manifest": str(offloaded_path),
                "active_manifest": str(active_path),
            }
        )
        if not args.dry_run:
            offloaded_path.rename(active_path)

        restored += 1
        if restore_reason == "untrained_batch":
            restored_untrained += 1
        elif restore_reason == "active_coverage_floor":
            restored_coverage += 1
        active_unique_inputs.update(inputs)

        if args.rehydrate_restored_bins:
            copied, copied_bytes, missing_warm = _rehydrate_manifest_bins(
                manifest_path=active_path,
                payload=payload,
                warm_root=warm_root,
                dry_run=args.dry_run,
            )
            rehydrated_files += copied
            rehydrated_bytes += copied_bytes
            rehydrate_missing_warm_files += missing_warm
            if copied > 0 or missing_warm > 0:
                changes.append(
                    {
                        "batch": batch_name,
                        "action": "rehydrate_restored_bins",
                        "copied_files": copied,
                        "copied_bytes": copied_bytes,
                        "missing_warm_files": missing_warm,
                    }
                )

    if args.rehydrate_active_symlink_bins:
        for manifest_path in _active_manifest_paths(shards_root):
            payload = _load_manifest(manifest_path)
            if payload is None:
                continue
            copied, copied_bytes, missing_warm = _rehydrate_manifest_bins(
                manifest_path=manifest_path,
                payload=payload,
                warm_root=warm_root,
                dry_run=args.dry_run,
            )
            rehydrated_files += copied
            rehydrated_bytes += copied_bytes
            rehydrate_missing_warm_files += missing_warm
            if copied > 0 or missing_warm > 0:
                changes.append(
                    {
                        "batch": manifest_path.parent.name,
                        "action": "rehydrate_active_symlink_bins",
                        "copied_files": copied,
                        "copied_bytes": copied_bytes,
                        "missing_warm_files": missing_warm,
                    }
                )

    active_after = len(_active_manifest_paths(shards_root))
    offloaded_after = len(_offloaded_manifest_paths(shards_root, args.manifest_disabled_suffix))

    summary = {
        "shards_root": str(shards_root),
        "dry_run": bool(args.dry_run),
        "trained_batches_file_arg": args.trained_batches_file,
        "trained_batches_file_resolved": str(trained_file) if trained_file is not None else "",
        "trained_registry_present": trained_registry_present,
        "trained_batch_count": len(trained_batches or set()),
        "warm_shards_root": str(warm_root),
        "active_manifest_count_before": len(active_paths),
        "offloaded_manifest_count_before": len(offloaded_paths),
        "active_manifest_parse_errors": active_parse_errors,
        "active_unique_input_files_after": len(active_unique_inputs),
        "min_active_unique_input_files": target_active_unique,
        "restored_manifest_count": restored,
        "restored_untrained_count": restored_untrained,
        "restored_coverage_floor_count": restored_coverage,
        "skipped_active_exists_count": skipped_active_exists,
        "skipped_parse_error_count": skipped_parse_error,
        "skipped_trained_or_not_needed_count": skipped_trained,
        "rehydrate_restored_bins": bool(args.rehydrate_restored_bins),
        "rehydrate_active_symlink_bins": bool(args.rehydrate_active_symlink_bins),
        "rehydrated_files": rehydrated_files,
        "rehydrated_bytes": rehydrated_bytes,
        "rehydrate_missing_warm_files": rehydrate_missing_warm_files,
        "active_manifest_count_after": active_after,
        "offloaded_manifest_count_after": offloaded_after,
        "max_restore": max_restore,
        "report_output": str(report_path),
    }

    payload = {"summary": summary, "changes": changes}
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "offload_reconcile_done",
        f"restored={restored}",
        f"restored_untrained={restored_untrained}",
        f"restored_coverage_floor={restored_coverage}",
        f"active_after={active_after}",
        f"offloaded_after={offloaded_after}",
        f"active_unique_after={len(active_unique_inputs)}",
        f"rehydrated_files={rehydrated_files}",
        f"rehydrated_bytes={rehydrated_bytes}",
        f"report={report_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

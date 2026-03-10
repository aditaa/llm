#!/usr/bin/env python3
"""Disable active manifests that reference symlinked shard binaries.

This keeps training strictly on hot-local shard files while allowing offloaded
batches to remain tracked via manifest.offloaded.json.
"""

from __future__ import annotations

import argparse
import json
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
        "--manifest-disabled-suffix",
        default=".offloaded.json",
        help="Suffix used when disabling active manifests (default: .offloaded.json)",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help="Optional JSON report path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    return parser.parse_args()


def _load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _manifest_shard_paths(manifest_path: Path, payload: dict[str, Any]) -> list[Path]:
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
                out.append(manifest_path.parent / rel)
    return out


def main() -> int:
    args = parse_args()
    shards_root = Path(args.shards_root)
    if not shards_root.exists():
        raise FileNotFoundError(f"shards-root not found: {shards_root}")

    active_manifests = sorted(
        p for p in shards_root.rglob("manifest.json") if p.is_file()
    )
    active_total = len(active_manifests)
    parse_errors = 0
    active_symlink_manifests = 0
    disabled = 0
    removed_active_existing_offloaded = 0
    inspected = 0
    changes: list[dict[str, str]] = []

    for manifest_path in active_manifests:
        payload = _load_manifest(manifest_path)
        if payload is None:
            parse_errors += 1
            continue
        inspected += 1
        shard_paths = _manifest_shard_paths(manifest_path, payload)
        if not shard_paths:
            continue
        has_symlink = any(path.is_symlink() for path in shard_paths)
        if not has_symlink:
            continue

        active_symlink_manifests += 1
        disabled_path = manifest_path.with_name(
            f"manifest{args.manifest_disabled_suffix}"
        )
        if disabled_path.exists():
            removed_active_existing_offloaded += 1
            changes.append(
                {
                    "manifest": str(manifest_path),
                    "action": "remove_active_existing_offloaded",
                    "target": str(disabled_path),
                }
            )
            if not args.dry_run:
                manifest_path.unlink(missing_ok=True)
            continue

        disabled += 1
        changes.append(
            {
                "manifest": str(manifest_path),
                "action": "disable_active_manifest",
                "target": str(disabled_path),
            }
        )
        if not args.dry_run:
            manifest_path.rename(disabled_path)

    active_after = len(list(shards_root.rglob("manifest.json")))
    offloaded_after = len(list(shards_root.rglob("manifest.offloaded.json")))

    summary = {
        "shards_root": str(shards_root),
        "dry_run": bool(args.dry_run),
        "active_manifest_total_before": active_total,
        "active_manifest_inspected": inspected,
        "active_manifest_parse_errors": parse_errors,
        "active_manifest_with_symlink_bins": active_symlink_manifests,
        "disabled_active_manifests": disabled,
        "removed_active_existing_offloaded": removed_active_existing_offloaded,
        "active_manifest_total_after": active_after,
        "offloaded_manifest_total_after": offloaded_after,
        "changes": changes,
    }

    if args.report_output:
        report_path = Path(args.report_output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"artifacts/reports/hot_manifest_guard_{ts}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "hot_manifest_guard_done",
        f"active_before={active_total}",
        f"active_symlink={active_symlink_manifests}",
        f"disabled={disabled}",
        f"removed_active_existing_offloaded={removed_active_existing_offloaded}",
        f"active_after={active_after}",
        f"offloaded_after={offloaded_after}",
        f"report={report_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Disable overlapping FineWeb shard manifests so each parquet basename is used once."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestEntry:
    manifest_path: Path
    dataset_dir: Path
    input_files: tuple[str, ...]
    mtime: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards-root",
        default="data/shards_global/fineweb-global-bpe-v1",
        help="Root directory containing shard batch subdirectories with manifest.json",
    )
    parser.add_argument(
        "--report-output",
        default="artifacts/reports/fineweb_stage_shard_loop/manifest_dedupe_report.json",
        help="JSON report output path",
    )
    parser.add_argument(
        "--keep",
        choices=("newest", "oldest"),
        default="newest",
        help="When overlaps exist, keep newer or older manifests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze overlaps without disabling duplicate manifests",
    )
    parser.add_argument(
        "--disabled-suffix",
        default=".duplicate.disabled.json",
        help="Suffix used when disabling duplicate manifest files",
    )
    return parser.parse_args()


def _load_manifest_entry(manifest_path: Path) -> ManifestEntry | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    input_files_raw = payload.get("input_files", [])
    if not isinstance(input_files_raw, list):
        return None
    basenames = sorted({Path(str(raw)).name for raw in input_files_raw if str(raw).strip()})
    if not basenames:
        return None
    mtime = manifest_path.stat().st_mtime
    return ManifestEntry(
        manifest_path=manifest_path,
        dataset_dir=manifest_path.parent,
        input_files=tuple(basenames),
        mtime=mtime,
    )


def _iter_entries(shards_root: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for manifest_path in sorted(shards_root.rglob("manifest.json")):
        if not manifest_path.is_file():
            continue
        entry = _load_manifest_entry(manifest_path)
        if entry is not None:
            entries.append(entry)
    return entries


def _disable_manifest(manifest_path: Path, suffix: str) -> Path:
    disabled_path = manifest_path.with_name(f"manifest{suffix}")
    if disabled_path.exists():
        # Ensure idempotent behavior if dedupe was already applied.
        disabled_path.unlink()
    manifest_path.rename(disabled_path)
    return disabled_path


def main() -> int:
    args = parse_args()
    shards_root = Path(args.shards_root)
    report_output = Path(args.report_output)
    if not shards_root.exists():
        raise SystemExit(f"shards-root not found: {shards_root}")

    entries = _iter_entries(shards_root)
    ordered = sorted(entries, key=lambda item: item.mtime, reverse=(args.keep == "newest"))

    claimed_files: dict[str, Path] = {}
    kept: list[ManifestEntry] = []
    duplicates: list[dict[str, Any]] = []

    for entry in ordered:
        overlap = sorted(name for name in entry.input_files if name in claimed_files)
        if overlap:
            duplicates.append(
                {
                    "manifest": str(entry.manifest_path),
                    "dataset_dir": str(entry.dataset_dir),
                    "overlap_files": overlap,
                    "kept_by": sorted(str(claimed_files[name]) for name in overlap),
                }
            )
            continue
        kept.append(entry)
        for name in entry.input_files:
            claimed_files[name] = entry.manifest_path

    disabled: list[dict[str, str]] = []
    if not args.dry_run:
        for item in duplicates:
            manifest_path = Path(str(item["manifest"]))
            if not manifest_path.exists():
                continue
            disabled_path = _disable_manifest(manifest_path, args.disabled_suffix)
            disabled.append({"manifest": str(manifest_path), "disabled_to": str(disabled_path)})

    unique_files = sorted(claimed_files.keys())
    report = {
        "shards_root": str(shards_root),
        "keep_strategy": args.keep,
        "dry_run": bool(args.dry_run),
        "manifest_total": len(entries),
        "manifest_kept": len(kept),
        "manifest_overlap": len(duplicates),
        "unique_input_files": len(unique_files),
        "kept_manifests": [str(item.manifest_path) for item in kept],
        "duplicates": duplicates,
        "disabled": disabled,
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"manifest_total={report['manifest_total']}")
    print(f"manifest_kept={report['manifest_kept']}")
    print(f"manifest_overlap={report['manifest_overlap']}")
    print(f"unique_input_files={report['unique_input_files']}")
    print(f"dry_run={int(args.dry_run)}")
    print(f"report={report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

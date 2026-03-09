#!/usr/bin/env python3
"""Disable exact-duplicate FineWeb shard manifests and report partial overlaps."""

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
    groups: dict[tuple[str, ...], list[ManifestEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.input_files, []).append(entry)

    kept: list[ManifestEntry] = []
    duplicates: list[dict[str, Any]] = []
    for input_set, group_entries in groups.items():
        ordered_group = sorted(
            group_entries,
            key=lambda item: item.mtime,
            reverse=(args.keep == "newest"),
        )
        keeper = ordered_group[0]
        kept.append(keeper)
        for duplicate in ordered_group[1:]:
            duplicates.append(
                {
                    "manifest": str(duplicate.manifest_path),
                    "dataset_dir": str(duplicate.dataset_dir),
                    "duplicate_of": str(keeper.manifest_path),
                    "input_files": list(input_set),
                }
            )

    claimed_files: dict[str, list[Path]] = {}
    for entry in kept:
        for name in entry.input_files:
            claimed_files.setdefault(name, []).append(entry.manifest_path)

    partial_overlaps: list[dict[str, Any]] = []
    for entry in kept:
        overlap_files = sorted(name for name in entry.input_files if len(claimed_files[name]) > 1)
        if not overlap_files:
            continue
        overlap_with = sorted(
            {
                str(other)
                for name in overlap_files
                for other in claimed_files[name]
                if other != entry.manifest_path
            }
        )
        partial_overlaps.append(
            {
                "manifest": str(entry.manifest_path),
                "dataset_dir": str(entry.dataset_dir),
                "overlap_files": overlap_files,
                "overlap_with": overlap_with,
            }
        )

    disabled: list[dict[str, str]] = []
    if not args.dry_run:
        for item in duplicates:
            manifest_path = Path(str(item["manifest"]))
            if not manifest_path.exists():
                continue
            disabled_path = _disable_manifest(manifest_path, args.disabled_suffix)
            disabled.append({"manifest": str(manifest_path), "disabled_to": str(disabled_path)})

    unique_files = sorted(claimed_files.keys())
    partial_overlap_input_files = sum(1 for refs in claimed_files.values() if len(refs) > 1)
    report = {
        "shards_root": str(shards_root),
        "keep_strategy": args.keep,
        "dry_run": bool(args.dry_run),
        "manifest_total": len(entries),
        "manifest_kept": len(kept),
        "manifest_overlap": len(duplicates),
        "manifest_exact_duplicates": len(duplicates),
        "partial_overlap_manifests": len(partial_overlaps),
        "partial_overlap_input_files": partial_overlap_input_files,
        "unique_input_files": len(unique_files),
        "kept_manifests": [str(item.manifest_path) for item in kept],
        "duplicates": duplicates,
        "partial_overlaps": partial_overlaps,
        "disabled": disabled,
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"manifest_total={report['manifest_total']}")
    print(f"manifest_kept={report['manifest_kept']}")
    print(f"manifest_overlap={report['manifest_overlap']}")
    print(f"partial_overlap_manifests={report['partial_overlap_manifests']}")
    print(f"partial_overlap_input_files={report['partial_overlap_input_files']}")
    print(f"unique_input_files={report['unique_input_files']}")
    print(f"dry_run={int(args.dry_run)}")
    print(f"report={report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

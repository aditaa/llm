"""FineWeb staging/sharding guardrail validation helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_expected_files(files_list: Path) -> list[str]:
    return [
        line.strip()
        for line in files_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _resolve_manifest_path(
    *,
    manifest_field: str,
    report_path: Path,
    output_dir: Path,
) -> Path:
    candidate = Path(manifest_field)
    if candidate.is_absolute():
        return candidate

    candidates = [
        output_dir / candidate,
        report_path.parent / candidate,
        Path.cwd() / candidate,
    ]
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()


def validate_job_artifacts(
    *,
    job_id: str,
    report_path: Path,
    output_dir: Path,
    files_list: Path,
) -> tuple[int, int, int]:
    if not report_path.exists():
        raise ValueError(f"{job_id}: report missing: {report_path}")
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{job_id}: manifest missing: {manifest_path}")

    report = _load_json(report_path)
    manifest_field = report.get("manifest")
    if not manifest_field:
        raise ValueError(f"{job_id}: report missing manifest path")
    manifest_path_from_report = _resolve_manifest_path(
        manifest_field=str(manifest_field),
        report_path=report_path,
        output_dir=output_dir,
    )
    if not manifest_path_from_report.exists():
        raise ValueError(f"{job_id}: manifest not found: {manifest_path_from_report}")

    manifest = _load_json(manifest_path_from_report)

    rows_sharded = int(report.get("rows_sharded", 0))
    if rows_sharded <= 0:
        raise ValueError(f"{job_id}: rows_sharded <= 0")

    train = manifest.get("train", {})
    val = manifest.get("val", {})
    if not isinstance(train, dict) or not isinstance(val, dict):
        raise ValueError(f"{job_id}: manifest split metadata is invalid")

    train_tokens = int(train.get("total_tokens", 0))
    val_tokens = int(val.get("total_tokens", 0))
    if train_tokens + val_tokens <= 0:
        raise ValueError(f"{job_id}: total token count <= 0")

    train_shards = train.get("shards", [])
    val_shards = val.get("shards", [])
    if not isinstance(train_shards, list) or not isinstance(val_shards, list):
        raise ValueError(f"{job_id}: manifest shard metadata is invalid")
    if not train_shards and not val_shards:
        raise ValueError(f"{job_id}: no shard entries in manifest")

    for shard in [*train_shards, *val_shards]:
        if not isinstance(shard, dict):
            raise ValueError(f"{job_id}: shard entry must be an object")
        shard_name = shard.get("path")
        if not shard_name:
            raise ValueError(f"{job_id}: shard entry missing path")
        shard_path = output_dir / str(shard_name)
        if not shard_path.exists():
            raise ValueError(f"{job_id}: missing shard file: {shard_path}")
        if shard_path.stat().st_size <= 0:
            raise ValueError(f"{job_id}: empty shard file: {shard_path}")

    expected_files = _load_expected_files(files_list)
    manifest_inputs = {Path(str(raw)).name for raw in manifest.get("input_files", [])}
    missing = [name for name in expected_files if name not in manifest_inputs]
    if missing:
        raise ValueError(f"{job_id}: manifest missing expected input files: {missing[:5]}")

    total_tokens = train_tokens + val_tokens
    total_shards = len(train_shards) + len(val_shards)
    return rows_sharded, total_tokens, total_shards


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--files-list", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows, tokens, shards = validate_job_artifacts(
        job_id=args.job_id,
        report_path=Path(args.report_json),
        output_dir=Path(args.output_dir),
        files_list=Path(args.files_list),
    )
    print(f"guardrail_ok id={args.job_id} rows={rows} tokens={tokens} shards={shards}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

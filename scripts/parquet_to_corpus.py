#!/usr/bin/env python3
"""Convert local Parquet datasets into newline text corpora for tokenization/sharding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _normalize_line(text: str) -> str:
    return " ".join(text.split())


def _iter_parquet_files(input_dir: Path, pattern: str, max_files: int) -> list[Path]:
    files = sorted(p for p in input_dir.rglob(pattern) if p.is_file())
    if max_files > 0:
        return files[:max_files]
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing parquet files")
    parser.add_argument("--output-dir", required=True, help="Output directory for .txt corpora")
    parser.add_argument("--pattern", default="*.parquet", help="Glob for parquet files")
    parser.add_argument("--field", default="text", help="Column name to extract")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Parquet row batch size",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Drop extracted rows shorter than this",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Truncate rows to this many chars (0 means no truncation)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on parquet file count (0 means all)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rebuild output files that already exist",
    )
    parser.add_argument(
        "--report-output",
        default="artifacts/reports/parquet_to_corpus_report.json",
        help="JSON report path",
    )
    return parser.parse_args()


def _process_one_file(
    parquet_path: Path,
    output_path: Path,
    *,
    field: str,
    batch_size: int,
    min_chars: int,
    max_chars: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    schema_names = set(parquet.schema.names)
    if field not in schema_names:
        raise ValueError(f"missing field '{field}' in {parquet_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_seen = 0
    rows_written = 0
    rows_filtered_short = 0

    with output_path.open("w", encoding="utf-8") as dst:
        for batch in parquet.iter_batches(columns=[field], batch_size=batch_size):
            values = batch.column(0).to_pylist()
            for value in values:
                rows_seen += 1
                if value is None:
                    continue

                text = _normalize_line(str(value))
                if len(text) < min_chars:
                    rows_filtered_short += 1
                    continue

                if max_chars > 0:
                    text = text[:max_chars]
                if not text:
                    continue

                dst.write(text + "\n")
                rows_written += 1

    return {
        "input_path": str(parquet_path),
        "output_path": str(output_path),
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_filtered_short": rows_filtered_short,
        "status": "ok",
    }


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.min_chars < 0:
        raise ValueError("min-chars must be >= 0")
    if args.max_chars < 0:
        raise ValueError("max-chars must be >= 0")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_output)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input-dir not found or not a dir: {input_dir}")

    try:
        import pyarrow.parquet as _pq  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet conversion. Install training extras or run: "
            "python -m pip install pyarrow"
        ) from exc

    parquet_files = _iter_parquet_files(input_dir=input_dir, pattern=args.pattern, max_files=args.max_files)
    if not parquet_files:
        raise ValueError("no parquet files matched")

    started_at = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    totals = {
        "files_total": len(parquet_files),
        "files_ok": 0,
        "files_skipped_existing": 0,
        "files_failed": 0,
        "rows_seen": 0,
        "rows_written": 0,
        "rows_filtered_short": 0,
    }

    for parquet_path in parquet_files:
        rel = parquet_path.relative_to(input_dir)
        output_path = (output_dir / rel).with_suffix(".txt")

        if (not args.no_skip_existing) and output_path.exists():
            totals["files_skipped_existing"] += 1
            results.append(
                {
                    "input_path": str(parquet_path),
                    "output_path": str(output_path),
                    "status": "skipped_existing",
                }
            )
            continue

        try:
            row = _process_one_file(
                parquet_path=parquet_path,
                output_path=output_path,
                field=args.field,
                batch_size=args.batch_size,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
            )
            totals["files_ok"] += 1
            totals["rows_seen"] += int(row["rows_seen"])
            totals["rows_written"] += int(row["rows_written"])
            totals["rows_filtered_short"] += int(row["rows_filtered_short"])
            results.append(row)
        except Exception as exc:  # pragma: no cover
            totals["files_failed"] += 1
            results.append(
                {
                    "input_path": str(parquet_path),
                    "output_path": str(output_path),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    elapsed = time.time() - started_at
    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pattern": args.pattern,
        "field": args.field,
        "batch_size": args.batch_size,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "skip_existing": not args.no_skip_existing,
        "elapsed_seconds": elapsed,
        "totals": totals,
        "files": results,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"input_dir={input_dir}")
    print(f"output_dir={output_dir}")
    print(f"report={report_path}")
    print(f"files_total={totals['files_total']}")
    print(f"files_ok={totals['files_ok']}")
    print(f"files_skipped_existing={totals['files_skipped_existing']}")
    print(f"files_failed={totals['files_failed']}")
    print(f"rows_seen={totals['rows_seen']}")
    print(f"rows_written={totals['rows_written']}")
    print(f"rows_filtered_short={totals['rows_filtered_short']}")
    print(f"elapsed_seconds={elapsed:.2f}")

    return 0 if totals["files_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

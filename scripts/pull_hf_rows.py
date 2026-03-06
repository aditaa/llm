#!/usr/bin/env python3
"""Pull text rows from the Hugging Face datasets-server API into newline text."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout_s: float,
) -> dict:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_rows_with_retries(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> dict:
    attempt = 0
    while True:
        try:
            return _fetch_rows(
                dataset=dataset,
                config=config,
                split=split,
                offset=offset,
                length=length,
                timeout_s=timeout_s,
            )
        except urllib.error.HTTPError as err:
            retriable = err.code in {429, 500, 502, 503, 504}
            if (not retriable) or attempt >= max_retries:
                raise
            wait_s = retry_backoff_s * (2**attempt)
            print(
                f"retryable_http_error={err.code} offset={offset} attempt={attempt + 1} "
                f"sleep_seconds={wait_s:.2f}"
            )
            time.sleep(wait_s)
            attempt += 1


def _normalize_line(text: str) -> str:
    return " ".join(text.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HF dataset id")
    parser.add_argument("--config", required=True, help="HF dataset config")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output", required=True, help="Output text file path")
    parser.add_argument(
        "--field",
        default="text",
        help="Row field to extract as text",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50_000,
        help="Maximum rows to write",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Rows per API page (server caps this around 100)",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Row offset for first request",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Drop rows shorter than this",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Truncate rows to this many chars (0 means no truncation)",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between page requests",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="HTTP timeout per request",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file if it already exists",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries per page for transient HTTP errors",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Base exponential backoff seconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.page_size <= 0:
        raise ValueError("page-size must be > 0")
    if args.max_rows <= 0:
        raise ValueError("max-rows must be > 0")
    if args.start_offset < 0:
        raise ValueError("start-offset must be >= 0")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.append:
        raise FileExistsError(f"output exists: {output_path} (pass --append to continue)")

    mode = "a" if args.append else "w"
    rows_written = 0
    rows_seen = 0
    rows_filtered_short = 0
    offset = args.start_offset
    started_at = time.time()

    with output_path.open(mode, encoding="utf-8") as handle:
        while rows_written < args.max_rows:
            page = _fetch_rows_with_retries(
                dataset=args.dataset,
                config=args.config,
                split=args.split,
                offset=offset,
                length=min(args.page_size, args.max_rows - rows_written),
                timeout_s=args.timeout_seconds,
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff_seconds,
            )
            rows = page.get("rows", [])
            if not rows:
                break

            for item in rows:
                row = item.get("row", {})
                raw_text = str(row.get(args.field, ""))
                text = _normalize_line(raw_text)
                rows_seen += 1

                if len(text) < args.min_chars:
                    rows_filtered_short += 1
                    continue
                if args.max_chars > 0:
                    text = text[: args.max_chars]
                if not text:
                    continue

                handle.write(text + "\n")
                rows_written += 1
                if rows_written >= args.max_rows:
                    break

            offset += len(rows)
            if rows_written % max(1_000, args.page_size * 10) == 0:
                elapsed = max(1e-9, time.time() - started_at)
                print(
                    f"rows_written={rows_written} rows_seen={rows_seen} "
                    f"offset={offset} rows_per_sec={rows_written/elapsed:.2f}"
                )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    elapsed = time.time() - started_at
    meta = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "field": args.field,
        "output": str(output_path),
        "max_rows": args.max_rows,
        "page_size": args.page_size,
        "start_offset": args.start_offset,
        "end_offset": offset,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_filtered_short": rows_filtered_short,
        "elapsed_seconds": elapsed,
    }
    meta_path = Path(str(output_path) + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"output={output_path}")
    print(f"meta={meta_path}")
    print(f"rows_seen={rows_seen}")
    print(f"rows_written={rows_written}")
    print(f"rows_filtered_short={rows_filtered_short}")
    print(f"end_offset={offset}")
    print(f"elapsed_seconds={elapsed:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

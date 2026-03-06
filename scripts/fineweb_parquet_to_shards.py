#!/usr/bin/env python3
"""Build token shards directly from FineWeb parquet files.

Default mode trains a char tokenizer from selected parquet rows, then writes shards.
Incremental mode reuses an existing tokenizer via --tokenizer-in.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from array import array
from pathlib import Path
from typing import Any, Iterator

from llm.tokenizer import BasicCharTokenizer


def _normalize_line(text: str) -> str:
    return " ".join(text.split())


def _array_type_for_vocab(vocab_size: int) -> tuple[str, str]:
    if vocab_size <= 65535:
        return "H", "uint16"
    return "I", "uint32"


class _ShardWriter:
    def __init__(
        self,
        *,
        split: str,
        output_dir: Path,
        array_type: str,
        shard_size_tokens: int,
    ) -> None:
        self.split = split
        self.output_dir = output_dir
        self.array_type = array_type
        self.shard_size_tokens = shard_size_tokens
        self.buffer = array(self.array_type)
        self.shard_index = 0
        self.total_tokens = 0
        self.shards: list[dict[str, int | str]] = []

    def _flush_full(self) -> None:
        shard_tokens = self.buffer[: self.shard_size_tokens]
        del self.buffer[: self.shard_size_tokens]

        path = self.output_dir / f"{self.split}_{self.shard_index:06d}.bin"
        with path.open("wb") as handle:
            shard_tokens.tofile(handle)

        self.shards.append({"path": path.name, "tokens": len(shard_tokens)})
        self.shard_index += 1

    def add(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        self.buffer.extend(token_ids)
        self.total_tokens += len(token_ids)

        while len(self.buffer) >= self.shard_size_tokens:
            self._flush_full()

    def finalize(self) -> None:
        if not self.buffer:
            return

        path = self.output_dir / f"{self.split}_{self.shard_index:06d}.bin"
        with path.open("wb") as handle:
            self.buffer.tofile(handle)

        self.shards.append({"path": path.name, "tokens": len(self.buffer)})
        self.buffer = array(self.array_type)
        self.shard_index += 1


def _iter_parquet_files(input_dir: Path, pattern: str, max_files: int) -> list[Path]:
    files = sorted(p for p in input_dir.rglob(pattern) if p.is_file())
    if max_files > 0:
        return files[:max_files]
    return files


def _read_files_list(files_list_path: Path, input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for raw_line in files_list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = (input_dir / path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"listed parquet file not found: {path}")
        files.append(path)
    if not files:
        raise ValueError(f"files-list has no usable parquet files: {files_list_path}")
    return files


def _iter_text_rows(
    *,
    parquet_files: list[Path],
    field: str,
    batch_size: int,
    min_chars: int,
    max_chars: int,
    max_rows_per_file: int,
) -> Iterator[str]:
    import pyarrow.parquet as pq

    for parquet_path in parquet_files:
        table = pq.ParquetFile(parquet_path)
        if field not in set(table.schema.names):
            raise ValueError(f"missing field '{field}' in {parquet_path}")

        emitted = 0
        for batch in table.iter_batches(columns=[field], batch_size=batch_size):
            values = batch.column(0).to_pylist()
            for value in values:
                if max_rows_per_file > 0 and emitted >= max_rows_per_file:
                    break
                if value is None:
                    continue

                text = _normalize_line(str(value))
                if len(text) < min_chars:
                    continue
                if max_chars > 0:
                    text = text[:max_chars]
                if not text:
                    continue

                emitted += 1
                yield text

            if max_rows_per_file > 0 and emitted >= max_rows_per_file:
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing parquet files")
    parser.add_argument("--output-dir", required=True, help="Output directory for token shards")
    parser.add_argument(
        "--tokenizer-out",
        default=None,
        help="Output tokenizer JSON path (required when --tokenizer-in is unset)",
    )
    parser.add_argument(
        "--tokenizer-in",
        default=None,
        help="Existing tokenizer JSON path (reuse for incremental shard builds)",
    )
    parser.add_argument("--pattern", default="*.parquet", help="Parquet glob pattern")
    parser.add_argument(
        "--files-list",
        default=None,
        help="Optional newline-separated parquet list (absolute or relative to --input-dir)",
    )
    parser.add_argument("--field", default="text", help="Parquet text column")
    parser.add_argument("--batch-size", type=int, default=8192, help="Parquet read batch size")
    parser.add_argument("--shard-size-tokens", type=int, default=5_000_000, help="Tokens per shard")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-chars", type=int, default=80, help="Drop short rows")
    parser.add_argument("--max-chars", type=int, default=0, help="Truncate rows to this length")
    parser.add_argument("--max-files", type=int, default=0, help="Max parquet files to process")
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="Optional max rows per parquet file (0 = all)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output manifest")
    parser.add_argument(
        "--report-output",
        default="artifacts/reports/fineweb_parquet_to_shards_report.json",
        help="Output report JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.shard_size_tokens <= 0:
        raise ValueError("shard-size-tokens must be > 0")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("val-ratio must be in [0, 1)")
    if args.min_chars < 0:
        raise ValueError("min-chars must be >= 0")
    if args.max_chars < 0:
        raise ValueError("max-chars must be >= 0")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tokenizer_out = Path(args.tokenizer_out) if args.tokenizer_out else None
    tokenizer_in = Path(args.tokenizer_in) if args.tokenizer_in else None
    report_output = Path(args.report_output)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"input-dir not found or not a directory: {input_dir}")

    try:
        import pyarrow.parquet as _pq  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required. Install train deps or run: python -m pip install pyarrow"
        ) from exc

    if args.files_list:
        parquet_files = _read_files_list(Path(args.files_list), input_dir)
    else:
        parquet_files = _iter_parquet_files(
            input_dir=input_dir,
            pattern=args.pattern,
            max_files=args.max_files,
        )
    if not parquet_files:
        raise ValueError("no parquet files matched")

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"manifest exists: {manifest_path} (use --force to overwrite)")

    started_at = time.time()

    if tokenizer_in is not None:
        if not tokenizer_in.exists():
            raise FileNotFoundError(f"tokenizer-in not found: {tokenizer_in}")
        tokenizer = BasicCharTokenizer.load(tokenizer_in)
        tokenizer_path = tokenizer_in
        pass1_rows = 0
        print(
            f"pass=1 action=load_tokenizer tokenizer={tokenizer_path} "
            f"vocab_size={tokenizer.vocab_size}"
        )
    else:
        if tokenizer_out is None:
            raise ValueError("tokenizer-out is required when tokenizer-in is not set")
        print(f"pass=1 action=scan_chars parquet_files={len(parquet_files)}")
        chars: set[str] = set()
        pass1_rows = 0
        for text in _iter_text_rows(
            parquet_files=parquet_files,
            field=args.field,
            batch_size=args.batch_size,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            max_rows_per_file=args.max_rows_per_file,
        ):
            chars.update(text)
            chars.add("\n")
            pass1_rows += 1
            if pass1_rows % 500_000 == 0:
                print(f"pass=1 rows={pass1_rows} unique_chars={len(chars)}")

        tokenizer = BasicCharTokenizer.train("".join(sorted(chars)))
        tokenizer_out.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tokenizer_out)
        tokenizer_path = tokenizer_out

    array_type, token_dtype = _array_type_for_vocab(tokenizer.vocab_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_writer = _ShardWriter(
        split="train",
        output_dir=output_dir,
        array_type=array_type,
        shard_size_tokens=args.shard_size_tokens,
    )
    val_writer = _ShardWriter(
        split="val",
        output_dir=output_dir,
        array_type=array_type,
        shard_size_tokens=args.shard_size_tokens,
    )

    rng = random.Random(args.seed)
    pass2_rows = 0
    print(f"pass=2 action=tokenize_and_shard token_dtype={token_dtype}")
    for text in _iter_text_rows(
        parquet_files=parquet_files,
        field=args.field,
        batch_size=args.batch_size,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_rows_per_file=args.max_rows_per_file,
    ):
        token_ids = tokenizer.encode(text + "\n")
        if rng.random() < args.val_ratio:
            val_writer.add(token_ids)
        else:
            train_writer.add(token_ids)

        pass2_rows += 1
        if pass2_rows % 500_000 == 0:
            print(
                f"pass=2 rows={pass2_rows} train_tokens={train_writer.total_tokens} "
                f"val_tokens={val_writer.total_tokens}"
            )

    train_writer.finalize()
    val_writer.finalize()

    elapsed = time.time() - started_at

    manifest: dict[str, Any] = {
        "input_path": str(input_dir),
        "input_pattern": args.pattern,
        "input_files": [str(path) for path in parquet_files],
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "token_dtype": token_dtype,
        "shard_size_tokens": args.shard_size_tokens,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "line_count": pass2_rows,
        "train": {
            "total_tokens": train_writer.total_tokens,
            "shards": train_writer.shards,
        },
        "val": {
            "total_tokens": val_writer.total_tokens,
            "shards": val_writer.shards,
        },
        "source": {
            "dataset": "HuggingFaceFW/fineweb",
            "field": args.field,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "max_rows_per_file": args.max_rows_per_file,
        },
        "elapsed_seconds": elapsed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "manifest": str(manifest_path),
        "tokenizer": str(tokenizer_path),
        "parquet_files": len(parquet_files),
        "rows_seen": pass1_rows,
        "rows_sharded": pass2_rows,
        "vocab_size": tokenizer.vocab_size,
        "token_dtype": token_dtype,
        "train_tokens": train_writer.total_tokens,
        "val_tokens": val_writer.total_tokens,
        "train_shards": len(train_writer.shards),
        "val_shards": len(val_writer.shards),
        "elapsed_seconds": elapsed,
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"manifest={manifest_path}")
    print(f"tokenizer={tokenizer_path}")
    print(f"parquet_files={len(parquet_files)}")
    print(f"rows_seen={pass1_rows}")
    print(f"rows_sharded={pass2_rows}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"train_tokens={train_writer.total_tokens}")
    print(f"val_tokens={val_writer.total_tokens}")
    print(f"train_shards={len(train_writer.shards)}")
    print(f"val_shards={len(val_writer.shards)}")
    print(f"report={report_output}")
    print(f"elapsed_seconds={elapsed:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

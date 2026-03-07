"""Corpus sharding utilities for tokenized training datasets."""

from __future__ import annotations

import json
import random
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm.tokenizer import (
    load_tokenizer,
    tokenizer_contract,
    tokenizer_contract_fingerprint,
    tokenizer_fingerprint,
)


@dataclass
class ShardConfig:
    input_path: Path
    tokenizer_path: Path
    output_dir: Path
    shard_size_tokens: int = 5_000_000
    val_ratio: float = 0.01
    seed: int = 42
    max_lines: int = 0


class _ShardWriter:
    def __init__(
        self, split: str, output_dir: Path, array_type: str, shard_size_tokens: int
    ) -> None:
        self.split = split
        self.output_dir = output_dir
        self.array_type = array_type
        self.shard_size_tokens = shard_size_tokens
        self.buffer = array(self.array_type)
        self.shard_index = 0
        self.shards: list[dict[str, int | str]] = []
        self.total_tokens = 0

    def _write_full_shard_from_buffer(self) -> None:
        shard_tokens = self.buffer[: self.shard_size_tokens]
        del self.buffer[: self.shard_size_tokens]

        shard_path = self.output_dir / f"{self.split}_{self.shard_index:06d}.bin"
        with shard_path.open("wb") as handle:
            shard_tokens.tofile(handle)

        self.shards.append(
            {
                "path": shard_path.name,
                "tokens": len(shard_tokens),
            }
        )
        self.shard_index += 1

    def add_tokens(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        self.buffer.extend(token_ids)
        self.total_tokens += len(token_ids)

        while len(self.buffer) >= self.shard_size_tokens:
            self._write_full_shard_from_buffer()

    def finalize(self) -> None:
        if not self.buffer:
            return

        shard_path = self.output_dir / f"{self.split}_{self.shard_index:06d}.bin"
        with shard_path.open("wb") as handle:
            self.buffer.tofile(handle)

        self.shards.append(
            {
                "path": shard_path.name,
                "tokens": len(self.buffer),
            }
        )
        self.buffer = array(self.array_type)
        self.shard_index += 1


def _array_type_for_vocab(vocab_size: int) -> tuple[str, str]:
    if vocab_size <= 65535:
        return "H", "uint16"
    return "I", "uint32"


def shard_corpus(config: ShardConfig) -> dict[str, Any]:
    if config.shard_size_tokens <= 0:
        raise ValueError("shard_size_tokens must be > 0")
    if not 0.0 <= config.val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if config.max_lines < 0:
        raise ValueError("max_lines must be >= 0")

    tokenizer_path = config.tokenizer_path.resolve()
    tokenizer = load_tokenizer(tokenizer_path)
    array_type, token_dtype = _array_type_for_vocab(tokenizer.vocab_size)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_writer = _ShardWriter("train", config.output_dir, array_type, config.shard_size_tokens)
    val_writer = _ShardWriter("val", config.output_dir, array_type, config.shard_size_tokens)
    rng = random.Random(config.seed)

    line_count = 0
    with config.input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if config.max_lines and line_count >= config.max_lines:
                break
            line_count += 1
            token_ids = tokenizer.encode(line)
            if rng.random() < config.val_ratio:
                val_writer.add_tokens(token_ids)
            else:
                train_writer.add_tokens(token_ids)

    train_writer.finalize()
    val_writer.finalize()

    manifest = {
        "input_path": str(config.input_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_hash": tokenizer_fingerprint(tokenizer_path),
        "tokenizer_contract": tokenizer_contract(tokenizer_path),
        "tokenizer_contract_hash": tokenizer_contract_fingerprint(tokenizer_path),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "token_dtype": token_dtype,
        "shard_size_tokens": config.shard_size_tokens,
        "val_ratio": config.val_ratio,
        "seed": config.seed,
        "max_lines": config.max_lines,
        "line_count": line_count,
        "train": {
            "total_tokens": train_writer.total_tokens,
            "shards": train_writer.shards,
        },
        "val": {
            "total_tokens": val_writer.total_tokens,
            "shards": val_writer.shards,
        },
    }

    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def iter_corpus_files(
    *,
    input_dir: Path,
    pattern: str = "*.txt",
    exclude_patterns: Iterable[str] | None = None,
    include_stems: set[str] | None = None,
    limit_files: int = 0,
) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input_dir must exist and be a directory: {input_dir}")

    exclude_patterns = tuple(exclude_patterns or ())
    files: list[Path] = []
    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file():
            continue
        if any(path.match(ex_pat) for ex_pat in exclude_patterns):
            continue
        if include_stems is not None and path.stem not in include_stems:
            continue
        files.append(path)
        if limit_files and len(files) >= limit_files:
            break
    return files


def shard_corpora_batch(
    *,
    input_files: list[Path],
    tokenizer_path: Path,
    output_root: Path,
    shard_size_tokens: int,
    val_ratio: float,
    seed: int,
    max_lines: int,
    skip_existing: bool = True,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for file_path in input_files:
        output_dir = output_root / file_path.stem
        manifest_path = output_dir / "manifest.json"
        if skip_existing and manifest_path.exists():
            results.append(
                {
                    "input_path": str(file_path),
                    "output_dir": str(output_dir),
                    "status": "skipped_existing",
                }
            )
            continue

        config = ShardConfig(
            input_path=file_path,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            shard_size_tokens=shard_size_tokens,
            val_ratio=val_ratio,
            seed=seed,
            max_lines=max_lines,
        )
        manifest = shard_corpus(config)
        results.append(
            {
                "input_path": str(file_path),
                "output_dir": str(output_dir),
                "status": "ok",
                "line_count": int(manifest["line_count"]),
                "train_tokens": int(manifest["train"]["total_tokens"]),
                "val_tokens": int(manifest["val"]["total_tokens"]),
                "train_shards": len(manifest["train"]["shards"]),
                "val_shards": len(manifest["val"]["shards"]),
            }
        )
    return results

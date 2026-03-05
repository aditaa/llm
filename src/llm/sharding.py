"""Corpus sharding utilities for tokenized training datasets."""

from __future__ import annotations

import json
import random
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.tokenizer import BasicCharTokenizer


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

    tokenizer = BasicCharTokenizer.load(config.tokenizer_path)
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
        "tokenizer_path": str(config.tokenizer_path),
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

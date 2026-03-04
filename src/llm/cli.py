"""Command-line utilities for data and experiment scaffolding."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm.tokenizer import BasicCharTokenizer


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def cmd_stats(input_path: str) -> int:
    text = _read_text(input_path)
    chars = len(text)
    lines = len(text.splitlines())
    unique_chars = len(set(text))
    print(f"file={input_path}")
    print(f"chars={chars}")
    print(f"lines={lines}")
    print(f"unique_chars={unique_chars}")
    return 0


def cmd_build_vocab(input_path: str, output_path: str) -> int:
    text = _read_text(input_path)
    tokenizer = BasicCharTokenizer.train(text)
    tokenizer.save(output_path)
    print(f"saved vocab to {output_path} (vocab_size={tokenizer.vocab_size})")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM project helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser("stats", help="Print quick corpus stats")
    stats_parser.add_argument("--input", required=True, help="Input text file")

    vocab_parser = subparsers.add_parser("build-vocab", help="Build char vocab JSON")
    vocab_parser.add_argument("--input", required=True, help="Input text file")
    vocab_parser.add_argument("--output", required=True, help="Output vocab JSON path")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "stats":
        return cmd_stats(args.input)
    if args.command == "build-vocab":
        return cmd_build_vocab(args.input, args.output)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line utilities for data and experiment scaffolding."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm.tokenizer import BasicCharTokenizer
from llm.zim import ZimExtractConfig, extract_text_from_zim


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


def cmd_extract_zim_text(
    input_zim: str,
    output_path: str,
    query: str,
    max_articles: int,
    min_chars: int,
    max_chars: int,
    include_title: bool,
    paths_file: str | None,
) -> int:
    config = ZimExtractConfig(
        zim_path=Path(input_zim),
        output_path=Path(output_path),
        query=query,
        max_articles=max_articles,
        min_chars=min_chars,
        max_chars=max_chars,
        include_title=include_title,
        paths_file=Path(paths_file) if paths_file else None,
    )
    stats = extract_text_from_zim(config)
    print(f"input_zim={input_zim}")
    print(f"output={output_path}")
    print(f"seen_paths={stats['seen_paths']}")
    print(f"written_articles={stats['written_articles']}")
    print(f"skipped_nontext={stats['skipped_nontext']}")
    print(f"skipped_too_short={stats['skipped_too_short']}")
    print(f"errors={stats['errors']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM project helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser("stats", help="Print quick corpus stats")
    stats_parser.add_argument("--input", required=True, help="Input text file")

    vocab_parser = subparsers.add_parser("build-vocab", help="Build char vocab JSON")
    vocab_parser.add_argument("--input", required=True, help="Input text file")
    vocab_parser.add_argument("--output", required=True, help="Output vocab JSON path")

    train_tok_parser = subparsers.add_parser("train-tokenizer", help="Alias for build-vocab")
    train_tok_parser.add_argument("--input", required=True, help="Input text file")
    train_tok_parser.add_argument("--output", required=True, help="Output vocab JSON path")

    zim_parser = subparsers.add_parser("extract-zim-text", help="Extract text corpus from ZIM")
    zim_parser.add_argument("--input-zim", required=True, help="Path to .zim file on server")
    zim_parser.add_argument("--output", required=True, help="Output corpus text path")
    zim_parser.add_argument("--query", default="*", help="Search query for indexed article paths")
    zim_parser.add_argument(
        "--max-articles", type=int, default=10000, help="Maximum articles to write (0 = no limit)"
    )
    zim_parser.add_argument("--min-chars", type=int, default=200, help="Skip shorter text blocks")
    zim_parser.add_argument(
        "--max-chars", type=int, default=0, help="Truncate each article text to this length"
    )
    zim_parser.add_argument(
        "--no-title", action="store_true", help="Do not prepend article title to text output"
    )
    zim_parser.add_argument(
        "--paths-file",
        default=None,
        help="Optional newline-separated entry paths if ZIM has no fulltext index",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "stats":
        return cmd_stats(args.input)
    if args.command == "build-vocab":
        return cmd_build_vocab(args.input, args.output)
    if args.command == "train-tokenizer":
        return cmd_build_vocab(args.input, args.output)
    if args.command == "extract-zim-text":
        return cmd_extract_zim_text(
            input_zim=args.input_zim,
            output_path=args.output,
            query=args.query,
            max_articles=args.max_articles,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            include_title=not args.no_title,
            paths_file=args.paths_file,
        )
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

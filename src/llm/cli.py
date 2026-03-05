"""Command-line utilities for data and experiment scaffolding."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm.integrity import verify_shards
from llm.sharding import ShardConfig, shard_corpus
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


def cmd_shard_corpus(
    input_path: str,
    tokenizer_path: str,
    output_dir: str,
    shard_size_tokens: int,
    val_ratio: float,
    seed: int,
    max_lines: int,
) -> int:
    config = ShardConfig(
        input_path=Path(input_path),
        tokenizer_path=Path(tokenizer_path),
        output_dir=Path(output_dir),
        shard_size_tokens=shard_size_tokens,
        val_ratio=val_ratio,
        seed=seed,
        max_lines=max_lines,
    )
    manifest = shard_corpus(config)
    print(f"input={manifest['input_path']}")
    print(f"tokenizer={manifest['tokenizer_path']}")
    print(f"output_dir={output_dir}")
    print(f"token_dtype={manifest['token_dtype']}")
    print(f"line_count={manifest['line_count']}")
    print(f"train_tokens={manifest['train']['total_tokens']}")
    print(f"train_shards={len(manifest['train']['shards'])}")
    print(f"val_tokens={manifest['val']['total_tokens']}")
    print(f"val_shards={len(manifest['val']['shards'])}")
    return 0


def cmd_verify_shards(
    path: str,
    check_token_ranges: bool,
    chunk_tokens: int,
    raw_zim_dir: str | None,
    strict_source: bool,
) -> int:
    results = verify_shards(
        path=Path(path),
        check_token_ranges=check_token_ranges,
        chunk_tokens=chunk_tokens,
        raw_zim_dir=Path(raw_zim_dir) if raw_zim_dir else None,
        strict_source=strict_source,
    )

    ok_count = 0
    fail_count = 0
    for result in results:
        print(f"manifest={result['manifest_path']}")
        print(f"ok={int(result['ok'])}")
        print(f"files_checked={result['files_checked']}")
        print(f"tokens_checked={result['tokens_checked']}")
        for warning in result["warnings"]:
            print(f"warning={warning}")
        for error in result["errors"]:
            print(f"error={error}")

        if result["ok"]:
            ok_count += 1
        else:
            fail_count += 1

    print(f"manifests_ok={ok_count}")
    print(f"manifests_failed={fail_count}")
    return 0 if fail_count == 0 else 1


def cmd_train(
    shards_path: str,
    output_dir: str,
    max_steps: int,
    batch_size: int,
    context_length: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    eval_interval: int,
    eval_steps: int,
    log_interval: int,
    seed: int,
    device: str,
    n_layers: int,
    n_heads: int,
    d_model: int,
    dropout: float,
    resume_from: str | None,
) -> int:
    from llm.train import TrainConfig, run_training

    config = TrainConfig(
        shards_path=Path(shards_path),
        output_dir=Path(output_dir),
        max_steps=max_steps,
        batch_size=batch_size,
        context_length=context_length,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        eval_interval=eval_interval,
        eval_steps=eval_steps,
        log_interval=log_interval,
        seed=seed,
        device=device,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        dropout=dropout,
        resume_from=Path(resume_from) if resume_from else None,
    )
    result = run_training(config)
    print(f"output_dir={result['output_dir']}")
    print(f"max_steps={result['max_steps']}")
    print(f"start_step={result['start_step']}")
    print(f"tokenizer_path={result['tokenizer_path']}")
    print(f"tokenizer_hash={result['tokenizer_hash']}")
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

    shard_parser = subparsers.add_parser(
        "shard-corpus", help="Tokenize corpus and write token shards"
    )
    shard_parser.add_argument("--input", required=True, help="Input corpus text file")
    shard_parser.add_argument("--tokenizer", required=True, help="Tokenizer vocab JSON path")
    shard_parser.add_argument(
        "--output-dir", required=True, help="Output directory for shard files"
    )
    shard_parser.add_argument(
        "--shard-size-tokens", type=int, default=5_000_000, help="Tokens per shard file"
    )
    shard_parser.add_argument(
        "--val-ratio", type=float, default=0.01, help="Validation split ratio"
    )
    shard_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for split assignment"
    )
    shard_parser.add_argument(
        "--max-lines", type=int, default=0, help="Optional line cap for test runs (0 = all lines)"
    )

    verify_parser = subparsers.add_parser(
        "verify-shards", help="Validate shard manifests and shard binary integrity"
    )
    verify_parser.add_argument(
        "--path",
        required=True,
        help="Path to manifest.json or directory containing manifest.json files",
    )
    verify_parser.add_argument(
        "--no-token-range-check",
        action="store_true",
        help="Skip deep token-id range scans for faster checks",
    )
    verify_parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=1_000_000,
        help="Token chunk size for range checks",
    )
    verify_parser.add_argument(
        "--raw-zim-dir",
        default=None,
        help="Optional raw ZIM directory to validate matching source archives",
    )
    verify_parser.add_argument(
        "--strict-source",
        action="store_true",
        help="Fail if matching source ZIM is missing when --raw-zim-dir is set",
    )

    train_parser = subparsers.add_parser(
        "train", help="Run baseline GPT training on shard manifests"
    )
    train_parser.add_argument(
        "--shards-path",
        required=True,
        help="Path to manifest.json or directory containing manifest.json files",
    )
    train_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for checkpoints and training logs",
    )
    train_parser.add_argument("--max-steps", type=int, default=1000, help="Number of train steps")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument(
        "--context-length", type=int, default=256, help="Sequence length for training windows"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Optimizer learning rate"
    )
    train_parser.add_argument(
        "--weight-decay", type=float, default=0.1, help="AdamW weight decay"
    )
    train_parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Gradient clipping max norm"
    )
    train_parser.add_argument(
        "--eval-interval", type=int, default=100, help="Run validation every N steps"
    )
    train_parser.add_argument(
        "--eval-steps", type=int, default=20, help="Validation batches per eval cycle"
    )
    train_parser.add_argument(
        "--log-interval", type=int, default=10, help="Log train loss every N steps"
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (auto, cpu, cuda, cuda:0, ...)",
    )
    train_parser.add_argument("--n-layers", type=int, default=4, help="Transformer block count")
    train_parser.add_argument("--n-heads", type=int, default=4, help="Attention head count")
    train_parser.add_argument("--d-model", type=int, default=256, help="Model hidden size")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    train_parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional checkpoint path to resume training from",
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
    if args.command == "shard-corpus":
        return cmd_shard_corpus(
            input_path=args.input,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            shard_size_tokens=args.shard_size_tokens,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_lines=args.max_lines,
        )
    if args.command == "verify-shards":
        return cmd_verify_shards(
            path=args.path,
            check_token_ranges=not args.no_token_range_check,
            chunk_tokens=args.chunk_tokens,
            raw_zim_dir=args.raw_zim_dir,
            strict_source=args.strict_source,
        )
    if args.command == "train":
        return cmd_train(
            shards_path=args.shards_path,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            context_length=args.context_length,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            eval_interval=args.eval_interval,
            eval_steps=args.eval_steps,
            log_interval=args.log_interval,
            seed=args.seed,
            device=args.device,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            dropout=args.dropout,
            resume_from=args.resume_from,
        )
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

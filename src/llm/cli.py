"""Command-line utilities for data and experiment scaffolding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.corpus import (
    CleanCorpusConfig,
    CorpusQualityConfig,
    analyze_corpora,
    clean_corpora_batch,
    load_boilerplate_lines_from_report,
    save_clean_report,
    save_quality_report,
)
from llm.integrity import verify_shards
from llm.sharding import ShardConfig, iter_corpus_files, shard_corpora_batch, shard_corpus
from llm.tokenizer import BasicCharTokenizer
from llm.zim import ZimExtractConfig, extract_text_from_zim


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _collect_dataset_stems_from_manifests(path: Path) -> set[str]:
    manifests: list[Path]
    if path.is_file():
        manifests = [path]
    else:
        manifests = sorted(path.rglob("manifest.json"))
    return {manifest.parent.name for manifest in manifests}


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


def cmd_train_tokenizer_global(
    input_dir: str,
    output_path: str,
    pattern: str,
    exclude_pattern: list[str],
    from_shards_path: str | None,
    max_files: int,
    max_chars_per_file: int,
) -> int:
    include_stems: set[str] | None = None
    if from_shards_path is not None:
        include_stems = _collect_dataset_stems_from_manifests(Path(from_shards_path))

    files = iter_corpus_files(
        input_dir=Path(input_dir),
        pattern=pattern,
        exclude_patterns=exclude_pattern,
        include_stems=include_stems,
        limit_files=max_files,
    )
    if not files:
        raise ValueError("no input files matched for global tokenizer training")

    tokenizer, stats = BasicCharTokenizer.train_from_files(
        files,
        max_chars_per_file=max_chars_per_file,
    )
    tokenizer.save(output_path)

    metadata_path = Path(output_path).with_suffix(Path(output_path).suffix + ".meta.json")
    metadata = {
        "output_path": output_path,
        "pattern": pattern,
        "exclude_patterns": exclude_pattern,
        "from_shards_path": from_shards_path,
        "files": [str(p) for p in files],
        "stats": stats,
        "vocab_size": tokenizer.vocab_size,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"output={output_path}")
    print(f"metadata={metadata_path}")
    print(f"files_used={len(files)}")
    print(f"chars_read={stats['chars_read']}")
    print(f"unique_chars={stats['unique_chars']}")
    print(f"vocab_size={tokenizer.vocab_size}")
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


def cmd_corpus_quality_report(
    input_dir: str,
    output_path: str,
    pattern: str,
    exclude_pattern: list[str],
    max_files: int,
    max_lines_per_file: int,
    max_total_lines: int,
    top_k: int,
    boilerplate_min_occurrences: int,
    boilerplate_min_files: int,
    boilerplate_min_chars: int,
    boilerplate_max_chars: int,
) -> int:
    files = iter_corpus_files(
        input_dir=Path(input_dir),
        pattern=pattern,
        exclude_patterns=exclude_pattern,
        include_stems=None,
        limit_files=max_files,
    )
    if not files:
        raise ValueError("no input files matched for quality report")

    report = analyze_corpora(
        input_files=files,
        config=CorpusQualityConfig(
            top_k=top_k,
            max_lines_per_file=max_lines_per_file,
            max_total_lines=max_total_lines,
            boilerplate_min_occurrences=boilerplate_min_occurrences,
            boilerplate_min_files=boilerplate_min_files,
            boilerplate_min_chars=boilerplate_min_chars,
            boilerplate_max_chars=boilerplate_max_chars,
        ),
    )
    save_quality_report(report, Path(output_path))

    print(f"output={output_path}")
    print(f"files_seen={report['files_seen']}")
    print(f"lines_seen={report['lines_seen']}")
    print(f"lines_nonempty={report['lines_nonempty']}")
    print(f"duplicate_nonempty_lines={report['duplicate_nonempty_lines']}")
    print(f"boilerplate_candidates={len(report['boilerplate_candidates'])}")
    print(f"truncated={int(report['truncated'])}")
    return 0


def cmd_clean_corpus_batch(
    input_dir: str,
    output_dir: str,
    pattern: str,
    exclude_pattern: list[str],
    max_files: int,
    max_lines_per_file: int,
    boilerplate_report: str | None,
    min_chars: int,
    max_chars: int,
    min_alpha_ratio: float,
    max_digit_ratio: float,
    dedupe_within_file: bool,
    dedupe_global: bool,
    skip_existing: bool,
    output_suffix: str,
    decode_html_entities: bool,
    strip_html_tags: bool,
    strip_site_suffixes: bool,
    strip_nav_phrases: bool,
    strip_stack_metadata: bool,
    collapse_repeated_prefix: bool,
    strip_inline_score_tokens: bool,
    english_only: bool,
    english_min_words: int,
    english_min_stopword_ratio: float,
    english_min_stopword_count: int,
    english_min_latin_ratio: float,
    drop_code_like: bool,
    code_symbol_ratio_threshold: float,
    code_keyword_hits_threshold: int,
    report_output: str,
) -> int:
    files = iter_corpus_files(
        input_dir=Path(input_dir),
        pattern=pattern,
        exclude_patterns=exclude_pattern,
        include_stems=None,
        limit_files=max_files,
    )
    if not files:
        raise ValueError("no input files matched for clean-corpus-batch")

    boilerplate_lines: set[str] = set()
    if boilerplate_report is not None:
        boilerplate_lines = load_boilerplate_lines_from_report(Path(boilerplate_report))

    report = clean_corpora_batch(
        input_files=files,
        output_dir=Path(output_dir),
        config=CleanCorpusConfig(
            min_chars=min_chars,
            max_chars=max_chars,
            min_alpha_ratio=min_alpha_ratio,
            max_digit_ratio=max_digit_ratio,
            dedupe_within_file=dedupe_within_file,
            dedupe_global=dedupe_global,
            max_lines_per_file=max_lines_per_file,
            skip_existing=skip_existing,
            output_suffix=output_suffix,
            decode_html_entities=decode_html_entities,
            strip_html_tags=strip_html_tags,
            strip_site_suffixes=strip_site_suffixes,
            strip_nav_phrases=strip_nav_phrases,
            strip_stack_metadata=strip_stack_metadata,
            collapse_repeated_prefix=collapse_repeated_prefix,
            strip_inline_score_tokens=strip_inline_score_tokens,
            english_only=english_only,
            english_min_words=english_min_words,
            english_min_stopword_ratio=english_min_stopword_ratio,
            english_min_stopword_count=english_min_stopword_count,
            english_min_latin_ratio=english_min_latin_ratio,
            drop_code_like=drop_code_like,
            code_symbol_ratio_threshold=code_symbol_ratio_threshold,
            code_keyword_hits_threshold=code_keyword_hits_threshold,
        ),
        boilerplate_lines=boilerplate_lines,
    )
    save_clean_report(report, Path(report_output))

    totals = report["totals"]
    print(f"output_dir={output_dir}")
    print(f"report={report_output}")
    print(f"files_total={len(files)}")
    print(f"files_skipped_existing={totals['files_skipped_existing']}")
    print(f"input_lines={totals['input_lines']}")
    print(f"kept_lines={totals['kept_lines']}")
    print(f"removed_empty={totals['removed_empty']}")
    print(f"removed_too_short={totals['removed_too_short']}")
    print(f"removed_too_long={totals['removed_too_long']}")
    print(f"removed_low_alpha={totals['removed_low_alpha']}")
    print(f"removed_high_digit={totals['removed_high_digit']}")
    print(f"removed_boilerplate={totals['removed_boilerplate']}")
    print(f"removed_non_english={totals['removed_non_english']}")
    print(f"removed_code_like={totals['removed_code_like']}")
    print(f"removed_duplicate_within={totals['removed_duplicate_within']}")
    print(f"removed_duplicate_global={totals['removed_duplicate_global']}")
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


def cmd_shard_corpus_batch(
    input_dir: str,
    tokenizer_path: str,
    output_root: str,
    pattern: str,
    exclude_pattern: list[str],
    from_shards_path: str | None,
    max_files: int,
    shard_size_tokens: int,
    val_ratio: float,
    seed: int,
    max_lines: int,
    skip_existing: bool,
) -> int:
    include_stems: set[str] | None = None
    if from_shards_path is not None:
        include_stems = _collect_dataset_stems_from_manifests(Path(from_shards_path))

    files = iter_corpus_files(
        input_dir=Path(input_dir),
        pattern=pattern,
        exclude_patterns=exclude_pattern,
        include_stems=include_stems,
        limit_files=max_files,
    )
    if not files:
        raise ValueError("no input files matched for batch sharding")

    results = shard_corpora_batch(
        input_files=files,
        tokenizer_path=Path(tokenizer_path),
        output_root=Path(output_root),
        shard_size_tokens=shard_size_tokens,
        val_ratio=val_ratio,
        seed=seed,
        max_lines=max_lines,
        skip_existing=skip_existing,
    )

    ok = 0
    skipped = 0
    for row in results:
        print(f"input={row['input_path']}")
        print(f"output_dir={row['output_dir']}")
        print(f"status={row['status']}")
        if row["status"] == "ok":
            ok += 1
            print(f"line_count={row['line_count']}")
            print(f"train_tokens={row['train_tokens']}")
            print(f"val_tokens={row['val_tokens']}")
            print(f"train_shards={row['train_shards']}")
            print(f"val_shards={row['val_shards']}")
        elif row["status"] == "skipped_existing":
            skipped += 1

    print(f"files_total={len(results)}")
    print(f"files_ok={ok}")
    print(f"files_skipped_existing={skipped}")
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


def cmd_generate(
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
    seed: int,
    no_stop_on_eos: bool,
) -> int:
    from llm.generate import GenerateConfig, run_generation

    config = GenerateConfig(
        checkpoint_path=Path(checkpoint_path),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
        seed=seed,
        stop_on_eos=not no_stop_on_eos,
    )
    result = run_generation(config)
    print(f"checkpoint_path={result['checkpoint_path']}")
    print(f"tokenizer_path={result['tokenizer_path']}")
    print(f"device={result['device']}")
    print(f"seed={result['seed']}")
    print(f"token_count={result['token_count']}")
    print("output_text_start")
    print(result["output_text"])
    print("output_text_end")
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

    global_tok_parser = subparsers.add_parser(
        "train-tokenizer-global",
        help="Train one shared char tokenizer from many corpus files",
    )
    global_tok_parser.add_argument("--input-dir", required=True, help="Directory of corpus files")
    global_tok_parser.add_argument("--output", required=True, help="Output global vocab JSON path")
    global_tok_parser.add_argument("--pattern", default="*.txt", help="Glob pattern")
    global_tok_parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=["*.paths.txt"],
        help="Glob pattern to exclude (repeatable)",
    )
    global_tok_parser.add_argument(
        "--from-shards-path",
        default=None,
        help="Optional path to existing shard manifests; include only matching dataset stems",
    )
    global_tok_parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on file count (0 = all matches)",
    )
    global_tok_parser.add_argument(
        "--max-chars-per-file",
        type=int,
        default=0,
        help="Optional cap on chars read per file (0 = entire file)",
    )

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

    quality_parser = subparsers.add_parser(
        "corpus-quality-report",
        help="Analyze extracted corpus files and emit a quality/boilerplate report",
    )
    quality_parser.add_argument("--input-dir", required=True, help="Directory of corpus files")
    quality_parser.add_argument("--output", required=True, help="Output quality JSON report path")
    quality_parser.add_argument("--pattern", default="*.txt", help="Glob pattern")
    quality_parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=["*.paths.txt"],
        help="Glob pattern to exclude (repeatable)",
    )
    quality_parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on file count (0 = all matches)",
    )
    quality_parser.add_argument(
        "--max-lines-per-file",
        type=int,
        default=0,
        help="Optional line cap per file (0 = all lines)",
    )
    quality_parser.add_argument(
        "--max-total-lines",
        type=int,
        default=0,
        help="Optional global line cap across all files (0 = all lines)",
    )
    quality_parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top repeated lines to include in report",
    )
    quality_parser.add_argument(
        "--boilerplate-min-occurrences",
        type=int,
        default=20,
        help="Minimum repeat count to mark boilerplate candidate",
    )
    quality_parser.add_argument(
        "--boilerplate-min-files",
        type=int,
        default=5,
        help="Minimum distinct file count for boilerplate candidate",
    )
    quality_parser.add_argument(
        "--boilerplate-min-chars",
        type=int,
        default=30,
        help="Minimum line length for boilerplate candidate",
    )
    quality_parser.add_argument(
        "--boilerplate-max-chars",
        type=int,
        default=240,
        help="Maximum line length for boilerplate candidate",
    )

    clean_parser = subparsers.add_parser(
        "clean-corpus-batch",
        help="Clean many extracted corpus files before tokenizer training",
    )
    clean_parser.add_argument("--input-dir", required=True, help="Directory of corpus files")
    clean_parser.add_argument("--output-dir", required=True, help="Output directory")
    clean_parser.add_argument("--pattern", default="*.txt", help="Glob pattern")
    clean_parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=["*.paths.txt"],
        help="Glob pattern to exclude (repeatable)",
    )
    clean_parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on file count (0 = all matches)",
    )
    clean_parser.add_argument(
        "--max-lines-per-file",
        type=int,
        default=0,
        help="Optional line cap per file (0 = all lines)",
    )
    clean_parser.add_argument(
        "--boilerplate-report",
        default=None,
        help="Optional quality report JSON; remove lines marked as boilerplate candidates",
    )
    clean_parser.add_argument("--min-chars", type=int, default=40, help="Minimum kept line length")
    clean_parser.add_argument("--max-chars", type=int, default=0, help="Maximum kept line length")
    clean_parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.20,
        help="Drop lines with lower alphabetic-char ratio",
    )
    clean_parser.add_argument(
        "--max-digit-ratio",
        type=float,
        default=0.35,
        help="Drop lines with higher digit-char ratio",
    )
    clean_parser.add_argument(
        "--no-dedupe-within-file",
        action="store_true",
        help="Keep duplicate lines within each output file",
    )
    clean_parser.add_argument(
        "--dedupe-global",
        action="store_true",
        help="Drop duplicate lines across all output files",
    )
    clean_parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rebuild cleaned file even when output already exists",
    )
    clean_parser.add_argument(
        "--output-suffix",
        default=".clean.txt",
        help="Suffix appended to each cleaned output filename",
    )
    clean_parser.add_argument(
        "--no-decode-html-entities",
        action="store_true",
        help="Disable HTML entity decoding (for example &amp; -> &)",
    )
    clean_parser.add_argument(
        "--no-strip-html-tags",
        action="store_true",
        help="Disable stripping HTML/XML-like tags",
    )
    clean_parser.add_argument(
        "--no-strip-site-suffixes",
        action="store_true",
        help="Disable stripping site suffixes like '- Stack Overflow'",
    )
    clean_parser.add_argument(
        "--no-strip-nav-phrases",
        action="store_true",
        help="Disable stripping repeated navigation phrase fragments",
    )
    clean_parser.add_argument(
        "--no-strip-stack-metadata",
        action="store_true",
        help="Disable stripping StackExchange timeline metadata fragments",
    )
    clean_parser.add_argument(
        "--no-collapse-repeated-prefix",
        action="store_true",
        help="Disable collapsing duplicated prefix spans",
    )
    clean_parser.add_argument(
        "--no-strip-inline-score-tokens",
        action="store_true",
        help="Disable stripping inline score tokens (for example '? 0 Body...')",
    )
    clean_parser.add_argument(
        "--en-only",
        action="store_true",
        help="Enable heuristic English-only filtering during cleanup",
    )
    clean_parser.add_argument(
        "--en-min-words",
        type=int,
        default=6,
        help="Minimum word count for English-line acceptance",
    )
    clean_parser.add_argument(
        "--en-min-stopword-ratio",
        type=float,
        default=0.02,
        help="Minimum English stopword ratio for English-line acceptance",
    )
    clean_parser.add_argument(
        "--en-min-stopword-count",
        type=int,
        default=1,
        help="Minimum English stopword count for English-line acceptance",
    )
    clean_parser.add_argument(
        "--en-min-latin-ratio",
        type=float,
        default=0.90,
        help="Minimum Latin-letter ratio for English-line acceptance",
    )
    clean_parser.add_argument(
        "--no-drop-code-like",
        action="store_true",
        help="Disable dropping lines detected as code-like",
    )
    clean_parser.add_argument(
        "--code-symbol-ratio-threshold",
        type=float,
        default=0.08,
        help="Drop line as code-like if symbol ratio exceeds this threshold",
    )
    clean_parser.add_argument(
        "--code-keyword-hits-threshold",
        type=int,
        default=2,
        help="Drop line as code-like if code-keyword hits meet this threshold",
    )
    clean_parser.add_argument(
        "--report-output",
        default="artifacts/reports/clean_corpus_batch_report.json",
        help="Output JSON report path",
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

    shard_batch_parser = subparsers.add_parser(
        "shard-corpus-batch",
        help="Shard many corpus files with one shared tokenizer",
    )
    shard_batch_parser.add_argument("--input-dir", required=True, help="Directory of corpus files")
    shard_batch_parser.add_argument("--tokenizer", required=True, help="Tokenizer vocab JSON path")
    shard_batch_parser.add_argument(
        "--output-root",
        required=True,
        help="Output root; each corpus gets output-root/<stem>/manifest.json",
    )
    shard_batch_parser.add_argument("--pattern", default="*.txt", help="Glob pattern")
    shard_batch_parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=["*.paths.txt"],
        help="Glob pattern to exclude (repeatable)",
    )
    shard_batch_parser.add_argument(
        "--from-shards-path",
        default=None,
        help="Optional path to existing shard manifests; include only matching dataset stems",
    )
    shard_batch_parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on file count (0 = all matches)",
    )
    shard_batch_parser.add_argument(
        "--shard-size-tokens", type=int, default=5_000_000, help="Tokens per shard file"
    )
    shard_batch_parser.add_argument(
        "--val-ratio", type=float, default=0.01, help="Validation split ratio"
    )
    shard_batch_parser.add_argument("--seed", type=int, default=42, help="Split RNG seed")
    shard_batch_parser.add_argument(
        "--max-lines", type=int, default=0, help="Optional line cap per corpus (0 = all lines)"
    )
    shard_batch_parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Rebuild corpus shards even if manifest already exists in output root",
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
    train_parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
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

    gen_parser = subparsers.add_parser("generate", help="Generate text from a model checkpoint")
    gen_parser.add_argument("--checkpoint", required=True, help="Checkpoint path (*.pt)")
    gen_parser.add_argument("--prompt", required=True, help="Prompt text")
    gen_parser.add_argument(
        "--max-new-tokens", type=int, default=200, help="Maximum new tokens to sample"
    )
    gen_parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature (> 0)"
    )
    gen_parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling cutoff (0 = disabled)",
    )
    gen_parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (auto, cpu, cuda, cuda:0, ...)",
    )
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.add_argument(
        "--no-stop-on-eos",
        action="store_true",
        help="Disable early stop when EOS token is sampled",
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
    if args.command == "train-tokenizer-global":
        return cmd_train_tokenizer_global(
            input_dir=args.input_dir,
            output_path=args.output,
            pattern=args.pattern,
            exclude_pattern=args.exclude_pattern,
            from_shards_path=args.from_shards_path,
            max_files=args.max_files,
            max_chars_per_file=args.max_chars_per_file,
        )
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
    if args.command == "corpus-quality-report":
        return cmd_corpus_quality_report(
            input_dir=args.input_dir,
            output_path=args.output,
            pattern=args.pattern,
            exclude_pattern=args.exclude_pattern,
            max_files=args.max_files,
            max_lines_per_file=args.max_lines_per_file,
            max_total_lines=args.max_total_lines,
            top_k=args.top_k,
            boilerplate_min_occurrences=args.boilerplate_min_occurrences,
            boilerplate_min_files=args.boilerplate_min_files,
            boilerplate_min_chars=args.boilerplate_min_chars,
            boilerplate_max_chars=args.boilerplate_max_chars,
        )
    if args.command == "clean-corpus-batch":
        return cmd_clean_corpus_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pattern=args.pattern,
            exclude_pattern=args.exclude_pattern,
            max_files=args.max_files,
            max_lines_per_file=args.max_lines_per_file,
            boilerplate_report=args.boilerplate_report,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            min_alpha_ratio=args.min_alpha_ratio,
            max_digit_ratio=args.max_digit_ratio,
            dedupe_within_file=not args.no_dedupe_within_file,
            dedupe_global=args.dedupe_global,
            skip_existing=not args.no_skip_existing,
            output_suffix=args.output_suffix,
            decode_html_entities=not args.no_decode_html_entities,
            strip_html_tags=not args.no_strip_html_tags,
            strip_site_suffixes=not args.no_strip_site_suffixes,
            strip_nav_phrases=not args.no_strip_nav_phrases,
            strip_stack_metadata=not args.no_strip_stack_metadata,
            collapse_repeated_prefix=not args.no_collapse_repeated_prefix,
            strip_inline_score_tokens=not args.no_strip_inline_score_tokens,
            english_only=args.en_only,
            english_min_words=args.en_min_words,
            english_min_stopword_ratio=args.en_min_stopword_ratio,
            english_min_stopword_count=args.en_min_stopword_count,
            english_min_latin_ratio=args.en_min_latin_ratio,
            drop_code_like=not args.no_drop_code_like,
            code_symbol_ratio_threshold=args.code_symbol_ratio_threshold,
            code_keyword_hits_threshold=args.code_keyword_hits_threshold,
            report_output=args.report_output,
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
    if args.command == "shard-corpus-batch":
        return cmd_shard_corpus_batch(
            input_dir=args.input_dir,
            tokenizer_path=args.tokenizer,
            output_root=args.output_root,
            pattern=args.pattern,
            exclude_pattern=args.exclude_pattern,
            from_shards_path=args.from_shards_path,
            max_files=args.max_files,
            shard_size_tokens=args.shard_size_tokens,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_lines=args.max_lines,
            skip_existing=not args.no_skip_existing,
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
    if args.command == "generate":
        return cmd_generate(
            checkpoint_path=args.checkpoint,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
            seed=args.seed,
            no_stop_on_eos=args.no_stop_on_eos,
        )
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

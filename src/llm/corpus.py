"""Corpus quality analysis and cleaning helpers."""

from __future__ import annotations

import hashlib
import html
import json
import re
import string
import unicodedata as ud
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WS_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]{0,200}>")
_WORD_RE = re.compile(r"[A-Za-z']+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_SITE_SUFFIX_RE = re.compile(r"\s+-\s*(stack overflow|stack exchange)\b", re.IGNORECASE)
_STACK_SHELL_RE = re.compile(
    r"\bstack overflow stack exchange\b.*?\bpublic questions tags users about\b",
    re.IGNORECASE,
)
_NAV_PHRASE_RE = re.compile(
    r"\b(?:questions|tags|users|about|teams|jobs|companies|products|help)"
    r"(?:\s+(?:questions|tags|users|about|teams|jobs|companies|products|help)){2,}\b",
    re.IGNORECASE,
)
_STACK_BRAND_SEQ_RE = re.compile(
    r"\b(?:stack overflow|stack exchange)(?:\s+(?:stack overflow|stack exchange)){1,}\b",
    re.IGNORECASE,
)
_STACK_TIMELINE_RE = re.compile(
    r"\b(?:public\s+)?(?:asked|active|modified)\b.{0,320}?\bviewed\s+\d[\d,]*\s+times\b",
    re.IGNORECASE,
)
_PUBLIC_TOKEN_RE = re.compile(r"\bpublic\b", re.IGNORECASE)
_INLINE_SCORE_RE = re.compile(
    r"^(?P<prefix>.{20,260}?)\s(?P<score>[+-]?\d{1,4})\s(?P<suffix>[A-Z].+)$"
)
_CODE_HINT_WORDS = {
    "select",
    "insert",
    "update",
    "delete",
    "where",
    "from",
    "join",
    "into",
    "values",
    "function",
    "class",
    "const",
    "let",
    "var",
    "return",
    "import",
    "def",
    "data",
    "frame",
    "ggplot",
    "library",
    "numpy",
    "pandas",
    "sql",
}
_CODE_DROP_RE = re.compile(
    r"(<-\s*data\.frame\()|(\bdata\.frame\()|(::)|(#include\s*<)|(\bSELECT\b.+\bFROM\b)",
    re.IGNORECASE,
)

_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class CorpusQualityConfig:
    top_k: int = 50
    max_lines_per_file: int = 0
    max_total_lines: int = 0
    boilerplate_min_occurrences: int = 20
    boilerplate_min_files: int = 5
    boilerplate_min_chars: int = 30
    boilerplate_max_chars: int = 240


@dataclass(frozen=True)
class CleanCorpusConfig:
    min_chars: int = 40
    max_chars: int = 0
    min_words: int = 6
    min_alpha_ratio: float = 0.20
    max_digit_ratio: float = 0.35
    max_symbol_ratio: float = 0.20
    max_urls_per_line: int = 1
    repeated_token_run_threshold: int = 8
    min_unique_token_ratio: float = 0.35
    dedupe_within_file: bool = True
    dedupe_global: bool = False
    max_lines_per_file: int = 0
    skip_existing: bool = True
    output_suffix: str = ".clean.txt"
    decode_html_entities: bool = True
    strip_html_tags: bool = True
    strip_site_suffixes: bool = True
    strip_nav_phrases: bool = True
    strip_stack_metadata: bool = True
    collapse_repeated_prefix: bool = True
    strip_inline_score_tokens: bool = True
    english_only: bool = False
    english_min_words: int = 6
    english_min_stopword_ratio: float = 0.02
    english_min_stopword_count: int = 1
    english_min_latin_ratio: float = 0.90
    drop_code_like: bool = True
    code_symbol_ratio_threshold: float = 0.08
    code_keyword_hits_threshold: int = 2


def normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for c in text if c.isalpha())
    return alpha / len(text)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(1 for c in text if c.isdigit())
    return digits / len(text)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    symbols = sum(1 for c in text if c in string.punctuation and c not in {"'", "-"})
    return symbols / len(text)


def _url_count(text: str) -> int:
    return len(_URL_RE.findall(text))


def _is_repetitive_token_noise(text: str, config: CleanCorpusConfig) -> bool:
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return False

    if len(tokens) >= config.repeated_token_run_threshold:
        run = 1
        for idx in range(1, len(tokens)):
            if tokens[idx] == tokens[idx - 1]:
                run += 1
                if run >= config.repeated_token_run_threshold:
                    return True
            else:
                run = 1

    if len(tokens) >= 12:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < config.min_unique_token_ratio:
            return True
    return False


def _strip_web_shell(text: str, config: CleanCorpusConfig) -> str:
    out = text
    if config.decode_html_entities:
        out = html.unescape(out)
    if config.strip_html_tags:
        out = _HTML_TAG_RE.sub(" ", out)
    if config.strip_site_suffixes:
        out = _SITE_SUFFIX_RE.sub(" ", out)
    if config.strip_nav_phrases:
        out = _STACK_SHELL_RE.sub(" ", out)
        out = _NAV_PHRASE_RE.sub(" ", out)
        out = _STACK_BRAND_SEQ_RE.sub(" ", out)
    if config.strip_stack_metadata:
        out = _STACK_TIMELINE_RE.sub(" ", out)
        out = _PUBLIC_TOKEN_RE.sub(" ", out)
    if config.collapse_repeated_prefix:
        out = _collapse_repeated_prefix(out)
    if config.strip_inline_score_tokens:
        out = _strip_inline_score_token(out)
    return normalize_whitespace(out)


def _collapse_repeated_prefix(text: str, *, min_words: int = 5, max_words: int = 24) -> str:
    words = text.split()
    if len(words) < min_words * 2:
        return text

    lowered = [w.lower() for w in words]
    upper = min(max_words, len(words) // 2)
    for span in range(upper, min_words - 1, -1):
        if lowered[:span] == lowered[span : span * 2]:
            collapsed = words[:span] + words[span * 2 :]
            return " ".join(collapsed)
    return text


def _strip_inline_score_token(text: str) -> str:
    normalized = normalize_whitespace(text)
    match = _INLINE_SCORE_RE.match(normalized)
    if not match:
        return normalized
    prefix = str(match.group("prefix")).strip()
    suffix = str(match.group("suffix")).strip()
    # Restrict this to question/title-like prefixes to avoid deleting legitimate numbers.
    if "?" not in prefix and ":" not in prefix:
        return normalized
    if len(suffix) < 20:
        return normalized
    return normalize_whitespace(f"{prefix} {suffix}")


def _looks_english(text: str, config: CleanCorpusConfig) -> bool:
    words = _WORD_RE.findall(text.lower())
    if len(words) < config.english_min_words:
        return False

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if "LATIN" in ud.name(c, ""))
    latin_ratio = latin / len(letters)
    if latin_ratio < config.english_min_latin_ratio:
        return False

    stop_hits = sum(1 for w in words if w in _EN_STOPWORDS)
    if stop_hits < config.english_min_stopword_count:
        return False
    if (stop_hits / len(words)) < config.english_min_stopword_ratio:
        return False

    return True


def _is_code_like(text: str, config: CleanCorpusConfig) -> bool:
    if _CODE_DROP_RE.search(text):
        return True
    words = _WORD_RE.findall(text.lower())
    code_hint_hits = sum(1 for w in words if w in _CODE_HINT_WORDS)
    code_symbols = sum(1 for c in text if c in "{}[]<>_=*/\\|`~$;")
    symbol_ratio = code_symbols / len(text) if text else 0.0
    if code_hint_hits >= config.code_keyword_hits_threshold:
        return True
    if symbol_ratio >= config.code_symbol_ratio_threshold and code_hint_hits >= 1:
        return True
    if text.count("{") + text.count("}") >= 2:
        return True
    return False


def analyze_corpora(
    input_files: list[Path],
    config: CorpusQualityConfig,
) -> dict[str, Any]:
    if config.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if config.max_lines_per_file < 0:
        raise ValueError("max_lines_per_file must be >= 0")
    if config.max_total_lines < 0:
        raise ValueError("max_total_lines must be >= 0")

    line_counts: Counter[str] = Counter()
    line_file_counts: dict[str, int] = {}
    line_last_file: dict[str, str] = {}

    files_seen = 0
    lines_seen = 0
    lines_nonempty = 0
    chars_seen = 0

    alpha_chars = 0
    digit_chars = 0
    ascii_chars = 0

    per_file: list[dict[str, Any]] = []
    truncated = False

    for file_path in input_files:
        files_seen += 1
        file_lines = 0
        file_nonempty = 0
        file_chars = 0

        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if config.max_lines_per_file and file_lines >= config.max_lines_per_file:
                    break
                if config.max_total_lines and lines_seen >= config.max_total_lines:
                    truncated = True
                    break

                file_lines += 1
                lines_seen += 1

                norm = normalize_whitespace(raw_line)
                if not norm:
                    continue

                file_nonempty += 1
                lines_nonempty += 1

                text_len = len(norm)
                file_chars += text_len
                chars_seen += text_len

                alpha_chars += sum(1 for c in norm if c.isalpha())
                digit_chars += sum(1 for c in norm if c.isdigit())
                ascii_chars += sum(1 for c in norm if ord(c) < 128)

                line_counts[norm] += 1
                key = str(file_path)
                if line_last_file.get(norm) != key:
                    line_file_counts[norm] = line_file_counts.get(norm, 0) + 1
                    line_last_file[norm] = key

            if truncated:
                break

        per_file.append(
            {
                "path": str(file_path),
                "lines_seen": file_lines,
                "nonempty_lines": file_nonempty,
                "chars_nonempty": file_chars,
                "avg_nonempty_line_chars": (
                    round(file_chars / file_nonempty, 2) if file_nonempty else 0.0
                ),
            }
        )

    unique_nonempty = len(line_counts)
    duplicate_nonempty = lines_nonempty - unique_nonempty

    def _line_row(line: str, count: int) -> dict[str, Any]:
        return {
            "count": count,
            "files": line_file_counts.get(line, 0),
            "chars": len(line),
            "line": line,
        }

    top_repeated_lines = [
        _line_row(line, count) for line, count in line_counts.most_common(config.top_k)
    ]

    boilerplate_candidates = [
        _line_row(line, count)
        for line, count in line_counts.items()
        if count >= config.boilerplate_min_occurrences
        and line_file_counts.get(line, 0) >= config.boilerplate_min_files
        and config.boilerplate_min_chars <= len(line) <= config.boilerplate_max_chars
    ]
    boilerplate_candidates.sort(key=lambda x: (-int(x["count"]), str(x["line"])))

    alpha_ratio = (alpha_chars / chars_seen) if chars_seen else 0.0
    digit_ratio = (digit_chars / chars_seen) if chars_seen else 0.0
    ascii_ratio = (ascii_chars / chars_seen) if chars_seen else 0.0

    return {
        "files_seen": files_seen,
        "lines_seen": lines_seen,
        "lines_nonempty": lines_nonempty,
        "unique_nonempty_lines": unique_nonempty,
        "duplicate_nonempty_lines": duplicate_nonempty,
        "chars_nonempty": chars_seen,
        "alpha_ratio": round(alpha_ratio, 5),
        "digit_ratio": round(digit_ratio, 5),
        "ascii_ratio": round(ascii_ratio, 5),
        "truncated": truncated,
        "config": {
            "top_k": config.top_k,
            "max_lines_per_file": config.max_lines_per_file,
            "max_total_lines": config.max_total_lines,
            "boilerplate_min_occurrences": config.boilerplate_min_occurrences,
            "boilerplate_min_files": config.boilerplate_min_files,
            "boilerplate_min_chars": config.boilerplate_min_chars,
            "boilerplate_max_chars": config.boilerplate_max_chars,
        },
        "top_repeated_lines": top_repeated_lines,
        "boilerplate_candidates": boilerplate_candidates,
        "per_file": per_file,
    }


def save_quality_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_boilerplate_lines_from_report(report_path: Path) -> set[str]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = payload.get("boilerplate_candidates", [])
    lines = {
        str(row["line"])
        for row in candidates
        if isinstance(row, dict) and "line" in row and str(row["line"]).strip()
    }
    return lines


def clean_corpora_batch(
    *,
    input_files: list[Path],
    output_dir: Path,
    config: CleanCorpusConfig,
    boilerplate_lines: set[str] | None = None,
) -> dict[str, Any]:
    if config.min_chars < 0:
        raise ValueError("min_chars must be >= 0")
    if config.max_chars < 0:
        raise ValueError("max_chars must be >= 0")
    if config.min_words < 0:
        raise ValueError("min_words must be >= 0")
    if not 0.0 <= config.min_alpha_ratio <= 1.0:
        raise ValueError("min_alpha_ratio must be in [0, 1]")
    if not 0.0 <= config.max_digit_ratio <= 1.0:
        raise ValueError("max_digit_ratio must be in [0, 1]")
    if not 0.0 <= config.max_symbol_ratio <= 1.0:
        raise ValueError("max_symbol_ratio must be in [0, 1]")
    if config.max_urls_per_line < 0:
        raise ValueError("max_urls_per_line must be >= 0")
    if config.repeated_token_run_threshold < 2:
        raise ValueError("repeated_token_run_threshold must be >= 2")
    if not 0.0 <= config.min_unique_token_ratio <= 1.0:
        raise ValueError("min_unique_token_ratio must be in [0, 1]")
    if config.max_lines_per_file < 0:
        raise ValueError("max_lines_per_file must be >= 0")
    if config.english_min_words < 0:
        raise ValueError("english_min_words must be >= 0")
    if config.english_min_stopword_count < 0:
        raise ValueError("english_min_stopword_count must be >= 0")
    if not 0.0 <= config.english_min_stopword_ratio <= 1.0:
        raise ValueError("english_min_stopword_ratio must be in [0, 1]")
    if not 0.0 <= config.english_min_latin_ratio <= 1.0:
        raise ValueError("english_min_latin_ratio must be in [0, 1]")
    if not 0.0 <= config.code_symbol_ratio_threshold <= 1.0:
        raise ValueError("code_symbol_ratio_threshold must be in [0, 1]")
    if config.code_keyword_hits_threshold < 0:
        raise ValueError("code_keyword_hits_threshold must be >= 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    boilerplate_lines = boilerplate_lines or set()

    global_seen: set[str] = set()
    files: list[dict[str, Any]] = []

    totals = {
        "input_lines": 0,
        "kept_lines": 0,
        "removed_empty": 0,
        "removed_too_short": 0,
        "removed_too_long": 0,
        "removed_too_few_words": 0,
        "removed_low_alpha": 0,
        "removed_high_digit": 0,
        "removed_high_symbol": 0,
        "removed_url_heavy": 0,
        "removed_repetitive_noise": 0,
        "removed_boilerplate": 0,
        "removed_non_english": 0,
        "removed_code_like": 0,
        "removed_duplicate_within": 0,
        "removed_duplicate_global": 0,
        "files_skipped_existing": 0,
    }

    for input_path in input_files:
        output_path = output_dir / f"{input_path.stem}{config.output_suffix}"
        if config.skip_existing and output_path.exists():
            totals["files_skipped_existing"] += 1
            files.append(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "status": "skipped_existing",
                }
            )
            continue

        within_seen: set[str] = set()
        kept = 0
        removed = {
            "empty": 0,
            "too_short": 0,
            "too_long": 0,
            "too_few_words": 0,
            "low_alpha": 0,
            "high_digit": 0,
            "high_symbol": 0,
            "url_heavy": 0,
            "repetitive_noise": 0,
            "boilerplate": 0,
            "non_english": 0,
            "code_like": 0,
            "duplicate_within": 0,
            "duplicate_global": 0,
        }

        line_no = 0
        with (
            input_path.open("r", encoding="utf-8") as src,
            output_path.open("w", encoding="utf-8") as dst,
        ):
            for raw_line in src:
                if config.max_lines_per_file and line_no >= config.max_lines_per_file:
                    break
                line_no += 1
                totals["input_lines"] += 1

                line = _strip_web_shell(raw_line, config)
                if not line:
                    removed["empty"] += 1
                    totals["removed_empty"] += 1
                    continue

                if len(line) < config.min_chars:
                    removed["too_short"] += 1
                    totals["removed_too_short"] += 1
                    continue

                if config.max_chars and len(line) > config.max_chars:
                    removed["too_long"] += 1
                    totals["removed_too_long"] += 1
                    continue

                if line in boilerplate_lines:
                    removed["boilerplate"] += 1
                    totals["removed_boilerplate"] += 1
                    continue

                if _digit_ratio(line) > config.max_digit_ratio:
                    removed["high_digit"] += 1
                    totals["removed_high_digit"] += 1
                    continue

                if _url_count(line) > config.max_urls_per_line:
                    removed["url_heavy"] += 1
                    totals["removed_url_heavy"] += 1
                    continue

                if _alpha_ratio(line) < config.min_alpha_ratio:
                    removed["low_alpha"] += 1
                    totals["removed_low_alpha"] += 1
                    continue

                if config.english_only and not _looks_english(line, config):
                    removed["non_english"] += 1
                    totals["removed_non_english"] += 1
                    continue

                if config.drop_code_like and _is_code_like(line, config):
                    removed["code_like"] += 1
                    totals["removed_code_like"] += 1
                    continue

                if _symbol_ratio(line) > config.max_symbol_ratio:
                    removed["high_symbol"] += 1
                    totals["removed_high_symbol"] += 1
                    continue

                if _is_repetitive_token_noise(line, config):
                    removed["repetitive_noise"] += 1
                    totals["removed_repetitive_noise"] += 1
                    continue

                if _word_count(line) < config.min_words:
                    removed["too_few_words"] += 1
                    totals["removed_too_few_words"] += 1
                    continue

                digest = hashlib.blake2b(line.encode("utf-8"), digest_size=16).hexdigest()
                if config.dedupe_within_file and digest in within_seen:
                    removed["duplicate_within"] += 1
                    totals["removed_duplicate_within"] += 1
                    continue
                within_seen.add(digest)

                if config.dedupe_global:
                    if digest in global_seen:
                        removed["duplicate_global"] += 1
                        totals["removed_duplicate_global"] += 1
                        continue
                    global_seen.add(digest)

                dst.write(f"{line}\n")
                kept += 1
                totals["kept_lines"] += 1

        files.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "status": "ok",
                "input_lines": line_no,
                "kept_lines": kept,
                "removed": removed,
            }
        )

    return {
        "files": files,
        "totals": totals,
        "config": {
            "min_chars": config.min_chars,
            "max_chars": config.max_chars,
            "min_words": config.min_words,
            "min_alpha_ratio": config.min_alpha_ratio,
            "max_digit_ratio": config.max_digit_ratio,
            "max_symbol_ratio": config.max_symbol_ratio,
            "max_urls_per_line": config.max_urls_per_line,
            "repeated_token_run_threshold": config.repeated_token_run_threshold,
            "min_unique_token_ratio": config.min_unique_token_ratio,
            "dedupe_within_file": config.dedupe_within_file,
            "dedupe_global": config.dedupe_global,
            "max_lines_per_file": config.max_lines_per_file,
            "skip_existing": config.skip_existing,
            "output_suffix": config.output_suffix,
            "boilerplate_lines": len(boilerplate_lines),
            "decode_html_entities": config.decode_html_entities,
            "strip_html_tags": config.strip_html_tags,
            "strip_site_suffixes": config.strip_site_suffixes,
            "strip_nav_phrases": config.strip_nav_phrases,
            "strip_stack_metadata": config.strip_stack_metadata,
            "collapse_repeated_prefix": config.collapse_repeated_prefix,
            "strip_inline_score_tokens": config.strip_inline_score_tokens,
            "english_only": config.english_only,
            "english_min_words": config.english_min_words,
            "english_min_stopword_ratio": config.english_min_stopword_ratio,
            "english_min_stopword_count": config.english_min_stopword_count,
            "english_min_latin_ratio": config.english_min_latin_ratio,
            "drop_code_like": config.drop_code_like,
            "code_symbol_ratio_threshold": config.code_symbol_ratio_threshold,
            "code_keyword_hits_threshold": config.code_keyword_hits_threshold,
        },
    }


def save_clean_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

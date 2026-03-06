"""Corpus quality analysis and cleaning helpers."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WS_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]{0,200}>")
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
    min_alpha_ratio: float = 0.20
    max_digit_ratio: float = 0.35
    dedupe_within_file: bool = True
    dedupe_global: bool = False
    max_lines_per_file: int = 0
    skip_existing: bool = True
    output_suffix: str = ".clean.txt"
    decode_html_entities: bool = True
    strip_html_tags: bool = True
    strip_site_suffixes: bool = True
    strip_nav_phrases: bool = True


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
    return normalize_whitespace(out)


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
    if not 0.0 <= config.min_alpha_ratio <= 1.0:
        raise ValueError("min_alpha_ratio must be in [0, 1]")
    if not 0.0 <= config.max_digit_ratio <= 1.0:
        raise ValueError("max_digit_ratio must be in [0, 1]")
    if config.max_lines_per_file < 0:
        raise ValueError("max_lines_per_file must be >= 0")

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
        "removed_low_alpha": 0,
        "removed_high_digit": 0,
        "removed_boilerplate": 0,
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
            "low_alpha": 0,
            "high_digit": 0,
            "boilerplate": 0,
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

                if _alpha_ratio(line) < config.min_alpha_ratio:
                    removed["low_alpha"] += 1
                    totals["removed_low_alpha"] += 1
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
            "min_alpha_ratio": config.min_alpha_ratio,
            "max_digit_ratio": config.max_digit_ratio,
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
        },
    }


def save_clean_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

"""Dataset risk auditing helpers for pretraining corpora."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

_TOXIC_TERMS = {
    "bastard",
    "bitch",
    "bullshit",
    "crap",
    "damn",
    "dumb",
    "dumbass",
    "fool",
    "hate",
    "hateful",
    "idiot",
    "kill",
    "killing",
    "moron",
    "nazi",
    "racist",
    "retard",
    "retarded",
    "sexist",
    "shit",
    "stupid",
    "trash",
}

_POLITICAL_TERMS = {
    "abortion",
    "biden",
    "capitalism",
    "communism",
    "conservative",
    "democrat",
    "democrats",
    "election",
    "elections",
    "fascism",
    "feminism",
    "immigration",
    "israel",
    "leftist",
    "liberal",
    "liberals",
    "palestine",
    "republican",
    "republicans",
    "russia",
    "socialism",
    "trump",
    "ukraine",
}

_REFUSAL_PHRASES = (
    "i am unable to help with that",
    "i can't help with that",
    "i cannot assist with that",
    "i cannot comply with that",
    "i cannot help with that",
    "i do not have the ability to help with that",
    "i must refuse",
    "i will not help with that",
    "i won't help with that",
    "as an ai language model",
    "i'm unable to help with that",
    "i'm sorry, but i can't",
)

_STEREOTYPE_PATTERNS = (
    re.compile(
        r"\ball\s+"
        r"(women|men|girls|boys|muslims|christians|jews|blacks|whites|immigrants|"
        r"gays|lesbians|trans(?:gender)?\s+people)\s+are\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b"
        r"(women|men|girls|boys|muslims|christians|jews|blacks|whites|immigrants|"
        r"gays|lesbians|trans(?:gender)?\s+people)\s+are\s+"
        r"(lazy|stupid|violent|inferior|criminal|evil|dangerous)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(race|gender|ethnicity|religion)\s+(makes|determines)\s+"
        r"(people|someone)\s+(inferior|superior)\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class DatasetRiskConfig:
    top_k: int = 25
    max_lines_per_file: int = 0
    max_total_lines: int = 0


def _normalize_line(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def _counter_top_k(counter: Counter[str], top_k: int) -> list[dict[str, int | str]]:
    return [{"term": term, "hits": hits} for term, hits in counter.most_common(top_k)]


def _rate_per_10k(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((count / total) * 10_000.0, 4)


def analyze_dataset_risk(
    input_files: list[Path],
    config: DatasetRiskConfig,
) -> dict[str, Any]:
    if config.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if config.max_lines_per_file < 0:
        raise ValueError("max_lines_per_file must be >= 0")
    if config.max_total_lines < 0:
        raise ValueError("max_total_lines must be >= 0")

    files_seen = 0
    lines_seen = 0
    lines_nonempty = 0
    truncated = False

    lines_with_toxicity = 0
    lines_with_stereotype = 0
    lines_with_political = 0
    lines_with_refusal = 0

    toxic_hits: Counter[str] = Counter()
    political_hits: Counter[str] = Counter()
    refusal_hits: Counter[str] = Counter()
    stereotype_hits: Counter[str] = Counter()

    for file_path in input_files:
        files_seen += 1
        file_lines = 0

        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if config.max_lines_per_file and file_lines >= config.max_lines_per_file:
                    break
                if config.max_total_lines and lines_seen >= config.max_total_lines:
                    truncated = True
                    break

                file_lines += 1
                lines_seen += 1

                line = raw_line.strip()
                if not line:
                    continue

                lines_nonempty += 1
                normalized = _normalize_line(line)
                words = _WORD_RE.findall(normalized)

                line_toxic = False
                line_political = False
                for word in words:
                    if word in _TOXIC_TERMS:
                        toxic_hits[word] += 1
                        line_toxic = True
                    if word in _POLITICAL_TERMS:
                        political_hits[word] += 1
                        line_political = True

                line_refusal = False
                for phrase in _REFUSAL_PHRASES:
                    if phrase in normalized:
                        refusal_hits[phrase] += 1
                        line_refusal = True

                line_stereotype = False
                for pattern in _STEREOTYPE_PATTERNS:
                    if pattern.search(normalized):
                        stereotype_hits[pattern.pattern] += 1
                        line_stereotype = True

                lines_with_toxicity += int(line_toxic)
                lines_with_political += int(line_political)
                lines_with_refusal += int(line_refusal)
                lines_with_stereotype += int(line_stereotype)

        if truncated:
            break

    return {
        "files_seen": files_seen,
        "lines_seen": lines_seen,
        "lines_nonempty": lines_nonempty,
        "truncated": truncated,
        "summary": {
            "lines_with_toxicity": lines_with_toxicity,
            "lines_with_stereotype": lines_with_stereotype,
            "lines_with_political": lines_with_political,
            "lines_with_refusal": lines_with_refusal,
            "toxicity_lines_per_10k": _rate_per_10k(lines_with_toxicity, lines_nonempty),
            "stereotype_lines_per_10k": _rate_per_10k(lines_with_stereotype, lines_nonempty),
            "political_lines_per_10k": _rate_per_10k(lines_with_political, lines_nonempty),
            "refusal_lines_per_10k": _rate_per_10k(lines_with_refusal, lines_nonempty),
        },
        "term_hits_top_k": {
            "toxic_terms": _counter_top_k(toxic_hits, config.top_k),
            "political_terms": _counter_top_k(political_hits, config.top_k),
            "refusal_phrases": _counter_top_k(refusal_hits, config.top_k),
            "stereotype_patterns": _counter_top_k(stereotype_hits, config.top_k),
        },
        "notes": [
            "This is a heuristic lexical audit, not a formal safety certification.",
            "Low score does not prove absence of bias; high score requires manual review.",
            "Factuality cannot be inferred from text alone; "
            "evaluate with downstream QA benchmarks.",
        ],
    }


def save_dataset_risk_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

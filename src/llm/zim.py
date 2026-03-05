"""Utilities for extracting training text from ZIM archives."""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable


class _HTMLToTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data.strip())

    def get_text(self) -> str:
        return " ".join(self._chunks)


def html_to_text(html: str) -> str:
    parser = _HTMLToTextParser()
    parser.feed(html)
    parser.close()
    return normalize_whitespace(parser.get_text())


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _blob_to_bytes(blob: Any) -> bytes:
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    try:
        return bytes(blob)
    except Exception:
        pass
    if hasattr(blob, "tobytes"):
        return blob.tobytes()
    raise TypeError(f"Unsupported blob type: {type(blob)}")


@dataclass
class ZimExtractConfig:
    zim_path: Path
    output_path: Path
    query: str = "*"
    max_articles: int = 10_000
    min_chars: int = 200
    max_chars: int = 0
    include_title: bool = True
    paths_file: Path | None = None
    batch_size: int = 128


def _iter_paths_from_file(paths_file: Path) -> Iterable[str]:
    for line in paths_file.read_text(encoding="utf-8").splitlines():
        path = line.strip()
        if path:
            yield path


def _iter_paths_from_search(archive: Any, query: str, batch_size: int) -> Iterable[str]:
    import libzim

    searcher = libzim.Searcher(archive)
    search_query = libzim.Query().set_query(query)
    search = searcher.search(search_query)

    offset = 0
    seen: set[str] = set()
    yielded = 0
    while True:
        results = search.getResults(offset, batch_size)
        paths = list(results)
        if not paths:
            break
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            yielded += 1
            yield path
        offset += len(paths)

    # Some ZIM files report a fulltext index but return no matches for all
    # normal queries. Fall back to suggestion index paths in that case.
    if yielded > 0:
        return

    suggester = libzim.SuggestionSearcher(archive)
    suggestions = suggester.suggest(query)
    offset = 0
    while True:
        results = suggestions.getResults(offset, batch_size)
        paths = list(results)
        if not paths:
            break
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            yield path
        offset += len(paths)


def extract_text_from_zim(config: ZimExtractConfig) -> dict[str, int]:
    """Extracts text/plain style corpus lines from a ZIM archive."""
    import libzim

    archive = libzim.Archive(config.zim_path)

    if config.paths_file is not None:
        path_iter = _iter_paths_from_file(config.paths_file)
    else:
        if not archive.has_fulltext_index:
            raise RuntimeError(
                "ZIM has no fulltext index. Provide --paths-file with article paths to extract."
            )
        path_iter = _iter_paths_from_search(archive, config.query, config.batch_size)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "seen_paths": 0,
        "written_articles": 0,
        "skipped_nontext": 0,
        "skipped_too_short": 0,
        "errors": 0,
    }

    with config.output_path.open("w", encoding="utf-8") as out:
        for path in path_iter:
            if config.max_articles and stats["written_articles"] >= config.max_articles:
                break

            stats["seen_paths"] += 1
            try:
                entry = archive.get_entry_by_path(path)
                is_redirect_attr = entry.is_redirect
                if callable(is_redirect_attr):
                    is_redirect = is_redirect_attr()
                else:
                    is_redirect = bool(is_redirect_attr)
                if is_redirect:
                    entry = entry.get_redirect_entry()

                item = entry.get_item()
                mimetype = str(item.mimetype).lower()
                if not (
                    mimetype.startswith("text/")
                    or "html" in mimetype
                    or "xhtml" in mimetype
                    or "xml" in mimetype
                ):
                    stats["skipped_nontext"] += 1
                    continue

                raw = _blob_to_bytes(item.content)
                text = raw.decode("utf-8", errors="ignore")
                if "html" in mimetype or "xhtml" in mimetype or "xml" in mimetype:
                    text = html_to_text(text)
                else:
                    text = normalize_whitespace(text)

                if len(text) < config.min_chars:
                    stats["skipped_too_short"] += 1
                    continue
                if config.max_chars > 0:
                    text = text[: config.max_chars]

                if config.include_title:
                    title = normalize_whitespace(str(item.title))
                    if title:
                        out.write(f"{title}\n")
                out.write(f"{text}\n\n")
                stats["written_articles"] += 1

            except Exception:
                stats["errors"] += 1
                continue

    return stats

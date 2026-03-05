"""Tokenizer utilities used in early project milestones."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SpecialTokens:
    unk: str = "<unk>"
    bos: str = "<bos>"
    eos: str = "<eos>"


class BasicCharTokenizer:
    """A simple character-level tokenizer for baseline experiments."""

    def __init__(self, stoi: dict[str, int], itos: dict[int, str]) -> None:
        self.stoi = stoi
        self.itos = itos
        self._unk_id = self.stoi["<unk>"]

    @classmethod
    def train(cls, text: str) -> "BasicCharTokenizer":
        specials = [SpecialTokens.unk, SpecialTokens.bos, SpecialTokens.eos]
        chars = sorted(set(text))
        vocab = specials + [c for c in chars if c not in specials]
        stoi = {token: idx for idx, token in enumerate(vocab)}
        itos = {idx: token for token, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @classmethod
    def train_from_files(
        cls,
        input_paths: Iterable[str | Path],
        *,
        max_chars_per_file: int = 0,
        chunk_size: int = 1_048_576,
    ) -> tuple["BasicCharTokenizer", dict[str, int]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if max_chars_per_file < 0:
            raise ValueError("max_chars_per_file must be >= 0")

        chars: set[str] = set()
        files_seen = 0
        chars_read = 0

        for raw_path in input_paths:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(path)
            files_seen += 1
            remaining = max_chars_per_file if max_chars_per_file > 0 else None
            with path.open("r", encoding="utf-8") as handle:
                while True:
                    if remaining is not None and remaining <= 0:
                        break
                    read_size = chunk_size if remaining is None else min(chunk_size, remaining)
                    chunk = handle.read(read_size)
                    if not chunk:
                        break
                    chars.update(chunk)
                    chars_read += len(chunk)
                    if remaining is not None:
                        remaining -= len(chunk)

        specials = [SpecialTokens.unk, SpecialTokens.bos, SpecialTokens.eos]
        vocab = specials + [c for c in sorted(chars) if c not in specials]
        stoi = {token: idx for idx, token in enumerate(vocab)}
        itos = {idx: token for token, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos), {
            "files_seen": files_seen,
            "chars_read": chars_read,
            "unique_chars": len(chars),
        }

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.stoi["<bos>"])
        ids.extend(self.stoi.get(ch, self._unk_id) for ch in text)
        if add_eos:
            ids.append(self.stoi["<eos>"])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        specials = {"<unk>", "<bos>", "<eos>"}
        tokens: list[str] = []
        for token_id in ids:
            token = self.itos.get(token_id, "<unk>")
            if skip_special_tokens and token in specials:
                continue
            tokens.append(token)
        return "".join(tokens)

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stoi": self.stoi}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, input_path: str | Path) -> "BasicCharTokenizer":
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
        itos = {idx: token for token, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

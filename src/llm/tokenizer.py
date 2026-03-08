"""Tokenizer utilities for BPE workflows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class SpecialTokens:
    unk: str = "<unk>"
    bos: str = "<bos>"
    eos: str = "<eos>"


class TokenizerLike(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def bos_id(self) -> int | None: ...

    @property
    def eos_id(self) -> int | None: ...

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...

    def save(self, output_path: str | Path) -> None: ...


class BPETokenizer:
    """Byte-level BPE tokenizer backed by Hugging Face `tokenizers`."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._unk_id = self._tokenizer.token_to_id(SpecialTokens.unk)
        self._bos_id = self._tokenizer.token_to_id(SpecialTokens.bos)
        self._eos_id = self._tokenizer.token_to_id(SpecialTokens.eos)

    @classmethod
    def train_from_iterator(
        cls,
        text_iter: Iterable[str],
        *,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")

        tokenizers = _require_tokenizers()
        tok = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=SpecialTokens.unk))
        tok.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = tokenizers.decoders.ByteLevel()
        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[SpecialTokens.unk, SpecialTokens.bos, SpecialTokens.eos],
            initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train_from_iterator(text_iter, trainer=trainer)
        return cls(tok)

    @classmethod
    def train_from_files(
        cls,
        input_paths: Iterable[str | Path],
        *,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        max_chars_per_file: int = 0,
        chunk_size: int = 1_048_576,
    ) -> tuple["BPETokenizer", dict[str, int]]:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        if max_chars_per_file < 0:
            raise ValueError("max_chars_per_file must be >= 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        files_seen = 0
        chars_read = 0

        def _iter_text() -> Iterable[str]:
            nonlocal files_seen, chars_read
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
                        chars_read += len(chunk)
                        if remaining is not None:
                            remaining -= len(chunk)
                        yield chunk

        wrapped = cls.train_from_iterator(
            _iter_text(),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
        return wrapped, {
            "files_seen": files_seen,
            "chars_read": chars_read,
            "unique_chars": 0,
        }

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    @property
    def bos_id(self) -> int | None:
        return self._bos_id

    @property
    def eos_id(self) -> int | None:
        return self._eos_id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        encoded = self._tokenizer.encode(text)
        ids = [int(tok) for tok in encoded.ids]
        if add_bos and self._bos_id is not None:
            ids = [self._bos_id, *ids]
        if add_eos and self._eos_id is not None:
            ids = [*ids, self._eos_id]
        return ids

    def encode_batch(
        self,
        texts: list[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        encoded_batch = self._tokenizer.encode_batch(texts)
        output: list[list[int]] = []
        for encoded in encoded_batch:
            ids = [int(tok) for tok in encoded.ids]
            if add_bos and self._bos_id is not None:
                ids = [self._bos_id, *ids]
            if add_eos and self._eos_id is not None:
                ids = [*ids, self._eos_id]
            output.append(ids)
        return output

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special_candidates = (self._unk_id, self._bos_id, self._eos_id)
            specials = {tok_id for tok_id in special_candidates if tok_id is not None}
            ids = [tok_id for tok_id in ids if tok_id not in specials]
        return str(self._tokenizer.decode(ids, skip_special_tokens=False))

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, input_path: str | Path) -> "BPETokenizer":
        tokenizers = _require_tokenizers()
        tok = tokenizers.Tokenizer.from_file(str(input_path))
        return cls(tok)


TOKENIZER_CONTRACT_VERSION = "bytelevel-bpe-v1"


def _require_tokenizers() -> Any:
    try:
        import tokenizers
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "BPE tokenizer support requires the 'tokenizers' package. "
            "Install with training extras: pip install -e '.[train]'"
        ) from exc
    return tokenizers


def tokenizer_fingerprint(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _component_signature(component: Any) -> str:
    if not isinstance(component, dict):
        return "unknown"
    comp_type = str(component.get("type", ""))
    if not comp_type:
        return "unknown"
    if comp_type != "Sequence":
        return comp_type
    nested = component.get("pretokenizers")
    if not isinstance(nested, list):
        nested = component.get("decoders")
    if not isinstance(nested, list):
        return comp_type
    parts = [str(item.get("type", "unknown")) for item in nested if isinstance(item, dict)]
    return f"{comp_type}({'+'.join(parts)})"


def tokenizer_contract(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid tokenizer payload: {path}")

    model = payload.get("model")
    model_type = "unknown"
    if isinstance(model, dict):
        model_type = str(model.get("type", "unknown")).upper()

    special_tokens = [SpecialTokens.unk, SpecialTokens.bos, SpecialTokens.eos]
    added_tokens = payload.get("added_tokens", [])
    if isinstance(added_tokens, list):
        available = {
            str(row.get("content"))
            for row in added_tokens
            if isinstance(row, dict) and row.get("content") is not None
        }
        special_tokens = [token for token in special_tokens if token in available]

    return {
        "version": TOKENIZER_CONTRACT_VERSION,
        "model_type": model_type,
        "pre_tokenizer": _component_signature(payload.get("pre_tokenizer")),
        "decoder": _component_signature(payload.get("decoder")),
        "special_tokens": special_tokens,
    }


def tokenizer_contract_fingerprint(path: str | Path) -> str:
    contract = tokenizer_contract(path)
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_tokenizer(path: str | Path) -> TokenizerLike:
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid tokenizer payload: {input_path}")

    model = payload.get("model")
    if isinstance(model, dict):
        model_type = str(model.get("type", "")).upper()
        if model_type == "BPE":
            return BPETokenizer.load(input_path)

    raise ValueError(f"unsupported tokenizer format: {input_path}")

"""Integrity checks for token shard datasets."""

from __future__ import annotations

import json
from array import array
from pathlib import Path
from typing import Any

from llm.tokenizer import load_tokenizer

_TOKEN_ARRAY_TYPES = {"uint16": "H", "uint32": "I"}


def _expected_token_dtype(vocab_size: int) -> str:
    if vocab_size <= 65535:
        return "uint16"
    return "uint32"


def _iter_manifest_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.name != "manifest.json":
            raise ValueError(f"Expected manifest.json file, got: {path}")
        return [path]

    if not path.exists() or not path.is_dir():
        raise ValueError(f"Path does not exist or is not a directory: {path}")

    manifests = sorted(path.rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"No manifest.json files found under: {path}")
    return manifests


def _append_source_zim_check(
    warnings: list[str], errors: list[str], input_path: str, raw_zim_dir: Path, strict_source: bool
) -> None:
    source_zim = raw_zim_dir / f"{Path(input_path).stem}.zim"
    if not source_zim.exists():
        message = f"source_zim_missing:{source_zim}"
        if strict_source:
            errors.append(message)
        else:
            warnings.append(message)
        return

    try:
        import libzim

        archive = libzim.Archive(source_zim)
        entry = archive.get_random_entry()
        is_redirect_attr = entry.is_redirect
        is_redirect = is_redirect_attr() if callable(is_redirect_attr) else bool(is_redirect_attr)
        if is_redirect:
            entry = entry.get_redirect_entry()
        item = entry.get_item()
        _ = len(bytes(item.content))
    except Exception as exc:
        errors.append(f"source_zim_invalid:{source_zim}:{type(exc).__name__}:{exc}")


def _check_shard_range(
    shard_path: Path,
    array_type: str,
    expected_tokens: int,
    vocab_size: int,
    chunk_tokens: int,
) -> tuple[int, str | None]:
    checked = 0
    try:
        with shard_path.open("rb") as handle:
            remaining = expected_tokens
            while remaining > 0:
                take = min(remaining, chunk_tokens)
                buf = array(array_type)
                buf.fromfile(handle, take)
                remaining -= take
                checked += take
                if buf and max(buf) >= vocab_size:
                    return checked, f"token_out_of_range:{shard_path.name}"
    except EOFError:
        return checked, f"unexpected_eof:{shard_path.name}"
    except Exception as exc:
        return checked, f"range_check_error:{shard_path.name}:{type(exc).__name__}:{exc}"
    return checked, None


def verify_shard_manifest(
    manifest_path: Path,
    check_token_ranges: bool = True,
    chunk_tokens: int = 1_000_000,
    raw_zim_dir: Path | None = None,
    strict_source: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    files_checked = 0
    tokens_checked = 0

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_dir = manifest_path.parent

    tokenizer_path = Path(manifest.get("tokenizer_path", ""))
    if not tokenizer_path.exists():
        errors.append(f"tokenizer_missing:{tokenizer_path}")
        tokenizer = None
        vocab_size = int(manifest.get("tokenizer_vocab_size", 0))
    else:
        try:
            tokenizer = load_tokenizer(tokenizer_path)
            vocab_size = tokenizer.vocab_size
        except Exception as exc:
            errors.append(f"tokenizer_load_error:{tokenizer_path}:{type(exc).__name__}:{exc}")
            tokenizer = None
            vocab_size = int(manifest.get("tokenizer_vocab_size", 0))

    manifest_vocab_size = int(manifest.get("tokenizer_vocab_size", 0))
    if vocab_size and manifest_vocab_size and vocab_size != manifest_vocab_size:
        errors.append(f"vocab_size_mismatch:manifest={manifest_vocab_size},tokenizer={vocab_size}")

    token_dtype = str(manifest.get("token_dtype", ""))
    if token_dtype not in _TOKEN_ARRAY_TYPES:
        errors.append(f"unsupported_token_dtype:{token_dtype}")
        array_type = "H"
        item_size = 2
    else:
        array_type = _TOKEN_ARRAY_TYPES[token_dtype]
        item_size = array(array_type).itemsize

    if vocab_size:
        expected_dtype = _expected_token_dtype(vocab_size)
        if expected_dtype != token_dtype:
            errors.append(f"token_dtype_mismatch:expected={expected_dtype},manifest={token_dtype}")

    shard_size_tokens = int(manifest.get("shard_size_tokens", 0))
    expected_shards: set[str] = set()

    for split in ("train", "val"):
        split_data = manifest.get(split, {})
        shard_list = list(split_data.get("shards", []))
        declared_total = int(split_data.get("total_tokens", 0))
        observed_total = 0

        for idx, shard in enumerate(shard_list):
            shard_name = str(shard.get("path", ""))
            expected_tokens = int(shard.get("tokens", 0))
            shard_path = dataset_dir / shard_name
            expected_shards.add(shard_name)

            if expected_tokens <= 0:
                errors.append(f"non_positive_tokens:{split}:{shard_name}:{expected_tokens}")
                continue

            if (
                idx < len(shard_list) - 1
                and shard_size_tokens > 0
                and expected_tokens != shard_size_tokens
            ):
                errors.append(f"non_full_nonfinal_shard:{split}:{shard_name}:{expected_tokens}")
            if shard_size_tokens > 0 and expected_tokens > shard_size_tokens:
                errors.append(f"oversized_shard:{split}:{shard_name}:{expected_tokens}")

            if not shard_path.exists():
                errors.append(f"missing_shard:{shard_name}")
                continue

            files_checked += 1
            actual_size = shard_path.stat().st_size
            expected_size = expected_tokens * item_size
            if actual_size != expected_size:
                errors.append(
                    f"size_mismatch:{shard_name}:expected={expected_size}:actual={actual_size}"
                )
                continue

            observed_total += expected_tokens
            if check_token_ranges and vocab_size > 0:
                checked, range_error = _check_shard_range(
                    shard_path=shard_path,
                    array_type=array_type,
                    expected_tokens=expected_tokens,
                    vocab_size=vocab_size,
                    chunk_tokens=chunk_tokens,
                )
                tokens_checked += checked
                if range_error:
                    errors.append(range_error)

        if observed_total != declared_total:
            errors.append(f"total_token_mismatch:{split}:declared={declared_total}:observed={observed_total}")

    extra_bin_files = {
        p.name for p in dataset_dir.glob("*.bin") if p.name not in expected_shards
    }
    for extra in sorted(extra_bin_files):
        warnings.append(f"extra_shard_file:{extra}")

    if raw_zim_dir is not None:
        input_path = str(manifest.get("input_path", ""))
        if input_path:
            _append_source_zim_check(
                warnings=warnings,
                errors=errors,
                input_path=input_path,
                raw_zim_dir=raw_zim_dir,
                strict_source=strict_source,
            )
        else:
            warnings.append("manifest_missing_input_path")

    return {
        "manifest_path": str(manifest_path),
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "files_checked": files_checked,
        "tokens_checked": tokens_checked,
    }


def verify_shards(
    path: Path,
    check_token_ranges: bool = True,
    chunk_tokens: int = 1_000_000,
    raw_zim_dir: Path | None = None,
    strict_source: bool = False,
) -> list[dict[str, Any]]:
    manifests = _iter_manifest_paths(path)
    return [
        verify_shard_manifest(
            manifest_path=manifest_path,
            check_token_ranges=check_token_ranges,
            chunk_tokens=chunk_tokens,
            raw_zim_dir=raw_zim_dir,
            strict_source=strict_source,
        )
        for manifest_path in manifests
    ]

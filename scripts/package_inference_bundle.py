#!/usr/bin/env python3
"""Package a portable local inference bundle with checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _resolve_tokenizer(checkpoint: dict[str, Any], override: str | None) -> Path:
    if override:
        path = Path(override)
    else:
        raw = checkpoint.get("tokenizer_path")
        if not isinstance(raw, str):
            raise ValueError("checkpoint missing tokenizer_path; pass --tokenizer")
        path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _default_output_dir(checkpoint_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = checkpoint_path.stem
    return Path("artifacts/release") / f"inference-{name}-{stamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/fineweb-350bt-bpe-v2-run1/best.pt",
        help="Checkpoint path to package (default: best.pt in main run)",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer override path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Bundle output directory",
    )
    parser.add_argument(
        "--model-id",
        default="local/llm-from-scratch",
        help="Model identifier label written to manifest/README",
    )
    parser.add_argument(
        "--create-tar",
        action="store_true",
        help="Also create <output-dir>.tar.gz",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer_path = _resolve_tokenizer(checkpoint, args.tokenizer)

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_checkpoint = output_dir / "checkpoint.pt"
    bundle_tokenizer = output_dir / "tokenizer.json"
    bundle_manifest = output_dir / "bundle_manifest.json"
    bundle_readme = output_dir / "README.md"

    _copy_file(checkpoint_path, bundle_checkpoint)
    _copy_file(tokenizer_path, bundle_tokenizer)

    # Optional sidecar artifacts from the same checkpoint directory.
    sidecars: list[tuple[str, Path]] = []
    ckpt_dir = checkpoint_path.parent
    for src_name, dst_name in [
        ("best.safetensors", "model.safetensors"),
        ("best_ema.safetensors", "model_ema.safetensors"),
        ("last.safetensors", "model_last.safetensors"),
        ("last_ema.safetensors", "model_last_ema.safetensors"),
        ("run_config.json", "run_config.json"),
    ]:
        src = ckpt_dir / src_name
        if src.exists():
            dst = output_dir / dst_name
            _copy_file(src, dst)
            sidecars.append((dst_name, dst))

    files = {
        "checkpoint.pt": bundle_checkpoint,
        "tokenizer.json": bundle_tokenizer,
    }
    for name, path in sidecars:
        files[name] = path

    file_rows: list[dict[str, Any]] = []
    for rel_name, abs_path in files.items():
        file_rows.append(
            {
                "name": rel_name,
                "bytes": abs_path.stat().st_size,
                "sha256": _sha256(abs_path),
            }
        )

    manifest = {
        "model_id": args.model_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint_path),
        "source_tokenizer": str(tokenizer_path),
        "step": checkpoint.get("step"),
        "tokenizer_hash": checkpoint.get("tokenizer_hash"),
        "tokenizer_contract_hash": checkpoint.get("tokenizer_contract_hash"),
        "model_config": checkpoint.get("model_config"),
        "files": file_rows,
    }
    bundle_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    bundle_readme.write_text(
        "\n".join(
            [
                f"# Inference Bundle: {args.model_id}",
                "",
                "This bundle contains a local checkpoint, tokenizer, and reproducibility metadata.",
                "",
                "## Files",
                "- `checkpoint.pt`: primary checkpoint payload",
                "- `tokenizer.json`: tokenizer used by checkpoint",
                "- `bundle_manifest.json`: checksums and metadata",
                "",
                "## Run",
                "```bash",
                "PYTHONPATH=src .venv/bin/python -m llm.inference_server \\",
                "  --checkpoint checkpoint.pt \\",
                "  --tokenizer tokenizer.json \\",
                f"  --model-id {args.model_id}",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    tar_path: Path | None = None
    if args.create_tar:
        tar_path = output_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)

    print(f"bundle_dir={output_dir}")
    print(f"bundle_manifest={bundle_manifest}")
    if tar_path is not None:
        print(f"bundle_tar={tar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

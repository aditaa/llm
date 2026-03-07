#!/usr/bin/env python3
"""Prepare a portable model bundle and optionally publish to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _resolve_tokenizer_path(checkpoint: dict[str, Any], tokenizer_override: str | None) -> Path:
    if tokenizer_override:
        path = Path(tokenizer_override)
    else:
        raw = checkpoint.get("tokenizer_path")
        if not isinstance(raw, str):
            raise ValueError("checkpoint missing tokenizer_path; pass --tokenizer")
        path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _default_card(repo_id: str, manifest: dict[str, Any]) -> str:
    model_cfg = manifest.get("model_config", {})
    train_cfg = manifest.get("train_config", {})
    return f"""---
license: other
language:
- en
pipeline_tag: text-generation
tags:
- llm
- from-scratch
- bpe
---

# {repo_id}

Decoder-only LLM checkpoint exported from the `aditaa/llm` training stack.

## Architecture
- n_layers: {model_cfg.get("n_layers")}
- n_heads: {model_cfg.get("n_heads")}
- d_model: {model_cfg.get("d_model")}
- max_seq_len: {model_cfg.get("max_seq_len")}
- architecture: {model_cfg.get("architecture")}

## Training Snapshot
- step: {manifest.get("step")}
- shards_path: {train_cfg.get("shards_path")}
- tokenizer_hash: {manifest.get("tokenizer_hash")}
- exported_at_utc: {manifest.get("exported_at_utc")}

## Files
- `checkpoint.pt`: training checkpoint payload (`model_state`, optimizer state, configs)
- `model.safetensors`: optional model weights-only export (when enabled)
- `tokenizer.json`: tokenizer payload used by checkpoint (char or BPE)
- `release_manifest.json`: metadata for reproducibility

## Usage (this repository runtime)
Use `scripts/hf_download_model.sh` then run:

```bash
PYTHONPATH=src .venv/bin/python -m llm.inference_server \\
  --checkpoint /path/to/model/checkpoint.pt \\
  --tokenizer /path/to/model/tokenizer.json \\
  --model-id {repo_id}
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. aditaa/llm-from-scratch-v1")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt file path")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer JSON path override")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Bundle output directory (default: artifacts/release/<repo-name>)",
    )
    parser.add_argument("--private", action="store_true", help="Create private HF repo")
    parser.add_argument("--push", action="store_true", help="Upload bundle to HF model repo")
    parser.add_argument("--token", default=None, help="HF token override (or HF_TOKEN env var)")
    parser.add_argument(
        "--include-safetensors",
        action="store_true",
        help="Also export model_state as model.safetensors (weights only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer_path = _resolve_tokenizer_path(checkpoint, args.tokenizer)

    repo_name = args.repo_id.split("/")[-1]
    output_dir = Path(args.output_dir or f"artifacts/release/{repo_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_checkpoint = output_dir / "checkpoint.pt"
    bundle_tokenizer = output_dir / "tokenizer.json"
    bundle_manifest = output_dir / "release_manifest.json"
    bundle_card = output_dir / "README.md"
    bundle_safetensors = output_dir / "model.safetensors"

    shutil.copy2(checkpoint_path, bundle_checkpoint)
    shutil.copy2(tokenizer_path, bundle_tokenizer)

    safetensors_written = False
    if args.include_safetensors:
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "--include-safetensors requires the 'safetensors' package. "
                "Install with training extras."
            ) from exc
        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("checkpoint missing model_state for safetensors export")
        tensor_state = {
            key: value.detach().cpu().contiguous()
            for key, value in model_state.items()
            if hasattr(value, "detach") and hasattr(value, "cpu")
        }
        save_file(tensor_state, str(bundle_safetensors))
        safetensors_written = True

    manifest = {
        "repo_id": args.repo_id,
        "source_checkpoint": str(checkpoint_path),
        "source_tokenizer": str(tokenizer_path),
        "step": checkpoint.get("step"),
        "tokenizer_hash": checkpoint.get("tokenizer_hash"),
        "model_config": checkpoint.get("model_config"),
        "train_config": checkpoint.get("train_config"),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "checkpoint": bundle_checkpoint.name,
            "tokenizer": bundle_tokenizer.name,
            "model_card": bundle_card.name,
        },
    }
    if safetensors_written:
        manifest["files"]["safetensors"] = bundle_safetensors.name
    bundle_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle_card.write_text(_default_card(args.repo_id, manifest), encoding="utf-8")

    print(f"bundle_dir={output_dir}")
    print(f"bundle_checkpoint={bundle_checkpoint}")
    print(f"bundle_tokenizer={bundle_tokenizer}")
    print(f"bundle_manifest={bundle_manifest}")

    if not args.push:
        return 0

    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --push. Install with: pip install huggingface_hub"
        ) from exc

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF token is required for --push (use --token or HF_TOKEN)")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Upload model bundle",
    )
    print(f"pushed_repo=https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

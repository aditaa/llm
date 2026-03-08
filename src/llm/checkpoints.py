"""Checkpoint averaging helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class AverageCheckpointsConfig:
    checkpoint_paths: list[Path]
    output_path: Path
    state_key: str = "model_state"
    export_safetensors: bool = False


def _read_state_dict(path: Path, state_key: str) -> tuple[dict[str, Any], dict[str, Tensor]]:
    payload = torch.load(path, map_location="cpu")
    state_raw = payload.get(state_key)
    if not isinstance(state_raw, dict):
        raise ValueError(f"checkpoint missing {state_key}: {path}")

    state: dict[str, Tensor] = {}
    for key, value in state_raw.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError(f"invalid tensor state entry in {path}: {key}")
        state[key] = value
    if not state:
        raise ValueError(f"empty {state_key} in checkpoint: {path}")
    return payload, state


def _average_state_dicts(states: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    if len(states) < 2:
        raise ValueError("at least two state dicts are required")

    keys = set(states[0].keys())
    if not keys:
        raise ValueError("state dict is empty")

    ref_state = states[0]
    float_sums: dict[str, Tensor] = {}
    non_float_ref: dict[str, Tensor] = {}

    for idx, state in enumerate(states):
        if set(state.keys()) != keys:
            raise ValueError("state dict keys mismatch across checkpoints")

        for key in keys:
            tensor = state[key]
            ref = ref_state[key]
            if tensor.shape != ref.shape:
                raise ValueError(f"shape mismatch for key '{key}'")
            if tensor.dtype != ref.dtype:
                raise ValueError(f"dtype mismatch for key '{key}'")

            if torch.is_floating_point(ref):
                if idx == 0:
                    float_sums[key] = tensor.to(dtype=torch.float32).clone()
                else:
                    float_sums[key].add_(tensor.to(dtype=torch.float32))
            else:
                if idx == 0:
                    non_float_ref[key] = tensor.clone()
                elif not torch.equal(non_float_ref[key], tensor):
                    raise ValueError(f"non-floating tensor mismatch for key '{key}'")

    averaged: dict[str, Tensor] = {}
    denom = float(len(states))
    for key in keys:
        ref = ref_state[key]
        if torch.is_floating_point(ref):
            averaged[key] = (float_sums[key] / denom).to(dtype=ref.dtype)
        else:
            averaged[key] = non_float_ref[key]
    return averaged


def run_checkpoint_average(config: AverageCheckpointsConfig) -> dict[str, Any]:
    if len(config.checkpoint_paths) < 2:
        raise ValueError("checkpoint_paths must include at least two entries")
    if config.state_key not in {"model_state", "ema_state"}:
        raise ValueError("state_key must be one of: model_state, ema_state")

    payloads: list[dict[str, Any]] = []
    states: list[dict[str, Tensor]] = []
    for path in config.checkpoint_paths:
        payload, state = _read_state_dict(path, config.state_key)
        payloads.append(payload)
        states.append(state)

    averaged_state = _average_state_dicts(states)

    output_payload = dict(payloads[-1])
    output_payload["model_state"] = averaged_state
    output_payload["optimizer_state"] = {}
    output_payload["scaler_state"] = None
    output_payload["averaged_checkpoint"] = True
    output_payload["averaged_count"] = len(config.checkpoint_paths)
    output_payload["averaged_from"] = [str(path) for path in config.checkpoint_paths]
    output_payload["averaged_state_key"] = config.state_key

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, config.output_path)

    output_safetensors: str | None = None
    if config.export_safetensors:
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "export_safetensors requires the 'safetensors' package. "
                "Install with training extras."
            ) from exc
        safe_state = {
            key: tensor.detach().cpu().contiguous()
            for key, tensor in averaged_state.items()
            if isinstance(tensor, Tensor)
        }
        safe_path = config.output_path.with_suffix(".safetensors")
        save_file(safe_state, str(safe_path))
        output_safetensors = str(safe_path)

    return {
        "output_checkpoint": str(config.output_path),
        "state_key": config.state_key,
        "averaged_count": len(config.checkpoint_paths),
        "output_safetensors": output_safetensors,
    }

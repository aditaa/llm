"""Text generation helpers for checkpointed GPT models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from llm.model import GPTModel, ModelConfig
from llm.tokenizer import load_tokenizer


@dataclass
class GenerateConfig:
    checkpoint_path: Path
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int = 0
    device: str = "auto"
    seed: int = 42
    stop_on_eos: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        return payload


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def _sample_next_token(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        top_values, _ = torch.topk(logits, k=k)
        cutoff = top_values[:, -1:].clone()
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def run_generation(config: GenerateConfig) -> dict[str, Any]:
    if config.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if config.top_k < 0:
        raise ValueError("top_k must be >= 0")

    device = _resolve_device(config.device)
    torch.manual_seed(config.seed)

    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model_cfg_raw = checkpoint.get("model_config")
    if not isinstance(model_cfg_raw, dict):
        raise ValueError("checkpoint missing model_config")
    model_config = ModelConfig(**model_cfg_raw)

    tokenizer_path = checkpoint.get("tokenizer_path")
    if not isinstance(tokenizer_path, str):
        raise ValueError("checkpoint missing tokenizer_path")
    tokenizer = load_tokenizer(tokenizer_path)

    model = GPTModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    token_ids = tokenizer.encode(config.prompt)
    if not token_ids:
        bos_id = tokenizer.bos_id if tokenizer.bos_id is not None else 0
        token_ids = [bos_id]

    eos_id = tokenizer.eos_id if tokenizer.eos_id is not None else -1

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            input_ids = input_ids[:, -model_config.max_seq_len :]
            logits, _ = model(input_ids)
            next_token = _sample_next_token(
                logits[:, -1, :],
                temperature=config.temperature,
                top_k=config.top_k,
            )
            next_id = int(next_token.item())
            token_ids.append(next_id)
            if config.stop_on_eos and next_id == eos_id:
                break

    output_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return {
        "checkpoint_path": str(config.checkpoint_path),
        "tokenizer_path": tokenizer_path,
        "device": str(device),
        "seed": config.seed,
        "prompt": config.prompt,
        "output_text": output_text,
        "token_count": len(token_ids),
    }

"""OpenAI-compatible inference server for local checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm.generate import _resolve_device
from llm.model import GPTModel, model_config_from_dict
from llm.tokenizer import TokenizerLike, load_tokenizer


@dataclass
class RuntimeState:
    model: GPTModel
    tokenizer: TokenizerLike
    device: torch.device
    model_id: str
    eos_id: int
    max_seq_len: int
    lock: asyncio.Lock


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=200, ge=1, le=4096)
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int = Field(default=0, ge=0, le=4096)
    stop: str | list[str] | None = None
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=200, ge=1, le=4096)
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int = Field(default=0, ge=0, le=4096)
    stop: str | list[str] | None = None
    stream: bool = False


def _sample_next_token(logits: torch.Tensor, *, temperature: float, top_k: int) -> torch.Tensor:
    scaled = logits / temperature
    if top_k > 0:
        k = min(top_k, scaled.shape[-1])
        top_values, _ = torch.topk(scaled, k=k)
        cutoff = top_values[:, -1:].clone()
        scaled = torch.where(scaled < cutoff, torch.full_like(scaled, float("-inf")), scaled)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _normalize_stop(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [item for item in stop if item]


def _apply_stop_sequences(text: str, stops: list[str]) -> tuple[str, bool]:
    if not stops:
        return text, False
    hits = [text.find(stop) for stop in stops if stop and stop in text]
    if not hits:
        return text, False
    cut = min(hits)
    return text[:cut], True


def _generate_completion(
    state: RuntimeState,
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    stop_sequences: list[str],
) -> tuple[str, str, int, int]:
    token_ids = state.tokenizer.encode(prompt)
    if not token_ids:
        bos_id = state.tokenizer.bos_id if state.tokenizer.bos_id is not None else 0
        token_ids = [bos_id]

    completion_text = ""
    generated_tokens = 0
    finish_reason = "length"

    with torch.no_grad():
        for _ in range(max_tokens):
            ctx = token_ids[-state.max_seq_len :]
            input_ids = torch.tensor([ctx], dtype=torch.long, device=state.device)
            logits, _ = state.model(input_ids)
            next_token = _sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
            next_id = int(next_token.item())
            token_ids.append(next_id)
            generated_tokens += 1

            if next_id == state.eos_id:
                finish_reason = "stop"
                break

            completion_text += state.tokenizer.decode([next_id], skip_special_tokens=True)
            clipped, hit_stop = _apply_stop_sequences(completion_text, stop_sequences)
            if hit_stop:
                completion_text = clipped
                finish_reason = "stop"
                break

    prompt_tokens = len(state.tokenizer.encode(prompt))
    completion_tokens = len(state.tokenizer.encode(completion_text))
    return completion_text, finish_reason, prompt_tokens, completion_tokens


def _format_chat_prompt(messages: list[ChatMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.role.strip().lower() or "user"
        lines.append(f"{role}: {message.content}")
    lines.append("assistant:")
    return "\n".join(lines)


def _load_state(
    *,
    checkpoint_path: Path,
    tokenizer_path: Path | None,
    device_arg: str,
    model_id: str,
) -> RuntimeState:
    device = _resolve_device(device_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_cfg_raw = checkpoint.get("model_config")
    if not isinstance(model_cfg_raw, dict):
        raise ValueError("checkpoint missing model_config")
    model_config = model_config_from_dict(model_cfg_raw)

    if tokenizer_path is None:
        tok_from_ckpt = checkpoint.get("tokenizer_path")
        if not isinstance(tok_from_ckpt, str):
            raise ValueError("checkpoint missing tokenizer_path; pass --tokenizer")
        tokenizer_path = Path(tok_from_ckpt)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

    tokenizer = load_tokenizer(tokenizer_path)
    model = GPTModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    eos_id = tokenizer.eos_id if tokenizer.eos_id is not None else -1
    return RuntimeState(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_id=model_id,
        eos_id=eos_id,
        max_seq_len=model_config.max_seq_len,
        lock=asyncio.Lock(),
    )


def create_app(state: RuntimeState) -> FastAPI:
    app = FastAPI(title="LLM From Scratch Inference API", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "model": state.model_id, "device": str(state.device)}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": state.model_id,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        if req.model and req.model != state.model_id:
            raise HTTPException(status_code=404, detail=f"model not found: {req.model}")

        async with state.lock:
            start = time.time()
            completion, finish_reason, prompt_tokens, completion_tokens = _generate_completion(
                state,
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                stop_sequences=_normalize_stop(req.stop),
            )
            elapsed = time.time() - start

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": state.model_id,
            "choices": [
                {
                    "index": 0,
                    "text": completion,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "meta": {"elapsed_seconds": round(elapsed, 4)},
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not supported")
        if req.model and req.model != state.model_id:
            raise HTTPException(status_code=404, detail=f"model not found: {req.model}")
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        prompt = _format_chat_prompt(req.messages)
        async with state.lock:
            start = time.time()
            completion, finish_reason, prompt_tokens, completion_tokens = _generate_completion(
                state,
                prompt=prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                stop_sequences=_normalize_stop(req.stop),
            )
            elapsed = time.time() - start

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": state.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "meta": {"elapsed_seconds": round(elapsed, 4)},
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve local checkpoint with OpenAI-compatible API."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer JSON path override")
    parser.add_argument(
        "--model-id", default="llm-from-scratch-local", help="Model id shown in API"
    )
    parser.add_argument(
        "--device", default="auto", help="Torch device (auto, cpu, cuda, cuda:0...)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn worker count")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = _load_state(
        checkpoint_path=Path(args.checkpoint),
        tokenizer_path=Path(args.tokenizer) if args.tokenizer else None,
        device_arg=args.device,
        model_id=args.model_id,
    )
    app = create_app(state)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

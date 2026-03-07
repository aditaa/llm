"""Baseline training loop over token shard manifests."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from llm.model import ARCH_MODERN, GPTModel, ModelConfig, model_config_from_dict
from llm.tokenizer import load_tokenizer, tokenizer_contract_fingerprint, tokenizer_fingerprint


@dataclass
class TrainConfig:
    shards_path: Path
    output_dir: Path
    max_steps: int = 1000
    batch_size: int = 8
    context_length: int = 256
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 200
    lr_min_ratio: float = 0.10
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_steps: int = 20
    eval_freeze_batches: bool = True
    fail_on_eval_regression: bool = False
    eval_regression_tolerance: float = 0.20
    log_interval: int = 10
    seed: int = 42
    device: str = "auto"
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    dropout: float = 0.1
    architecture: str = ARCH_MODERN
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    ffn_hidden_multiplier: float = 8.0 / 3.0
    use_bias: bool = False
    resume_from: Path | None = None
    precision: str = "auto"
    tf32: bool = True
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    export_safetensors: bool = False
    safetensors_every_checkpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shards_path"] = str(self.shards_path)
        payload["output_dir"] = str(self.output_dir)
        payload["resume_from"] = str(self.resume_from) if self.resume_from else None
        return payload


@dataclass
class ShardTrainingInfo:
    manifest_paths: list[Path]
    tokenizer_path: Path
    tokenizer_hash: str
    tokenizer_contract_hash: str
    token_dtype: str
    vocab_size: int
    train_shards: list[Path]
    val_shards: list[Path]
    train_tokens: int
    val_tokens: int


def _lr_for_step(
    *,
    step: int,
    max_steps: int,
    base_lr: float,
    schedule: str,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    if base_lr <= 0:
        raise ValueError("base_lr must be > 0")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be in [0, 1]")

    normalized = schedule.strip().lower()
    if normalized not in {"constant", "cosine"}:
        raise ValueError("schedule must be 'constant' or 'cosine'")

    if normalized == "constant":
        return base_lr

    warmup = min(warmup_steps, max_steps)
    if warmup > 0 and step <= warmup:
        return base_lr * (step / warmup)

    if max_steps <= warmup:
        return base_lr

    progress = (step - warmup) / (max_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _build_optimizer(
    model: GPTModel,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    decay_params: list[Tensor] = []
    no_decay_params: list[Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        no_decay = (
            param.ndim < 2
            or lowered.endswith("bias")
            or "norm" in lowered
            or "embed" in lowered
        )
        if no_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
    meta = {"decay_tensors": len(decay_params), "no_decay_tensors": len(no_decay_params)}
    return optimizer, meta


def _iter_manifest_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.name != "manifest.json":
            raise ValueError(f"expected manifest.json file, got: {path}")
        return [path]

    if not path.exists():
        raise ValueError(f"path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"path must be a file or directory: {path}")

    manifests = sorted(p for p in path.rglob("manifest.json") if p.is_file())
    if not manifests:
        raise ValueError(f"no manifest.json files found under: {path}")
    return manifests


def _tokenizer_hash(path: Path) -> str:
    return tokenizer_fingerprint(path)


def collect_shard_training_info(path: Path) -> ShardTrainingInfo:
    manifest_paths = _iter_manifest_paths(path)

    train_shards: list[Path] = []
    val_shards: list[Path] = []
    train_tokens = 0
    val_tokens = 0

    tokenizer_path: Path | None = None
    tokenizer_hash: str | None = None
    tokenizer_contract_hash: str | None = None
    token_dtype: str | None = None
    vocab_size: int | None = None

    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_dir = manifest_path.parent

        this_tok_path = Path(str(manifest["tokenizer_path"]))
        if not this_tok_path.is_absolute():
            this_tok_path = (dataset_dir / this_tok_path).resolve()
        this_tok_hash = _tokenizer_hash(this_tok_path)
        manifest_tok_hash = str(manifest.get("tokenizer_hash", "")).strip()
        if manifest_tok_hash and manifest_tok_hash != this_tok_hash:
            raise ValueError(
                f"tokenizer_hash mismatch for manifest {manifest_path}: "
                f"manifest={manifest_tok_hash} actual={this_tok_hash}"
            )
        this_tok_contract_hash = tokenizer_contract_fingerprint(this_tok_path)
        manifest_contract_hash = str(manifest.get("tokenizer_contract_hash", "")).strip()
        if manifest_contract_hash and manifest_contract_hash != this_tok_contract_hash:
            raise ValueError(
                f"tokenizer_contract_hash mismatch for manifest {manifest_path}: "
                f"manifest={manifest_contract_hash} actual={this_tok_contract_hash}"
            )
        this_token_dtype = str(manifest["token_dtype"])
        this_vocab_size = int(manifest["tokenizer_vocab_size"])

        if tokenizer_path is None:
            tokenizer_path = this_tok_path
            tokenizer_hash = this_tok_hash
            tokenizer_contract_hash = this_tok_contract_hash
            token_dtype = this_token_dtype
            vocab_size = this_vocab_size
        else:
            if tokenizer_hash != this_tok_hash:
                raise ValueError(
                    "mismatched tokenizers detected across manifests. "
                    "Train on one tokenizer family at a time."
                )
            if tokenizer_contract_hash != this_tok_contract_hash:
                raise ValueError(
                    "mismatched tokenizer contracts detected across manifests. "
                    "Train on one tokenizer contract at a time."
                )
            if token_dtype != this_token_dtype:
                raise ValueError("mismatched token_dtype across manifests")
            if vocab_size != this_vocab_size:
                raise ValueError("mismatched vocab_size across manifests")

        for shard in manifest["train"]["shards"]:
            shard_path = dataset_dir / str(shard["path"])
            train_shards.append(shard_path)
            train_tokens += int(shard["tokens"])
        for shard in manifest["val"]["shards"]:
            shard_path = dataset_dir / str(shard["path"])
            val_shards.append(shard_path)
            val_tokens += int(shard["tokens"])

    if (
        tokenizer_path is None
        or tokenizer_hash is None
        or tokenizer_contract_hash is None
        or token_dtype is None
        or vocab_size is None
    ):
        raise ValueError("no valid manifests found")
    if not train_shards:
        raise ValueError("no train shards found")
    if not val_shards:
        raise ValueError("no val shards found")

    return ShardTrainingInfo(
        manifest_paths=manifest_paths,
        tokenizer_path=tokenizer_path,
        tokenizer_hash=tokenizer_hash,
        tokenizer_contract_hash=tokenizer_contract_hash,
        token_dtype=token_dtype,
        vocab_size=vocab_size,
        train_shards=train_shards,
        val_shards=val_shards,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
    )


def _np_dtype(token_dtype: str) -> np.dtype[np.generic]:
    if token_dtype == "uint16":
        return np.dtype(np.uint16)
    if token_dtype == "uint32":
        return np.dtype(np.uint32)
    raise ValueError(f"unsupported token dtype: {token_dtype}")


class ShardBatchSampler:
    def __init__(
        self,
        shard_paths: list[Path],
        token_dtype: str,
        context_length: int,
        seed: int,
        device: torch.device,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        self.context_length = context_length
        self.np_rng = np.random.default_rng(seed)
        self.device = device
        dtype = _np_dtype(token_dtype)
        self.arrays = [np.memmap(path, mode="r", dtype=dtype) for path in shard_paths]
        self.eligible: list[int] = []
        self.weights: list[int] = []
        for idx, arr in enumerate(self.arrays):
            if int(arr.size) > context_length:
                self.eligible.append(idx)
                self.weights.append(int(arr.size) - context_length)
        if not self.eligible:
            raise ValueError("no shards have enough tokens for context_length")
        self._eligible_np = np.asarray(self.eligible, dtype=np.int32)
        probs = np.asarray(self.weights, dtype=np.float64)
        self._weight_probs = probs / probs.sum()
        self._offsets = np.arange(self.context_length, dtype=np.int64)

    def sample_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        x = np.zeros((batch_size, self.context_length), dtype=np.int64)
        y = np.zeros((batch_size, self.context_length), dtype=np.int64)
        chosen = self.np_rng.choice(self._eligible_np, size=batch_size, p=self._weight_probs)
        for arr_idx in np.unique(chosen):
            rows = np.nonzero(chosen == arr_idx)[0]
            arr = self.arrays[int(arr_idx)]
            max_start = int(arr.size) - self.context_length - 1
            if max_start <= 0:
                raise ValueError("encountered shard smaller than context window")
            starts = self.np_rng.integers(0, max_start + 1, size=rows.size, dtype=np.int64)
            gather_idx = starts[:, None] + self._offsets[None, :]
            x[rows, :] = arr[gather_idx]
            y[rows, :] = arr[gather_idx + 1]

        xb = torch.from_numpy(x)
        yb = torch.from_numpy(y)
        if self.device.type == "cuda":
            return (
                xb.pin_memory().to(device=self.device, dtype=torch.long, non_blocking=True),
                yb.pin_memory().to(device=self.device, dtype=torch.long, non_blocking=True),
            )
        return (
            xb.to(device=self.device, dtype=torch.long),
            yb.to(device=self.device, dtype=torch.long),
        )


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def _cuda_bf16_supported() -> bool:
    is_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if is_supported is None:
        return False
    return bool(is_supported())


def _resolve_amp_mode(
    device: torch.device, precision: str
) -> tuple[bool, torch.dtype | None, bool, str]:
    normalized = precision.lower().strip()
    allowed = {"auto", "fp32", "fp16", "bf16"}
    if normalized not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"precision must be one of: {allowed_values}")

    if device.type != "cuda":
        if normalized in {"fp16", "bf16"}:
            raise ValueError(f"precision={normalized} requires a CUDA device")
        return False, None, False, "fp32"

    if normalized == "auto":
        if _cuda_bf16_supported():
            return True, torch.bfloat16, False, "bf16"
        return True, torch.float16, True, "fp16"

    if normalized == "fp32":
        return False, None, False, "fp32"

    if normalized == "bf16":
        if not _cuda_bf16_supported():
            raise ValueError("precision=bf16 is not supported on this GPU")
        return True, torch.bfloat16, False, "bf16"

    return True, torch.float16, True, "fp16"


def _estimate_loss(
    model: torch.nn.Module,
    sampler: ShardBatchSampler | None,
    *,
    batch_size: int,
    eval_steps: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    device_type: str,
    fixed_batches: list[tuple[Tensor, Tensor]] | None = None,
) -> float:
    model.eval()
    losses: list[float] = []
    autocast_kwargs: dict[str, Any] = {"device_type": device_type, "enabled": amp_enabled}
    if amp_dtype is not None:
        autocast_kwargs["dtype"] = amp_dtype
    with torch.no_grad():
        if fixed_batches is not None:
            batch_iter = fixed_batches
        else:
            if sampler is None:
                raise ValueError("sampler is required when fixed_batches is not provided")
            batch_iter = [sampler.sample_batch(batch_size) for _ in range(eval_steps)]
        for xb, yb in batch_iter:
            with torch.autocast(**autocast_kwargs):
                _, loss = model(xb, yb)
            if loss is None:
                raise RuntimeError("loss should not be None when targets are provided")
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def _save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: TrainConfig,
    model_config: ModelConfig,
    info: ShardTrainingInfo,
    lr: float,
    best_val_loss: float | None,
    best_val_ppl: float | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"ckpt_step_{step:07d}.pt"
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "train_config": config.to_dict(),
        "model_config": model_config.to_dict(),
        "tokenizer_path": str(info.tokenizer_path),
        "tokenizer_hash": info.tokenizer_hash,
        "tokenizer_contract_hash": info.tokenizer_contract_hash,
        "lr": lr,
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_val_ppl,
    }
    torch.save(payload, ckpt_path)
    torch.save(payload, output_dir / "last.pt")
    if config.export_safetensors:
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "export_safetensors requires the 'safetensors' package. "
                "Install with training extras."
            ) from exc
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in model.state_dict().items()
            if isinstance(tensor, torch.Tensor)
        }
        last_safe_path = output_dir / "last.safetensors"
        save_file(state, str(last_safe_path))
        if config.safetensors_every_checkpoint:
            step_safe_path = output_dir / f"ckpt_step_{step:07d}.safetensors"
            save_file(state, str(step_safe_path))

        safe_meta = {
            "step": step,
            "source_checkpoint": str(ckpt_path),
            "model_config": model_config.to_dict(),
            "tokenizer_path": str(info.tokenizer_path),
            "tokenizer_hash": info.tokenizer_hash,
            "tokenizer_contract_hash": info.tokenizer_contract_hash,
            "exported_by": "llm.train",
        }
        (output_dir / "last.safetensors.meta.json").write_text(
            json.dumps(safe_meta, indent=2),
            encoding="utf-8",
        )
    return ckpt_path


def run_training(config: TrainConfig) -> dict[str, Any]:
    if config.max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.eval_steps <= 0:
        raise ValueError("eval_steps must be > 0")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be > 0")
    if config.context_length <= 0:
        raise ValueError("context_length must be > 0")
    if config.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    if config.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be >= 0")
    if not 0.0 <= config.lr_min_ratio <= 1.0:
        raise ValueError("lr_min_ratio must be in [0, 1]")
    if config.lr_schedule not in {"constant", "cosine"}:
        raise ValueError("lr_schedule must be one of: constant, cosine")
    if config.eval_regression_tolerance < 0.0:
        raise ValueError("eval_regression_tolerance must be >= 0")
    if config.safetensors_every_checkpoint and not config.export_safetensors:
        raise ValueError("safetensors_every_checkpoint requires export_safetensors")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = _resolve_device(config.device)
    amp_enabled, amp_dtype, use_grad_scaler, effective_precision = _resolve_amp_mode(
        device, config.precision
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.tf32
        torch.backends.cudnn.allow_tf32 = config.tf32

    info = collect_shard_training_info(config.shards_path)

    model_config = ModelConfig(
        vocab_size=info.vocab_size,
        max_seq_len=config.context_length,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        dropout=config.dropout,
        architecture=config.architecture,
        rope_theta=config.rope_theta,
        norm_eps=config.norm_eps,
        ffn_hidden_multiplier=config.ffn_hidden_multiplier,
        use_bias=config.use_bias,
    )

    start_step = 0
    best_val_loss: float | None = None
    best_val_ppl: float | None = None
    resume_checkpoint: dict[str, Any] | None = None
    if config.resume_from is not None:
        resume_checkpoint = torch.load(config.resume_from, map_location=device)
        checkpoint_model_config = resume_checkpoint.get("model_config")
        if isinstance(checkpoint_model_config, dict):
            model_config = model_config_from_dict(checkpoint_model_config)
        if model_config.vocab_size != info.vocab_size:
            raise ValueError(
                "resume checkpoint vocab_size does not match shard tokenizer vocab_size"
            )

    model = GPTModel(model_config).to(device)
    optimizer, optimizer_meta = _build_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scaler_state = resume_checkpoint.get("scaler_state")
        if use_grad_scaler and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        start_step = int(resume_checkpoint["step"])
        best_val_loss_raw = resume_checkpoint.get("best_val_loss")
        best_val_ppl_raw = resume_checkpoint.get("best_val_ppl")
        if isinstance(best_val_loss_raw, (int, float)):
            best_val_loss = float(best_val_loss_raw)
        if isinstance(best_val_ppl_raw, (int, float)):
            best_val_ppl = float(best_val_ppl_raw)

    train_model: torch.nn.Module = model
    if config.compile_model:
        if not hasattr(torch, "compile"):
            raise ValueError("compile_model requested but torch.compile is unavailable")
        train_model = cast(torch.nn.Module, torch.compile(model, mode=config.compile_mode))

    train_sampler = ShardBatchSampler(
        shard_paths=info.train_shards,
        token_dtype=info.token_dtype,
        context_length=config.context_length,
        seed=config.seed,
        device=device,
    )
    val_sampler = ShardBatchSampler(
        shard_paths=info.val_shards,
        token_dtype=info.token_dtype,
        context_length=config.context_length,
        seed=config.seed + 1,
        device=device,
    )
    frozen_eval_batches: list[tuple[Tensor, Tensor]] | None = None
    if config.eval_freeze_batches:
        frozen_eval_batches = [
            val_sampler.sample_batch(config.batch_size) for _ in range(config.eval_steps)
        ]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "train_config": config.to_dict(),
                "model_config": model_config.to_dict(),
                "tokenizer_path": str(info.tokenizer_path),
                "tokenizer_hash": info.tokenizer_hash,
                "tokenizer_contract_hash": info.tokenizer_contract_hash,
                "manifests": [str(p) for p in info.manifest_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tokenizer = load_tokenizer(info.tokenizer_path)
    print(f"device={device}")
    print(f"architecture={model_config.architecture}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"tokenizer_contract_hash={info.tokenizer_contract_hash}")
    print(f"grad_accum_steps={config.grad_accum_steps}")
    print(f"lr_schedule={config.lr_schedule}")
    print(f"lr_warmup_steps={config.lr_warmup_steps}")
    print(f"lr_min_ratio={config.lr_min_ratio}")
    print(f"train_tokens={info.train_tokens}")
    print(f"val_tokens={info.val_tokens}")
    print(f"start_step={start_step}")
    print(f"precision={effective_precision}")
    print(f"optimizer_decay_tensors={optimizer_meta['decay_tensors']}")
    print(f"optimizer_no_decay_tensors={optimizer_meta['no_decay_tensors']}")
    print(f"tf32={int(config.tf32)}")
    print(f"compile_model={int(config.compile_model)}")
    print(f"eval_freeze_batches={int(config.eval_freeze_batches)}")
    print(f"fail_on_eval_regression={int(config.fail_on_eval_regression)}")
    if config.compile_model:
        print(f"compile_mode={config.compile_mode}")
    if config.export_safetensors:
        print("export_safetensors=1")
        print(f"safetensors_every_checkpoint={int(config.safetensors_every_checkpoint)}")

    autocast_kwargs: dict[str, Any] = {"device_type": device.type, "enabled": amp_enabled}
    if amp_dtype is not None:
        autocast_kwargs["dtype"] = amp_dtype
    tokens_seen = 0
    log_tokens_mark = 0
    log_time_mark = time.perf_counter()
    for step in range(start_step + 1, config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        lr = _lr_for_step(
            step=step,
            max_steps=config.max_steps,
            base_lr=config.learning_rate,
            schedule=config.lr_schedule,
            warmup_steps=config.lr_warmup_steps,
            min_ratio=config.lr_min_ratio,
        )
        _set_optimizer_lr(optimizer, lr)
        micro_losses: list[float] = []
        for _ in range(config.grad_accum_steps):
            xb, yb = train_sampler.sample_batch(config.batch_size)
            tokens_seen += int(xb.numel())
            with torch.autocast(**autocast_kwargs):
                _, loss = train_model(xb, yb)
            if loss is None:
                raise RuntimeError("loss should not be None when targets are provided")
            micro_losses.append(float(loss.item()))
            loss_to_backward = loss / config.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()

        train_loss = sum(micro_losses) / len(micro_losses)
        if step == 1 or step % config.log_interval == 0:
            now = time.perf_counter()
            elapsed = max(now - log_time_mark, 1e-9)
            tok_window = tokens_seen - log_tokens_mark
            toks_per_sec = tok_window / elapsed
            print(
                f"step={step} train_loss={train_loss:.6f} lr={lr:.8f} "
                f"tokens_seen={tokens_seen} toks_per_sec={toks_per_sec:.2f}"
            )
            log_tokens_mark = tokens_seen
            log_time_mark = now

        should_eval = step == 1 or step % config.eval_interval == 0 or step == config.max_steps
        if should_eval:
            val_loss = _estimate_loss(
                train_model,
                val_sampler if frozen_eval_batches is None else None,
                batch_size=config.batch_size,
                eval_steps=config.eval_steps,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                device_type=device.type,
                fixed_batches=frozen_eval_batches,
            )
            val_ppl = float(math.exp(min(val_loss, 20.0)))
            prev_best_val_ppl = best_val_ppl
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
            if best_val_ppl is None or val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
            ckpt = _save_checkpoint(
                output_dir=config.output_dir,
                step=step,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                model_config=model_config,
                info=info,
                lr=lr,
                best_val_loss=best_val_loss,
                best_val_ppl=best_val_ppl,
            )
            print(
                f"step={step} val_loss={val_loss:.6f} val_ppl={val_ppl:.4f} "
                f"best_val_ppl={best_val_ppl:.4f} checkpoint={ckpt}"
            )
            if (
                config.fail_on_eval_regression
                and prev_best_val_ppl is not None
                and val_ppl > (prev_best_val_ppl * (1.0 + config.eval_regression_tolerance))
            ):
                threshold = prev_best_val_ppl * (1.0 + config.eval_regression_tolerance)
                raise RuntimeError(
                    "held-out eval regression gate failed: "
                    f"val_ppl={val_ppl:.4f} exceeded threshold={threshold:.4f}"
                )

    return {
        "output_dir": str(config.output_dir),
        "max_steps": config.max_steps,
        "start_step": start_step,
        "tokenizer_path": str(info.tokenizer_path),
        "tokenizer_hash": info.tokenizer_hash,
        "tokenizer_contract_hash": info.tokenizer_contract_hash,
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_val_ppl,
    }

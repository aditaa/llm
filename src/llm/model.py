"""Decoder-only GPT model with modern and legacy architecture support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ARCH_LEGACY = "gpt_learnedpos_layernorm_gelu_v0"
ARCH_MODERN = "gpt_rope_rmsnorm_swiglu_v1"
SUPPORTED_ARCHITECTURES = {ARCH_LEGACY, ARCH_MODERN}


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    dropout: float = 0.1
    architecture: str = ARCH_MODERN
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    ffn_hidden_multiplier: float = 8.0 / 3.0
    use_bias: bool = False

    def __post_init__(self) -> None:
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            allowed = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
            raise ValueError(f"architecture must be one of: {allowed}")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.ffn_hidden_multiplier <= 0:
            raise ValueError("ffn_hidden_multiplier must be > 0")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be > 0")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be > 0")

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def model_config_from_dict(payload: dict[str, Any]) -> ModelConfig:
    data = dict(payload)
    if "architecture" not in data:
        data["architecture"] = ARCH_LEGACY
    return ModelConfig(**data)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        use_rope: bool,
        rope_theta: float,
        use_bias: bool,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.use_rope = use_rope

        if self.use_rope and (self.head_dim % 2 != 0):
            raise ValueError("head_dim must be even when using RoPE")

        if self.use_rope:
            self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
            self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
            self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        else:
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.out = nn.Linear(d_model, d_model, bias=use_bias)

        if self.use_rope:
            inv_freq = 1.0 / (
                rope_theta
                ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _apply_rope(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        seq_len = q.shape[2]
        positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
        inv_freq = cast(Tensor, self.inv_freq)
        freqs = torch.outer(positions, inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)
        cos = angles.cos().to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        sin = angles.sin().to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        return ((q * cos) + (self._rotate_half(q) * sin), (k * cos) + (self._rotate_half(k) * sin))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        if self.use_rope:
            q = self.q_proj(x).view(
                batch_size, seq_len, self.n_heads, self.head_dim
            ).transpose(1, 2)
            k = self.k_proj(x).view(
                batch_size, seq_len, self.n_heads, self.head_dim
            ).transpose(1, 2)
            v = self.v_proj(x).view(
                batch_size, seq_len, self.n_heads, self.head_dim
            ).transpose(1, 2)
            q, k = self._apply_rope(q, k)
        else:
            qkv = self.qkv(x)
            qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out(attn)


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        *,
        use_swiglu: bool,
        hidden_multiplier: float,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.use_swiglu = use_swiglu
        self.dropout = nn.Dropout(dropout)

        if self.use_swiglu:
            hidden = max(1, int(d_model * hidden_multiplier))
            self.gate_proj = nn.Linear(d_model, hidden, bias=use_bias)
            self.up_proj = nn.Linear(d_model, hidden, bias=use_bias)
            self.down_proj = nn.Linear(hidden, d_model, bias=use_bias)
        else:
            hidden = 4 * d_model
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden, bias=use_bias),
                nn.GELU(),
                nn.Linear(hidden, d_model, bias=use_bias),
                self.dropout,
            )

    def forward(self, x: Tensor) -> Tensor:
        if not self.use_swiglu:
            return self.net(x)
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        modern = config.architecture == ARCH_MODERN

        self.norm1: nn.Module
        self.norm2: nn.Module
        if modern:
            self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
            self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        else:
            self.norm1 = nn.LayerNorm(config.d_model)
            self.norm2 = nn.LayerNorm(config.d_model)

        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_rope=modern,
            rope_theta=config.rope_theta,
            use_bias=config.use_bias,
        )
        self.ffn = FeedForward(
            d_model=config.d_model,
            dropout=config.dropout,
            use_swiglu=modern,
            hidden_multiplier=config.ffn_hidden_multiplier,
            use_bias=config.use_bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.modern = config.architecture == ARCH_MODERN

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.ln_f: nn.Module
        if self.modern:
            self.pos_embed = None
            self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)
        else:
            self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
            self.ln_f = nn.LayerNorm(config.d_model)

        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}"
            )

        x = self.token_embed(input_ids)
        if self.pos_embed is not None:
            pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_embed(pos_ids)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, -1),
                targets.reshape(batch_size * seq_len),
            )
        return logits, loss

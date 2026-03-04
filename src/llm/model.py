"""Model module placeholder for upcoming GPT-style implementation."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    dropout: float = 0.1

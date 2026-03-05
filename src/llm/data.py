"""Data utilities for turning token IDs into next-token training batches."""

from __future__ import annotations

import random
from typing import Iterator, Sequence


class TokenWindowDataset:
    """Creates fixed-length autoregressive windows from token IDs."""

    def __init__(self, token_ids: Sequence[int], context_length: int, stride: int = 1) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        self.token_ids = list(token_ids)
        self.context_length = context_length
        self.stride = stride
        self._max_start = len(self.token_ids) - self.context_length

    def __len__(self) -> int:
        if self._max_start <= 0:
            return 0
        return len(range(0, self._max_start, self.stride))

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        if index < 0 or index >= len(self):
            raise IndexError("TokenWindowDataset index out of range")
        start = index * self.stride
        end = start + self.context_length
        x = self.token_ids[start:end]
        y = self.token_ids[start + 1 : end + 1]
        return x, y


def split_token_ids(
    token_ids: Sequence[int], train_ratio: float = 0.9
) -> tuple[list[int], list[int]]:
    """Contiguous train/validation split of token IDs."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    split_at = int(len(token_ids) * train_ratio)
    return list(token_ids[:split_at]), list(token_ids[split_at:])


def split_indices(
    num_items: int, train_ratio: float = 0.9, seed: int = 42, shuffle: bool = True
) -> tuple[list[int], list[int]]:
    """Train/validation index split with optional deterministic shuffling."""
    if num_items < 0:
        raise ValueError("num_items must be >= 0")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    indices = list(range(num_items))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    split_at = int(num_items * train_ratio)
    return indices[:split_at], indices[split_at:]


def collate_batch(
    examples: Sequence[tuple[Sequence[int], Sequence[int]]], as_tensors: bool = False
) -> tuple[object, object]:
    """Stacks per-example token windows into batch-first arrays/tensors."""
    if not examples:
        raise ValueError("examples must not be empty")
    inputs = [list(x) for x, _ in examples]
    targets = [list(y) for _, y in examples]

    if not as_tensors:
        return inputs, targets

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required when as_tensors=True") from exc

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def iter_batches(
    dataset: TokenWindowDataset,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 42,
    drop_last: bool = False,
    as_tensors: bool = False,
) -> Iterator[tuple[object, object]]:
    """Yields collated mini-batches from a TokenWindowDataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    indices = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        if drop_last and len(chunk) < batch_size:
            continue
        examples = [dataset[idx] for idx in chunk]
        yield collate_batch(examples, as_tensors=as_tensors)

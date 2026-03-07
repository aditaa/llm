#!/usr/bin/env bash
set -euo pipefail

# Tuned for: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM)
# Profile source: configs/train/rtx5070/fineweb_global_bpe_v1_big.json

PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src \
.venv/bin/python -m llm.cli train \
  --shards-path data/shards_global/fineweb-global-bpe-v1 \
  --output-dir artifacts/checkpoints/fineweb-global-bpe-v1-big-run1 \
  --device cuda \
  --max-steps 50000 \
  --batch-size 34 \
  --context-length 512 \
  --n-layers 12 \
  --n-heads 12 \
  --d-model 768 \
  --learning-rate 1.5e-4 \
  --lr-schedule cosine \
  --lr-warmup-steps 1000 \
  --lr-min-ratio 0.10 \
  --eval-interval 2000 \
  --eval-steps 5 \
  --fail-on-eval-regression \
  --eval-regression-tolerance 0.20 \
  --log-interval 100 \
  --precision auto

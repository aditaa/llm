#!/usr/bin/env bash
set -euo pipefail

# Tuned for: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM)
# Profile source: configs/train/rtx5070/fineweb_350bt_bpe_v2_longrun.json

PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src \
.venv/bin/python -m llm.cli train \
  --shards-path data/shards_global/fineweb-global-bpe-v1 \
  --output-dir artifacts/checkpoints/fineweb-350bt-bpe-v2-run1 \
  --device cuda \
  --max-steps 100000 \
  --batch-size 34 \
  --context-length 512 \
  --n-layers 12 \
  --n-heads 12 \
  --d-model 768 \
  --learning-rate 3e-4 \
  --lr-schedule cosine \
  --lr-warmup-steps 2000 \
  --lr-min-ratio 0.10 \
  --eval-interval 1000 \
  --eval-steps 6 \
  --fail-on-eval-regression \
  --eval-regression-tolerance 0.20 \
  --log-interval 100 \
  --precision auto \
  --checkpoint-keep-last 6 \
  --checkpoint-keep-every 10000 \
  --ema-decay 0.999 \
  --ema-update-every 1 \
  --ema-start-step 1000 \
  --export-safetensors

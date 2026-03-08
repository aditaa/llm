#!/usr/bin/env bash
set -euo pipefail

# Context-extension continuation stage (1024 tokens) on RTX 5070 Ti 12 GB.
# Source profile: configs/train/rtx5070/fineweb_350bt_bpe_v2_ctx1024_stage.json

PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src \
.venv/bin/python -m llm.cli train \
  --shards-path data/shards_global/fineweb-global-bpe-v1 \
  --output-dir artifacts/checkpoints/fineweb-350bt-bpe-v2-ctx1024 \
  --device cuda \
  --resume-from artifacts/checkpoints/fineweb-350bt-bpe-v2-run1/last.pt \
  --allow-context-extension \
  --max-steps 150000 \
  --batch-size 6 \
  --grad-accum-steps 4 \
  --context-length 1024 \
  --n-layers 12 \
  --n-heads 12 \
  --d-model 768 \
  --learning-rate 1.5e-4 \
  --lr-schedule cosine \
  --lr-warmup-steps 3000 \
  --lr-min-ratio 0.10 \
  --eval-interval 1500 \
  --eval-steps 6 \
  --log-interval 100 \
  --precision auto \
  --checkpoint-keep-last 6 \
  --checkpoint-keep-every 10000 \
  --ema-decay 0.999 \
  --ema-update-every 1 \
  --ema-start-step 2000 \
  --export-safetensors

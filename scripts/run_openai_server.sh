#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-auto}"
MODEL_ID="${MODEL_ID:-llm-from-scratch-local}"

if [[ -z "$MODEL_DIR" ]]; then
  echo "Usage: bash scripts/run_openai_server.sh <model-dir>" >&2
  echo "Expected files in model-dir: checkpoint.pt tokenizer.json" >&2
  exit 1
fi

CHECKPOINT_PATH="${MODEL_DIR}/checkpoint.pt"
TOKENIZER_PATH="${MODEL_DIR}/tokenizer.json"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "error: missing checkpoint: $CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "error: missing tokenizer: $TOKENIZER_PATH" >&2
  exit 1
fi

PYTHONPATH=src .venv/bin/python -m llm.inference_server \
  --checkpoint "$CHECKPOINT_PATH" \
  --tokenizer "$TOKENIZER_PATH" \
  --model-id "$MODEL_ID" \
  --device "$DEVICE" \
  --host "$HOST" \
  --port "$PORT"

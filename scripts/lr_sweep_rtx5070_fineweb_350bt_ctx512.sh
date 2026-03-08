#!/usr/bin/env bash
set -euo pipefail

# RTX 5070 Ti (12 GB) LR sweep on staged FineWeb 350BT shards.
# Override via env vars as needed.

SHARDS_PATH="${SHARDS_PATH:-data/shards_global/fineweb-global-bpe-v1}"
OUT_ROOT="${OUT_ROOT:-artifacts/checkpoints/lr_sweep_350bt_ctx512_$(date +%Y%m%d_%H%M%S)}"
LRS="${LRS:-2e-4 3e-4 4e-4}"

MAX_STEPS="${MAX_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-34}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
N_LAYERS="${N_LAYERS:-12}"
N_HEADS="${N_HEADS:-12}"
D_MODEL="${D_MODEL:-768}"

LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-300}"
EVAL_STEPS="${EVAL_STEPS:-6}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-auto}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "error: .venv/bin/python not found; run make setup-train first" >&2
  exit 1
fi

if [[ ! -e "$SHARDS_PATH" ]]; then
  echo "error: shards path not found: $SHARDS_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
SUMMARY_TSV="$OUT_ROOT/summary.tsv"
echo -e "lr\trun_dir\tbest_val_ppl\texit_code" > "$SUMMARY_TSV"

for lr in $LRS; do
  lr_slug="$(echo "$lr" | sed 's/-/m/g; s/\./p/g')"
  run_dir="$OUT_ROOT/lr_${lr_slug}"
  mkdir -p "$run_dir"
  log_path="$run_dir/train.log"

  echo "[lr-sweep] start lr=$lr run_dir=$run_dir"

  set +e
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=src \
  .venv/bin/python -u -m llm.cli train \
    --shards-path "$SHARDS_PATH" \
    --output-dir "$run_dir" \
    --device "$DEVICE" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --context-length "$CONTEXT_LENGTH" \
    --n-layers "$N_LAYERS" \
    --n-heads "$N_HEADS" \
    --d-model "$D_MODEL" \
    --learning-rate "$lr" \
    --lr-schedule cosine \
    --lr-warmup-steps "$LR_WARMUP_STEPS" \
    --lr-min-ratio "$LR_MIN_RATIO" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-steps "$EVAL_STEPS" \
    --fail-on-eval-regression \
    --eval-regression-tolerance 0.20 \
    --log-interval "$LOG_INTERVAL" \
    --precision "$PRECISION" \
    > "$log_path" 2>&1
  rc=$?
  set -e

  best_val_ppl="$(python3 - <<'PY' "$log_path"
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
best = None
pattern = re.compile(r"best_val_ppl=([0-9]+(?:\.[0-9]+)?)")
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    m = pattern.search(line)
    if m:
        best = m.group(1)
print(best if best is not None else "NA")
PY
)"

  echo -e "${lr}\t${run_dir}\t${best_val_ppl}\t${rc}" >> "$SUMMARY_TSV"
  echo "[lr-sweep] done lr=$lr rc=$rc best_val_ppl=$best_val_ppl"
done

echo "[lr-sweep] summary=$SUMMARY_TSV"

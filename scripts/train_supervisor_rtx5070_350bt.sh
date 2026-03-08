#!/usr/bin/env bash
set -euo pipefail

# Auto-resume supervisor for staged FineWeb 350BT training on RTX 5070.
# Runs training in step chunks and resumes from last checkpoint each cycle.
# Each new cycle re-reads shard manifests so newly built batches are included.

SHARDS_PATH="data/shards_global/fineweb-global-bpe-v1"
OUTPUT_DIR="artifacts/checkpoints/fineweb-350bt-bpe-v2-run1"
STATE_DIR="artifacts/reports/train_supervisor_350bt"

POLL_SECONDS=120
STEP_CHUNK=2000
MIN_MANIFESTS=1
MAX_FAILURE_STREAK=0  # 0 = unlimited retries

DEVICE="cuda"
BATCH_SIZE=34
CONTEXT_LENGTH=512
N_LAYERS=12
N_HEADS=12
D_MODEL=768
LEARNING_RATE="3e-4"
LR_WARMUP_STEPS=2000
LR_MIN_RATIO="0.10"
EVAL_INTERVAL=1000
EVAL_STEPS=6
LOG_INTERVAL=100
PRECISION="auto"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/train_supervisor_rtx5070_350bt.sh [options]

Options:
  --shards-path DIR            Root containing shard manifest.json files
  --output-dir DIR             Training output directory (last.pt lives here)
  --state-dir DIR              Supervisor logs/state directory
  --poll-seconds N             Sleep between checks/restarts (default: 120)
  --step-chunk N               Steps per training cycle before restart (default: 2000)
  --min-manifests N            Wait until at least N manifests exist (default: 1)
  --max-failure-streak N       Stop after N consecutive train failures (0 = never)

  --device NAME                Training device (default: cuda)
  --batch-size N               Batch size (default: 34)
  --context-length N           Context length (default: 512)
  --n-layers N                 Transformer layer count (default: 12)
  --n-heads N                  Attention head count (default: 12)
  --d-model N                  Hidden size (default: 768)
  --learning-rate X            Learning rate (default: 3e-4)
  --lr-warmup-steps N          Warmup steps (default: 2000)
  --lr-min-ratio X             Cosine min ratio (default: 0.10)
  --eval-interval N            Eval interval (default: 1000)
  --eval-steps N               Eval steps (default: 6)
  --log-interval N             Train log interval (default: 100)
  --precision MODE             Precision mode (default: auto)
  -h, --help                   Show help

Example:
  bash scripts/train_supervisor_rtx5070_350bt.sh \
    --step-chunk 2000 \
    --poll-seconds 120
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shards-path) SHARDS_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --state-dir) STATE_DIR="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --step-chunk) STEP_CHUNK="$2"; shift 2 ;;
    --min-manifests) MIN_MANIFESTS="$2"; shift 2 ;;
    --max-failure-streak) MAX_FAILURE_STREAK="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
    --n-layers) N_LAYERS="$2"; shift 2 ;;
    --n-heads) N_HEADS="$2"; shift 2 ;;
    --d-model) D_MODEL="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --lr-warmup-steps) LR_WARMUP_STEPS="$2"; shift 2 ;;
    --lr-min-ratio) LR_MIN_RATIO="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --eval-steps) EVAL_STEPS="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -x ".venv/bin/python" ]]; then
  echo "error: .venv/bin/python not found; run make setup-train first" >&2
  exit 1
fi
if [[ ! -d "$SHARDS_PATH" ]]; then
  echo "error: shards-path not found: $SHARDS_PATH" >&2
  exit 1
fi
if [[ "$STEP_CHUNK" -le 0 ]]; then
  echo "error: step-chunk must be > 0" >&2
  exit 1
fi
if [[ "$POLL_SECONDS" -le 0 ]]; then
  echo "error: poll-seconds must be > 0" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$STATE_DIR"

LOCK_FILE="$STATE_DIR/supervisor.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "error: another train supervisor instance is already running" >&2
  exit 1
fi

SUP_LOG="$STATE_DIR/supervisor_$(date +%Y%m%d_%H%M%S).log"
touch "$SUP_LOG"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$SUP_LOG"
}

current_step() {
  local ckpt="$OUTPUT_DIR/last.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "0"
    return 0
  fi
  python3 - <<'PY' "$ckpt"
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt.get("step", 0)))
PY
}

manifest_count() {
  find "$SHARDS_PATH" -name manifest.json 2>/dev/null | wc -l | tr -d ' '
}

failure_streak=0
log "supervisor_start shards_path=$SHARDS_PATH output_dir=$OUTPUT_DIR step_chunk=$STEP_CHUNK"

while true; do
  mcount="$(manifest_count)"
  if [[ "$mcount" -lt "$MIN_MANIFESTS" ]]; then
    log "waiting_for_manifests have=$mcount need=$MIN_MANIFESTS sleep=${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
    continue
  fi

  step_now="$(current_step)"
  target_step=$((step_now + STEP_CHUNK))
  run_tag="$(date +%Y%m%d_%H%M%S)"
  run_log="$STATE_DIR/train_${step_now}_to_${target_step}_${run_tag}.log"
  resume_args=()
  if [[ -f "$OUTPUT_DIR/last.pt" ]]; then
    resume_args=(--resume-from "$OUTPUT_DIR/last.pt")
  fi

  log "train_launch manifests=$mcount step_now=$step_now target_step=$target_step run_log=$run_log"

  set +e
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=src \
  .venv/bin/python -u -m llm.cli train \
    --shards-path "$SHARDS_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --max-steps "$target_step" \
    --batch-size "$BATCH_SIZE" \
    --context-length "$CONTEXT_LENGTH" \
    --n-layers "$N_LAYERS" \
    --n-heads "$N_HEADS" \
    --d-model "$D_MODEL" \
    --learning-rate "$LEARNING_RATE" \
    --lr-schedule cosine \
    --lr-warmup-steps "$LR_WARMUP_STEPS" \
    --lr-min-ratio "$LR_MIN_RATIO" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-steps "$EVAL_STEPS" \
    --fail-on-eval-regression \
    --eval-regression-tolerance 0.20 \
    --log-interval "$LOG_INTERVAL" \
    --precision "$PRECISION" \
    --export-safetensors \
    "${resume_args[@]}" \
    > >(tee -a "$run_log") 2>&1
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    failure_streak=0
    new_step="$(current_step)"
    log "train_done rc=0 step_now=$new_step"
  else
    failure_streak=$((failure_streak + 1))
    log "train_failed rc=$rc failure_streak=$failure_streak"
    if [[ "$MAX_FAILURE_STREAK" -gt 0 && "$failure_streak" -ge "$MAX_FAILURE_STREAK" ]]; then
      log "supervisor_stop reason=max_failure_streak_reached"
      exit 10
    fi
  fi

  sleep "$POLL_SECONDS"
done

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/benchmark_rtx5070_context_profiles.sh [options]

Short throughput/memory benchmark sweep for RTX 5070 profile sizing.
Each profile runs a bounded training probe and writes a summary TSV.

Options:
  --shards-path PATH         Shards root or manifest path
                             (default: data/shards_global/fineweb-global-bpe-v1)
  --out-root PATH            Output root for benchmark runs
                             (default: artifacts/reports/rtx5070_ctx_bench_<ts>)
  --profiles CSV             Comma-separated context:batch:grad_accum entries
                             (default: 512:34:1,768:16:2,1024:6:4)
  --max-steps N              Max train steps per profile (default: 1200)
  --learning-rate FLOAT      Learning rate for probe runs (default: 1.5e-4)
  --lr-warmup-steps N        LR warmup steps (default: 200)
  --lr-min-ratio FLOAT       Cosine floor ratio (default: 0.10)
  --eval-interval N          Eval interval (default: 300)
  --eval-steps N             Eval steps (default: 6)
  --log-interval N           Train log interval (default: 50)
  --sample-seconds N         GPU sampling interval seconds (default: 2)
  --n-layers N               Model layers (default: 12)
  --n-heads N                Attention heads (default: 12)
  --d-model N                Hidden width (default: 768)
  --device DEVICE            Train device (default: cuda)
  --precision MODE           Train precision (default: auto)
  --compile-model            Enable torch.compile during probes
  --help                     Show this help

Outputs:
  <out-root>/summary.tsv
  <out-root>/<profile>/train.log
  <out-root>/<profile>/gpu_samples.csv
USAGE
}

SHARDS_PATH="data/shards_global/fineweb-global-bpe-v1"
OUT_ROOT="artifacts/reports/rtx5070_ctx_bench_$(date +%Y%m%d_%H%M%S)"
PROFILES="512:34:1,768:16:2,1024:6:4"
MAX_STEPS=1200
LEARNING_RATE="1.5e-4"
LR_WARMUP_STEPS=200
LR_MIN_RATIO="0.10"
EVAL_INTERVAL=300
EVAL_STEPS=6
LOG_INTERVAL=50
SAMPLE_SECONDS=2
N_LAYERS=12
N_HEADS=12
D_MODEL=768
DEVICE="cuda"
PRECISION="auto"
COMPILE_MODEL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shards-path)
      SHARDS_PATH="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --profiles)
      PROFILES="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --lr-warmup-steps)
      LR_WARMUP_STEPS="$2"
      shift 2
      ;;
    --lr-min-ratio)
      LR_MIN_RATIO="$2"
      shift 2
      ;;
    --eval-interval)
      EVAL_INTERVAL="$2"
      shift 2
      ;;
    --eval-steps)
      EVAL_STEPS="$2"
      shift 2
      ;;
    --log-interval)
      LOG_INTERVAL="$2"
      shift 2
      ;;
    --sample-seconds)
      SAMPLE_SECONDS="$2"
      shift 2
      ;;
    --n-layers)
      N_LAYERS="$2"
      shift 2
      ;;
    --n-heads)
      N_HEADS="$2"
      shift 2
      ;;
    --d-model)
      D_MODEL="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --compile-model)
      COMPILE_MODEL=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x ".venv/bin/python" ]]; then
  echo "error: .venv/bin/python not found; run make setup-train first" >&2
  exit 1
fi

if [[ ! -e "$SHARDS_PATH" ]]; then
  echo "error: shards path not found: $SHARDS_PATH" >&2
  exit 1
fi

if ! [[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || [[ "$MAX_STEPS" -le 0 ]]; then
  echo "error: --max-steps must be a positive integer" >&2
  exit 2
fi

if ! [[ "$SAMPLE_SECONDS" =~ ^[0-9]+$ ]] || [[ "$SAMPLE_SECONDS" -le 0 ]]; then
  echo "error: --sample-seconds must be a positive integer" >&2
  exit 2
fi

has_nvidia_smi=0
if command -v nvidia-smi >/dev/null 2>&1; then
  has_nvidia_smi=1
fi

mkdir -p "$OUT_ROOT"
SUMMARY_TSV="$OUT_ROOT/summary.tsv"

echo -e "profile\tcontext\tbatch\tgrad_accum\teffective_batch\tavg_toks_per_sec\tmax_toks_per_sec\tbest_val_ppl\tavg_gpu_util\tmax_gpu_util\tavg_gpu_mem_mib\tpeak_gpu_mem_mib\tgpu_samples\texit_code\trun_dir" > "$SUMMARY_TSV"

parse_metrics() {
  local log_path="$1"
  local gpu_csv="$2"
  python3 - <<'PY' "$log_path" "$gpu_csv"
import csv
import re
import statistics
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
gpu_path = Path(sys.argv[2])

tok_values = []
best_val = None
tok_pat = re.compile(r"toks_per_sec=([0-9]+(?:\.[0-9]+)?)")
best_pat = re.compile(r"best_val_ppl=([0-9]+(?:\.[0-9]+)?)")
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    tok_match = tok_pat.search(line)
    if tok_match:
        tok_values.append(float(tok_match.group(1)))
    best_match = best_pat.search(line)
    if best_match:
        best_val = float(best_match.group(1))

if tok_values:
    avg_tok = f"{statistics.fmean(tok_values):.2f}"
    max_tok = f"{max(tok_values):.2f}"
else:
    avg_tok = "NA"
    max_tok = "NA"

util_values = []
mem_values = []
if gpu_path.exists():
    with gpu_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                util = float(str(row[1]).strip())
                mem = float(str(row[2]).strip())
            except ValueError:
                continue
            util_values.append(util)
            mem_values.append(mem)

if util_values:
    avg_util = f"{statistics.fmean(util_values):.1f}"
    max_util = f"{max(util_values):.1f}"
else:
    avg_util = "NA"
    max_util = "NA"

if mem_values:
    avg_mem = f"{statistics.fmean(mem_values):.0f}"
    peak_mem = f"{max(mem_values):.0f}"
else:
    avg_mem = "NA"
    peak_mem = "NA"

best_text = f"{best_val:.4f}" if best_val is not None else "NA"
print(
    "\t".join(
        [
            avg_tok,
            max_tok,
            best_text,
            avg_util,
            max_util,
            avg_mem,
            peak_mem,
            str(len(util_values)),
        ]
    )
)
PY
}

IFS=',' read -r -a profile_items <<< "$PROFILES"
for raw_profile in "${profile_items[@]}"; do
  profile="$(echo "$raw_profile" | xargs)"
  if [[ -z "$profile" ]]; then
    continue
  fi

  IFS=':' read -r context batch grad_accum <<< "$profile"
  if [[ -z "${context:-}" || -z "${batch:-}" || -z "${grad_accum:-}" ]]; then
    echo "error: invalid --profiles entry '$profile' (expected context:batch:grad_accum)" >&2
    exit 2
  fi
  if ! [[ "$context" =~ ^[0-9]+$ && "$batch" =~ ^[0-9]+$ && "$grad_accum" =~ ^[0-9]+$ ]]; then
    echo "error: non-integer profile entry '$profile'" >&2
    exit 2
  fi
  if [[ "$context" -le 0 || "$batch" -le 0 || "$grad_accum" -le 0 ]]; then
    echo "error: profile values must be > 0 ('$profile')" >&2
    exit 2
  fi

  run_slug="ctx${context}_b${batch}_ga${grad_accum}"
  run_dir="$OUT_ROOT/$run_slug"
  log_path="$run_dir/train.log"
  gpu_csv="$run_dir/gpu_samples.csv"
  mkdir -p "$run_dir"

  echo "[ctx-bench] start profile=$run_slug run_dir=$run_dir"

  sampler_pid=""
  if [[ "$has_nvidia_smi" -eq 1 ]]; then
    : > "$gpu_csv"
    (
      while true; do
        ts="$(date +%s)"
        line="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
        if [[ -n "$line" ]]; then
          util="$(echo "$line" | cut -d',' -f1 | tr -d ' ')"
          mem="$(echo "$line" | cut -d',' -f2 | tr -d ' ')"
          echo "$ts,$util,$mem" >> "$gpu_csv"
        fi
        sleep "$SAMPLE_SECONDS"
      done
    ) &
    sampler_pid=$!
  fi

  cmd=(
    .venv/bin/python -u -m llm.cli train
    --shards-path "$SHARDS_PATH"
    --output-dir "$run_dir"
    --device "$DEVICE"
    --max-steps "$MAX_STEPS"
    --batch-size "$batch"
    --grad-accum-steps "$grad_accum"
    --context-length "$context"
    --n-layers "$N_LAYERS"
    --n-heads "$N_HEADS"
    --d-model "$D_MODEL"
    --learning-rate "$LEARNING_RATE"
    --lr-schedule cosine
    --lr-warmup-steps "$LR_WARMUP_STEPS"
    --lr-min-ratio "$LR_MIN_RATIO"
    --eval-interval "$EVAL_INTERVAL"
    --eval-steps "$EVAL_STEPS"
    --log-interval "$LOG_INTERVAL"
    --precision "$PRECISION"
    --checkpoint-keep-last 1
    --checkpoint-keep-every 0
  )

  if [[ "$COMPILE_MODEL" -eq 1 ]]; then
    cmd+=(--compile-model)
  fi

  set +e
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=src \
  "${cmd[@]}" > "$log_path" 2>&1
  rc=$?
  set -e

  if [[ -n "$sampler_pid" ]]; then
    kill "$sampler_pid" >/dev/null 2>&1 || true
    wait "$sampler_pid" 2>/dev/null || true
  fi

  metric_line="$(parse_metrics "$log_path" "$gpu_csv")"
  IFS=$'\t' read -r avg_tok max_tok best_val avg_util max_util avg_mem peak_mem gpu_samples <<< "$metric_line"
  effective_batch=$((batch * grad_accum))

  echo -e "${run_slug}\t${context}\t${batch}\t${grad_accum}\t${effective_batch}\t${avg_tok}\t${max_tok}\t${best_val}\t${avg_util}\t${max_util}\t${avg_mem}\t${peak_mem}\t${gpu_samples}\t${rc}\t${run_dir}" >> "$SUMMARY_TSV"

  echo "[ctx-bench] done profile=$run_slug rc=$rc avg_toks_per_sec=$avg_tok max_toks_per_sec=$max_tok peak_gpu_mem_mib=$peak_mem"
done

echo "[ctx-bench] summary=$SUMMARY_TSV"

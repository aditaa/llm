#!/usr/bin/env bash
set -euo pipefail

# Auto-resume supervisor for staged FineWeb 350BT training on RTX 5070.
# Runs training in step chunks and resumes from last checkpoint each cycle.
# Each chunk restart re-reads shard manifests so newly built batches are included.
# Optional features:
# - GPU telemetry + simple batch/grad_accum auto-tuning between chunks
# - automatic post-chunk eval + trend files

SHARDS_PATH="data/shards_global/fineweb-global-bpe-v1"
OUTPUT_DIR="artifacts/checkpoints/fineweb-350bt-bpe-v2-run1"
STATE_DIR="artifacts/reports/train_supervisor_350bt"

POLL_SECONDS=120
STEP_CHUNK=2000
MIN_MANIFESTS=1
MAX_FAILURE_STREAK=0  # 0 = unlimited retries

DEVICE="cuda"
BATCH_SIZE=34
GRAD_ACCUM_STEPS=1
TARGET_EFFECTIVE_BATCH=34
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

AUTO_TUNE=1
BATCH_STEP=2
MIN_BATCH_SIZE=8
MAX_BATCH_SIZE=40
UTIL_LOW_PCT=85
MEM_LOW_WATER_MIB=10000
MEM_HIGH_WATER_MIB=11800
GPU_SAMPLE_SECONDS=2

EVAL_AFTER_CHUNK=1
EVAL_SUITE="configs/eval/standard_prompt_suite_v2.json"
EVAL_MAX_NEW_TOKENS=120
EVAL_TEMPERATURE="0.2"
EVAL_TOP_K=0
EVAL_SEED=42
EVAL_SEED_STRIDE=97
EVAL_DEVICE="auto"
EVAL_PROMOTION_POLICY="configs/eval/promotion_policy_v1.json"
EVAL_FAIL_ON_REGRESSION=1
EVAL_MAX_PASS_RATE_DROP="0.01"
EVAL_MAX_CHECK_PASS_RATE_DROP="0.01"
EVAL_MAX_AVG_CASE_SCORE_DROP="0.01"
EVAL_FAIL_ON_NO_PROMOTION=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/train_supervisor_rtx5070_350bt.sh [options]

Core options:
  --shards-path DIR            Root containing shard manifest.json files
  --output-dir DIR             Training output directory (last.pt lives here)
  --state-dir DIR              Supervisor logs/state directory
  --poll-seconds N             Sleep between checks/restarts (default: 120)
  --step-chunk N               Steps per training cycle before restart (default: 2000)
  --min-manifests N            Wait until at least N manifests exist (default: 1)
  --max-failure-streak N       Stop after N consecutive train failures (0 = never)

Training shape:
  --device NAME                Training device (default: cuda)
  --batch-size N               Initial batch size (default: 34)
  --grad-accum-steps N         Initial grad accumulation steps (default: 1)
  --target-effective-batch N   Auto recompute grad_accum=ceil(target/batch) (default: 34)
  --context-length N           Context length (default: 512)
  --n-layers N                 Transformer layer count (default: 12)
  --n-heads N                  Attention head count (default: 12)
  --d-model N                  Hidden size (default: 768)
  --learning-rate X            Learning rate (default: 3e-4)
  --lr-warmup-steps N          Warmup steps (default: 2000)
  --lr-min-ratio X             Cosine min ratio (default: 0.10)
  --eval-interval N            Train-loop eval interval (default: 1000)
  --eval-steps N               Train-loop eval steps (default: 6)
  --log-interval N             Train log interval (default: 100)
  --precision MODE             Precision mode (default: auto)

Auto-tune options:
  --no-auto-tune               Disable automatic batch tuning
  --batch-step N               Batch step size when tuning (default: 2)
  --min-batch-size N           Lower batch bound (default: 8)
  --max-batch-size N           Upper batch bound (default: 40)
  --util-low-pct N             If avg GPU util < threshold and memory headroom exists, increase batch
  --mem-low-water-mib N        Memory threshold for safe batch increase (default: 10000)
  --mem-high-water-mib N       Memory threshold for forced batch decrease (default: 11800)
  --gpu-sample-seconds N       nvidia-smi sample period during training (default: 2)

Post-chunk eval:
  --no-eval-after-chunk        Disable checkpoint prompt-suite eval after each successful chunk
  --eval-suite FILE            Eval suite JSON (default: configs/eval/standard_prompt_suite_v1.json)
  --eval-max-new-tokens N      Eval max new tokens per case (default: 120)
  --eval-temperature X         Eval sampling temperature (default: 0.2)
  --eval-top-k N               Eval top-k (default: 0)
  --eval-seed N                Eval base seed (default: 42)
  --eval-seed-stride N         Eval seed stride (default: 97)
  --eval-device NAME           Eval device (default: auto)
  --eval-promotion-policy FILE Eval promotion policy JSON (default: configs/eval/promotion_policy_v1.json)
  --no-eval-fail-on-regression Disable eval-script regression fail flag
  --eval-max-pass-rate-drop X  Allowed pass_rate drop vs baseline (default: 0.01)
  --eval-max-check-pass-rate-drop X
                               Allowed check_pass_rate drop vs baseline (default: 0.01)
  --eval-max-avg-case-score-drop X
                               Allowed avg_case_score drop vs baseline (default: 0.01)
  --eval-fail-on-no-promotion  Fail eval step when promotion policy criteria are not met
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
    --grad-accum-steps) GRAD_ACCUM_STEPS="$2"; shift 2 ;;
    --target-effective-batch) TARGET_EFFECTIVE_BATCH="$2"; shift 2 ;;
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
    --no-auto-tune) AUTO_TUNE=0; shift ;;
    --batch-step) BATCH_STEP="$2"; shift 2 ;;
    --min-batch-size) MIN_BATCH_SIZE="$2"; shift 2 ;;
    --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
    --util-low-pct) UTIL_LOW_PCT="$2"; shift 2 ;;
    --mem-low-water-mib) MEM_LOW_WATER_MIB="$2"; shift 2 ;;
    --mem-high-water-mib) MEM_HIGH_WATER_MIB="$2"; shift 2 ;;
    --gpu-sample-seconds) GPU_SAMPLE_SECONDS="$2"; shift 2 ;;
    --no-eval-after-chunk) EVAL_AFTER_CHUNK=0; shift ;;
    --eval-suite) EVAL_SUITE="$2"; shift 2 ;;
    --eval-max-new-tokens) EVAL_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --eval-temperature) EVAL_TEMPERATURE="$2"; shift 2 ;;
    --eval-top-k) EVAL_TOP_K="$2"; shift 2 ;;
    --eval-seed) EVAL_SEED="$2"; shift 2 ;;
    --eval-seed-stride) EVAL_SEED_STRIDE="$2"; shift 2 ;;
    --eval-device) EVAL_DEVICE="$2"; shift 2 ;;
    --eval-promotion-policy) EVAL_PROMOTION_POLICY="$2"; shift 2 ;;
    --no-eval-fail-on-regression) EVAL_FAIL_ON_REGRESSION=0; shift ;;
    --eval-max-pass-rate-drop) EVAL_MAX_PASS_RATE_DROP="$2"; shift 2 ;;
    --eval-max-check-pass-rate-drop) EVAL_MAX_CHECK_PASS_RATE_DROP="$2"; shift 2 ;;
    --eval-max-avg-case-score-drop) EVAL_MAX_AVG_CASE_SCORE_DROP="$2"; shift 2 ;;
    --eval-fail-on-no-promotion) EVAL_FAIL_ON_NO_PROMOTION=1; shift ;;
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
PYTHON_BIN=".venv/bin/python"
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
if [[ "$BATCH_SIZE" -le 0 || "$GRAD_ACCUM_STEPS" -le 0 ]]; then
  echo "error: batch-size and grad-accum-steps must be > 0" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$STATE_DIR" artifacts/reports/evals

LOCK_FILE="$STATE_DIR/supervisor.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "error: another train supervisor instance is already running" >&2
  exit 1
fi

SUP_LOG="$STATE_DIR/supervisor_$(date +%Y%m%d_%H%M%S).log"
TRAIN_TREND_TSV="$STATE_DIR/train_trend.tsv"
EVAL_TREND_TSV="$STATE_DIR/eval_trend.tsv"
touch "$SUP_LOG"

if [[ ! -f "$TRAIN_TREND_TSV" ]]; then
  echo -e "run_tag\tstep_start\tstep_target\tstep_end\trc\tmanifests\tbatch_size\tgrad_accum\tbest_val_ppl\tgpu_avg_util\tgpu_max_mem_mib\ttrain_log" > "$TRAIN_TREND_TSV"
fi
if [[ ! -f "$EVAL_TREND_TSV" ]]; then
  echo -e "run_tag\tstep\teval_rc\tpass_rate\tcheck_pass_rate\tavg_case_score\tcases_passed\tcases_total\tregression_pass\tpromotion_pass\tfailed_checks\tbaseline_report\treport_json" > "$EVAL_TREND_TSV"
fi

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$SUP_LOG"
}

ceil_div() {
  local a="$1"
  local b="$2"
  echo $(( (a + b - 1) / b ))
}

set_grad_accum_from_target() {
  if [[ "$TARGET_EFFECTIVE_BATCH" -gt 0 ]]; then
    GRAD_ACCUM_STEPS="$(ceil_div "$TARGET_EFFECTIVE_BATCH" "$BATCH_SIZE")"
    if [[ "$GRAD_ACCUM_STEPS" -lt 1 ]]; then
      GRAD_ACCUM_STEPS=1
    fi
  fi
}

clamp_batch_size() {
  if [[ "$BATCH_SIZE" -lt "$MIN_BATCH_SIZE" ]]; then
    BATCH_SIZE="$MIN_BATCH_SIZE"
  fi
  if [[ "$BATCH_SIZE" -gt "$MAX_BATCH_SIZE" ]]; then
    BATCH_SIZE="$MAX_BATCH_SIZE"
  fi
}

current_step() {
  local ckpt="$OUTPUT_DIR/last.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "0"
    return 0
  fi
  "$PYTHON_BIN" - <<'PY' "$ckpt"
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt.get("step", 0)))
PY
}

manifest_count() {
  find "$SHARDS_PATH" -name manifest.json 2>/dev/null | wc -l | tr -d ' '
}

start_gpu_monitor() {
  local train_pid="$1"
  local monitor_file="$2"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  (
    echo "ts,util,mem_used_mib"
    while kill -0 "$train_pid" 2>/dev/null; do
      local row
      row="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
      if [[ -n "$row" ]]; then
        echo "$row" | awk -F',' -v ts="$(date +%s)" '{gsub(/ /,"",$1); gsub(/ /,"",$2); print ts "," $1 "," $2}'
      fi
      sleep "$GPU_SAMPLE_SECONDS"
    done
  ) > "$monitor_file" 2>/dev/null &
  echo "$!"
}

gpu_summary() {
  local monitor_file="$1"
  if [[ ! -f "$monitor_file" ]]; then
    echo "NA NA"
    return 0
  fi
  awk -F',' '
    NR > 1 {
      util_sum += $2
      n += 1
      if ($3 > mem_max) mem_max = $3
    }
    END {
      if (n > 0) {
        printf("%.2f %d\n", util_sum / n, mem_max)
      } else {
        printf("NA NA\n")
      }
    }
  ' "$monitor_file"
}

best_val_ppl_from_log() {
  local run_log="$1"
  local v
  v="$(grep -oE 'best_val_ppl=[0-9]+(\.[0-9]+)?' "$run_log" | tail -n 1 | cut -d= -f2 || true)"
  if [[ -z "${v:-}" ]]; then
    echo "NA"
  else
    echo "$v"
  fi
}

auto_tune_after_chunk() {
  local rc="$1"
  local run_log="$2"
  local gpu_avg_util="$3"
  local gpu_max_mem="$4"
  local reason="none"
  local changed=0

  if [[ "$rc" -ne 0 ]]; then
    if grep -Eqi 'out of memory|cuda.*out of memory|cublas.*alloc' "$run_log"; then
      if [[ "$BATCH_SIZE" -gt "$MIN_BATCH_SIZE" ]]; then
        BATCH_SIZE=$((BATCH_SIZE - BATCH_STEP))
        reason="oom_decrease"
        changed=1
      fi
    fi
  elif [[ "$AUTO_TUNE" -eq 1 ]]; then
    if [[ "$gpu_max_mem" != "NA" && "$gpu_max_mem" -ge "$MEM_HIGH_WATER_MIB" && "$BATCH_SIZE" -gt "$MIN_BATCH_SIZE" ]]; then
      BATCH_SIZE=$((BATCH_SIZE - BATCH_STEP))
      reason="mem_high_decrease"
      changed=1
    elif [[ "$gpu_avg_util" != "NA" && "$gpu_max_mem" != "NA" ]]; then
      if awk "BEGIN {exit !($gpu_avg_util < $UTIL_LOW_PCT)}"; then
        if [[ "$gpu_max_mem" -le "$MEM_LOW_WATER_MIB" && "$BATCH_SIZE" -lt "$MAX_BATCH_SIZE" ]]; then
          BATCH_SIZE=$((BATCH_SIZE + BATCH_STEP))
          reason="util_low_increase"
          changed=1
        fi
      fi
    fi
  fi

  clamp_batch_size
  if [[ "$TARGET_EFFECTIVE_BATCH" -gt 0 ]]; then
    set_grad_accum_from_target
  fi

  log "auto_tune rc=$rc changed=$changed reason=$reason batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS gpu_avg_util=$gpu_avg_util gpu_max_mem=$gpu_max_mem"
}

run_post_chunk_eval() {
  local run_tag="$1"
  local step="$2"
  if [[ "$EVAL_AFTER_CHUNK" -ne 1 ]]; then
    return 0
  fi
  if [[ ! -f "$OUTPUT_DIR/last.pt" ]]; then
    log "eval_skip reason=no_last_checkpoint"
    return 0
  fi

  local eval_report="artifacts/reports/evals/supervisor_350bt_step$(printf '%07d' "$step")_${run_tag}.json"
  local eval_log="$STATE_DIR/eval_${step}_${run_tag}.log"
  local baseline_report=""
  if [[ -f "$EVAL_TREND_TSV" ]]; then
    baseline_report="$(awk -F'\t' 'NR>1 && $3=="0" {print $NF}' "$EVAL_TREND_TSV" | tail -n 1)"
  fi
  if [[ -n "$baseline_report" && ! -f "$baseline_report" ]]; then
    baseline_report=""
  fi
  log "eval_start step=$step suite=$EVAL_SUITE report=$eval_report baseline=${baseline_report:-none}"

  local -a eval_extra_args=()
  if [[ -n "$baseline_report" ]]; then
    eval_extra_args+=(--baseline-report "$baseline_report")
    eval_extra_args+=(--max-pass-rate-drop "$EVAL_MAX_PASS_RATE_DROP")
    eval_extra_args+=(--max-check-pass-rate-drop "$EVAL_MAX_CHECK_PASS_RATE_DROP")
    eval_extra_args+=(--max-avg-case-score-drop "$EVAL_MAX_AVG_CASE_SCORE_DROP")
    if [[ "$EVAL_FAIL_ON_REGRESSION" -eq 1 ]]; then
      eval_extra_args+=(--fail-on-regression)
    fi
  fi
  if [[ -n "$EVAL_PROMOTION_POLICY" && -f "$EVAL_PROMOTION_POLICY" ]]; then
    eval_extra_args+=(--promotion-policy "$EVAL_PROMOTION_POLICY")
    if [[ "$EVAL_FAIL_ON_NO_PROMOTION" -eq 1 ]]; then
      eval_extra_args+=(--fail-on-no-promotion)
    fi
  fi

  set +e
  PYTHONPATH=src \
  .venv/bin/python scripts/eval_checkpoint_prompts.py \
    --checkpoint "$OUTPUT_DIR/last.pt" \
    --suite "$EVAL_SUITE" \
    --output "$eval_report" \
    --device "$EVAL_DEVICE" \
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
    --temperature "$EVAL_TEMPERATURE" \
    --top-k "$EVAL_TOP_K" \
    --seed "$EVAL_SEED" \
    --seed-stride "$EVAL_SEED_STRIDE" \
    "${eval_extra_args[@]}" \
    > "$eval_log" 2>&1
  local eval_rc=$?
  set -e

  local pass_rate="NA"
  local check_pass_rate="NA"
  local avg_case_score="NA"
  local cases_passed="NA"
  local cases_total="NA"
  local regression_pass="NA"
  local promotion_pass="NA"
  local failed_checks="NA"
  if [[ -f "$eval_report" ]]; then
    read -r pass_rate check_pass_rate avg_case_score cases_passed cases_total regression_pass promotion_pass failed_checks < <(
      python3 - <<'PY' "$eval_report"
import json
import sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
s = obj.get("summary", {})
r = obj.get("regression", {})
p = obj.get("promotion", {})
failed = p.get("failed_checks", []) if isinstance(p, dict) else []
print(
    s.get("pass_rate", "NA"),
    s.get("check_pass_rate", "NA"),
    s.get("avg_case_score", "NA"),
    s.get("cases_passed", "NA"),
    s.get("cases_total", "NA"),
    r.get("pass", "NA") if isinstance(r, dict) else "NA",
    p.get("promoted", "NA") if isinstance(p, dict) else "NA",
    ",".join(str(x) for x in failed) if failed else "none",
)
PY
    )
  fi

  echo -e "${run_tag}\t${step}\t${eval_rc}\t${pass_rate}\t${check_pass_rate}\t${avg_case_score}\t${cases_passed}\t${cases_total}\t${regression_pass}\t${promotion_pass}\t${failed_checks}\t${baseline_report:-NA}\t${eval_report}" >> "$EVAL_TREND_TSV"
  log "eval_done rc=$eval_rc pass_rate=$pass_rate check_pass_rate=$check_pass_rate avg_case_score=$avg_case_score regression_pass=$regression_pass promotion_pass=$promotion_pass baseline=${baseline_report:-none} report=$eval_report"
}

clamp_batch_size
if [[ "$TARGET_EFFECTIVE_BATCH" -gt 0 ]]; then
  set_grad_accum_from_target
fi

failure_streak=0
log "supervisor_start shards_path=$SHARDS_PATH output_dir=$OUTPUT_DIR step_chunk=$STEP_CHUNK"
log "tuning_start batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS auto_tune=$AUTO_TUNE target_effective_batch=$TARGET_EFFECTIVE_BATCH"

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
  gpu_log="$STATE_DIR/gpu_${step_now}_to_${target_step}_${run_tag}.csv"
  resume_args=()
  if [[ -f "$OUTPUT_DIR/last.pt" ]]; then
    resume_args=(--resume-from "$OUTPUT_DIR/last.pt")
  fi

  log "train_launch manifests=$mcount step_now=$step_now target_step=$target_step batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS run_log=$run_log"

  set +e
  (
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=src \
    .venv/bin/python -u -m llm.cli train \
      --shards-path "$SHARDS_PATH" \
      --output-dir "$OUTPUT_DIR" \
      --device "$DEVICE" \
      --max-steps "$target_step" \
      --batch-size "$BATCH_SIZE" \
      --context-length "$CONTEXT_LENGTH" \
      --grad-accum-steps "$GRAD_ACCUM_STEPS" \
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
      "${resume_args[@]}"
  ) > >(tee -a "$run_log") 2>&1 &
  train_pid=$!
  monitor_pid="$(start_gpu_monitor "$train_pid" "$gpu_log")"
  wait "$train_pid"
  rc=$?
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  set -e

  new_step="$(current_step)"
  best_val_ppl="$(best_val_ppl_from_log "$run_log")"
  read -r gpu_avg_util gpu_max_mem < <(gpu_summary "$gpu_log")
  echo -e "${run_tag}\t${step_now}\t${target_step}\t${new_step}\t${rc}\t${mcount}\t${BATCH_SIZE}\t${GRAD_ACCUM_STEPS}\t${best_val_ppl}\t${gpu_avg_util}\t${gpu_max_mem}\t${run_log}" >> "$TRAIN_TREND_TSV"

  if [[ "$rc" -eq 0 ]]; then
    failure_streak=0
    log "train_done rc=0 step_now=$new_step best_val_ppl=$best_val_ppl gpu_avg_util=$gpu_avg_util gpu_max_mem=$gpu_max_mem"
    run_post_chunk_eval "$run_tag" "$new_step"
  else
    failure_streak=$((failure_streak + 1))
    log "train_failed rc=$rc failure_streak=$failure_streak best_val_ppl=$best_val_ppl gpu_avg_util=$gpu_avg_util gpu_max_mem=$gpu_max_mem"
    if [[ "$MAX_FAILURE_STREAK" -gt 0 && "$failure_streak" -ge "$MAX_FAILURE_STREAK" ]]; then
      log "supervisor_stop reason=max_failure_streak_reached"
      exit 10
    fi
  fi

  auto_tune_after_chunk "$rc" "$run_log" "$gpu_avg_util" "$gpu_max_mem"
  sleep "$POLL_SECONDS"
done

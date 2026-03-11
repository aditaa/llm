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
MIN_UNIQUE_INPUT_FILES=0
MIN_TRAIN_TOKENS=0
MAX_FAILURE_STREAK=0  # 0 = unlimited retries
TRAIN_STALL_CHECK_SECONDS=60
TRAIN_STALL_KILL_SECONDS=1200

DEVICE="cuda"
BATCH_SIZE=34
GRAD_ACCUM_STEPS=1
TARGET_EFFECTIVE_BATCH=34
CONTEXT_LENGTH=512
ALLOW_CONTEXT_EXTENSION=0
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
TRAIN_FAIL_ON_EVAL_REGRESSION=1
CHECKPOINT_KEEP_LAST=6
CHECKPOINT_KEEP_EVERY=10000
EMA_DECAY="0.0"
EMA_UPDATE_EVERY=1
EMA_START_STEP=0

AUTO_TUNE=1
BATCH_STEP=2
MIN_BATCH_SIZE=8
MAX_BATCH_SIZE=40
UTIL_LOW_PCT=85
MEM_LOW_WATER_MIB=10000
MEM_HIGH_WATER_MIB=11800
GPU_SAMPLE_SECONDS=2

EVAL_AFTER_CHUNK=1
EVAL_SUITE="configs/eval/standard_prompt_suite_v3.json"
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
RENDER_EVAL_DASHBOARD=1

GENERATION_GATE=1
GENERATION_SUITE="configs/eval/generation_smoke_suite_v1.json"
GENERATION_MAX_NEW_TOKENS=120
GENERATION_TEMPERATURE="0.8"
GENERATION_TOP_K=40
GENERATION_SEED=314
GENERATION_SEED_STRIDE=31
GENERATION_DEVICE="auto"
GENERATION_FAIL_ON_REGRESSION=1
GENERATION_MAX_PASS_RATE_DROP="0.02"
GENERATION_MAX_CHECK_PASS_RATE_DROP="0.02"
GENERATION_MAX_AVG_CASE_SCORE_DROP="0.02"
GENERATION_FAIL_BELOW_PASS_RATE=""
GENERATION_EVERY_CHUNKS=1
GENERATION_STOP_ON_FAIL=0

HOLDOUT_GATE=0
HOLDOUT_SUITE=""
HOLDOUT_MAX_NEW_TOKENS=120
HOLDOUT_TEMPERATURE="0.2"
HOLDOUT_TOP_K=1
HOLDOUT_SEED=2718
HOLDOUT_SEED_STRIDE=53
HOLDOUT_DEVICE="auto"
HOLDOUT_FAIL_ON_REGRESSION=1
HOLDOUT_MAX_PASS_RATE_DROP="0.01"
HOLDOUT_MAX_CHECK_PASS_RATE_DROP="0.01"
HOLDOUT_MAX_AVG_CASE_SCORE_DROP="0.01"
HOLDOUT_FAIL_BELOW_PASS_RATE=""
HOLDOUT_EVERY_CHUNKS=1
HOLDOUT_STOP_ON_FAIL=0

PROMOTION_REQUIRE_POLICY_PASS=1
PROMOTION_REQUIRE_GENERATION_PASS=1
PROMOTION_REQUIRE_HOLDOUT_PASS=1
PROMOTION_MIN_QUALITY_STREAK=2

QUALITY_ROLLBACK_STREAK=3
QUALITY_ROLLBACK_COOLDOWN_STEPS=4000

DEDUPE_OVERLAP_MANIFESTS=1
DEDUPE_KEEP="newest"
DEDUPE_DRY_RUN=0
DEDUPE_REPORT_KEEP=240

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
  --min-unique-input-files N   Wait until at least N unique manifest input files exist (default: 0)
  --min-train-tokens N         Wait until manifests cover at least N train tokens (default: 0)
  --max-failure-streak N       Stop after N consecutive train failures (0 = never)
  --train-stall-check-seconds N
                               Poll interval for train-step stall detection (default: 60)
  --train-stall-kill-seconds N
                               Restart train chunk if no step progress for N seconds (0 = disabled, default: 1200)
  --no-dedupe-overlap-manifests
                               Disable manifest overlap dedupe before each train chunk
  --dedupe-keep MODE           Overlap dedupe strategy: newest|oldest (default: newest)
  --dedupe-dry-run             Analyze overlap only; do not disable duplicate manifests
  --dedupe-report-keep N       Keep latest N manifest dedupe report/log files (default: 240)

Training shape:
  --device NAME                Training device (default: cuda)
  --batch-size N               Initial batch size (default: 34)
  --grad-accum-steps N         Initial grad accumulation steps (default: 1)
  --target-effective-batch N   Auto recompute grad_accum=ceil(target/batch) (default: 34)
  --context-length N           Context length (default: 512)
  --allow-context-extension    Allow resume from shorter-context checkpoint to larger context
  --n-layers N                 Transformer layer count (default: 12)
  --n-heads N                  Attention head count (default: 12)
  --d-model N                  Hidden size (default: 768)
  --learning-rate X            Learning rate (default: 3e-4)
  --lr-warmup-steps N          Warmup steps (default: 2000)
  --lr-min-ratio X             Cosine min ratio (default: 0.10)
  --eval-interval N            Train-loop eval interval (default: 1000)
  --eval-steps N               Train-loop eval steps (default: 6)
  --log-interval N             Train log interval (default: 100)
  --no-train-fail-on-eval-regression
                               Disable train-loop held-out eval regression gate
  --precision MODE             Precision mode (default: auto)
  --checkpoint-keep-last N     Keep last N step checkpoints (default: 6)
  --checkpoint-keep-every N    Keep every Nth checkpoint step (default: 10000)
  --ema-decay X                EMA decay for model weights (default: 0.0 disabled)
  --ema-update-every N         EMA update interval in optimizer steps (default: 1)
  --ema-start-step N           First optimizer step to apply EMA updates (default: 0)

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
  --eval-suite FILE            Eval suite JSON (default: configs/eval/standard_prompt_suite_v3.json)
                               Baselines auto-match the active suite (name/path) from eval trend history
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
  --no-render-eval-dashboard   Disable HTML+JSON eval trend dashboard rendering

Generation gate (scheduled post-chunk prompt generation checks):
  --no-generation-gate         Disable post-chunk generation gate
  --generation-suite FILE      Generation suite JSON (default: configs/eval/generation_smoke_suite_v1.json)
                               Baselines auto-match the active suite (name/path) from generation trend history
  --generation-max-new-tokens N
                               Generation gate max new tokens per case (default: 120)
  --generation-temperature X   Generation gate sampling temperature (default: 0.8)
  --generation-top-k N         Generation gate top-k (default: 40)
  --generation-seed N          Generation gate base seed (default: 314)
  --generation-seed-stride N   Generation gate seed stride (default: 31)
  --generation-device NAME     Generation gate device (default: auto)
  --no-generation-fail-on-regression
                               Disable generation gate regression fail flag
  --generation-max-pass-rate-drop X
                               Allowed generation pass_rate drop vs baseline (default: 0.02)
  --generation-max-check-pass-rate-drop X
                               Allowed generation check_pass_rate drop vs baseline (default: 0.02)
  --generation-max-avg-case-score-drop X
                               Allowed generation avg_case_score drop vs baseline (default: 0.02)
  --generation-fail-below-pass-rate X
                               Fail generation gate if pass_rate drops below X
  --generation-every-chunks N  Run generation gate every N successful chunks (default: 1)
  --generation-stop-on-fail    Stop supervisor when generation gate returns non-zero

Fixed holdout gate (frozen quality suite):
  --no-holdout-gate            Disable fixed holdout gate
  --holdout-suite FILE         Holdout suite JSON path (enables holdout gate)
  --holdout-max-new-tokens N   Holdout max new tokens per case (default: 120)
  --holdout-temperature X      Holdout sampling temperature (default: 0.2)
  --holdout-top-k N            Holdout top-k (default: 1)
  --holdout-seed N             Holdout base seed (default: 2718)
  --holdout-seed-stride N      Holdout seed stride (default: 53)
  --holdout-device NAME        Holdout device (default: auto)
  --no-holdout-fail-on-regression
                               Disable holdout regression fail flag
  --holdout-max-pass-rate-drop X
                               Allowed holdout pass_rate drop vs fixed baseline (default: 0.01)
  --holdout-max-check-pass-rate-drop X
                               Allowed holdout check_pass_rate drop vs fixed baseline (default: 0.01)
  --holdout-max-avg-case-score-drop X
                               Allowed holdout avg_case_score drop vs fixed baseline (default: 0.01)
  --holdout-fail-below-pass-rate X
                               Fail holdout gate if pass_rate drops below X
  --holdout-every-chunks N     Run holdout gate every N successful chunks (default: 1)
  --holdout-stop-on-fail       Stop supervisor when holdout gate returns non-zero

Promotion discipline:
  --no-promotion-require-policy-pass
                               Allow best promotion without eval policy promotion flag
  --no-promotion-require-generation-pass
                               Allow best promotion without passing generation gate
  --no-promotion-require-holdout-pass
                               Allow best promotion without passing holdout gate
  --promotion-min-quality-streak N
                               Require N consecutive quality-passing chunks before promotion
                               (default: 2)

Quality rollback:
  --quality-rollback-streak N  Roll back to best checkpoint after N consecutive failed
                               quality chunks (eval/gen gate regressions). 0 disables.
                               (default: 3)
  --quality-rollback-cooldown-steps N
                               Minimum training-step gap between auto-rollbacks
                               (default: 4000)
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
    --min-unique-input-files) MIN_UNIQUE_INPUT_FILES="$2"; shift 2 ;;
    --min-train-tokens) MIN_TRAIN_TOKENS="$2"; shift 2 ;;
    --max-failure-streak) MAX_FAILURE_STREAK="$2"; shift 2 ;;
    --train-stall-check-seconds) TRAIN_STALL_CHECK_SECONDS="$2"; shift 2 ;;
    --train-stall-kill-seconds) TRAIN_STALL_KILL_SECONDS="$2"; shift 2 ;;
    --no-dedupe-overlap-manifests) DEDUPE_OVERLAP_MANIFESTS=0; shift ;;
    --dedupe-keep) DEDUPE_KEEP="$2"; shift 2 ;;
    --dedupe-dry-run) DEDUPE_DRY_RUN=1; shift ;;
    --dedupe-report-keep) DEDUPE_REPORT_KEEP="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum-steps) GRAD_ACCUM_STEPS="$2"; shift 2 ;;
    --target-effective-batch) TARGET_EFFECTIVE_BATCH="$2"; shift 2 ;;
    --context-length) CONTEXT_LENGTH="$2"; shift 2 ;;
    --allow-context-extension) ALLOW_CONTEXT_EXTENSION=1; shift ;;
    --n-layers) N_LAYERS="$2"; shift 2 ;;
    --n-heads) N_HEADS="$2"; shift 2 ;;
    --d-model) D_MODEL="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --lr-warmup-steps) LR_WARMUP_STEPS="$2"; shift 2 ;;
    --lr-min-ratio) LR_MIN_RATIO="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --eval-steps) EVAL_STEPS="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --no-train-fail-on-eval-regression) TRAIN_FAIL_ON_EVAL_REGRESSION=0; shift ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --checkpoint-keep-last) CHECKPOINT_KEEP_LAST="$2"; shift 2 ;;
    --checkpoint-keep-every) CHECKPOINT_KEEP_EVERY="$2"; shift 2 ;;
    --ema-decay) EMA_DECAY="$2"; shift 2 ;;
    --ema-update-every) EMA_UPDATE_EVERY="$2"; shift 2 ;;
    --ema-start-step) EMA_START_STEP="$2"; shift 2 ;;
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
    --no-render-eval-dashboard) RENDER_EVAL_DASHBOARD=0; shift ;;
    --no-generation-gate) GENERATION_GATE=0; shift ;;
    --generation-suite) GENERATION_SUITE="$2"; shift 2 ;;
    --generation-max-new-tokens) GENERATION_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --generation-temperature) GENERATION_TEMPERATURE="$2"; shift 2 ;;
    --generation-top-k) GENERATION_TOP_K="$2"; shift 2 ;;
    --generation-seed) GENERATION_SEED="$2"; shift 2 ;;
    --generation-seed-stride) GENERATION_SEED_STRIDE="$2"; shift 2 ;;
    --generation-device) GENERATION_DEVICE="$2"; shift 2 ;;
    --no-generation-fail-on-regression) GENERATION_FAIL_ON_REGRESSION=0; shift ;;
    --generation-max-pass-rate-drop) GENERATION_MAX_PASS_RATE_DROP="$2"; shift 2 ;;
    --generation-max-check-pass-rate-drop) GENERATION_MAX_CHECK_PASS_RATE_DROP="$2"; shift 2 ;;
    --generation-max-avg-case-score-drop) GENERATION_MAX_AVG_CASE_SCORE_DROP="$2"; shift 2 ;;
    --generation-fail-below-pass-rate) GENERATION_FAIL_BELOW_PASS_RATE="$2"; shift 2 ;;
    --generation-every-chunks) GENERATION_EVERY_CHUNKS="$2"; shift 2 ;;
    --generation-stop-on-fail) GENERATION_STOP_ON_FAIL=1; shift ;;
    --no-holdout-gate) HOLDOUT_GATE=0; shift ;;
    --holdout-suite) HOLDOUT_SUITE="$2"; HOLDOUT_GATE=1; shift 2 ;;
    --holdout-max-new-tokens) HOLDOUT_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --holdout-temperature) HOLDOUT_TEMPERATURE="$2"; shift 2 ;;
    --holdout-top-k) HOLDOUT_TOP_K="$2"; shift 2 ;;
    --holdout-seed) HOLDOUT_SEED="$2"; shift 2 ;;
    --holdout-seed-stride) HOLDOUT_SEED_STRIDE="$2"; shift 2 ;;
    --holdout-device) HOLDOUT_DEVICE="$2"; shift 2 ;;
    --no-holdout-fail-on-regression) HOLDOUT_FAIL_ON_REGRESSION=0; shift ;;
    --holdout-max-pass-rate-drop) HOLDOUT_MAX_PASS_RATE_DROP="$2"; shift 2 ;;
    --holdout-max-check-pass-rate-drop) HOLDOUT_MAX_CHECK_PASS_RATE_DROP="$2"; shift 2 ;;
    --holdout-max-avg-case-score-drop) HOLDOUT_MAX_AVG_CASE_SCORE_DROP="$2"; shift 2 ;;
    --holdout-fail-below-pass-rate) HOLDOUT_FAIL_BELOW_PASS_RATE="$2"; shift 2 ;;
    --holdout-every-chunks) HOLDOUT_EVERY_CHUNKS="$2"; shift 2 ;;
    --holdout-stop-on-fail) HOLDOUT_STOP_ON_FAIL=1; shift ;;
    --no-promotion-require-policy-pass) PROMOTION_REQUIRE_POLICY_PASS=0; shift ;;
    --no-promotion-require-generation-pass) PROMOTION_REQUIRE_GENERATION_PASS=0; shift ;;
    --no-promotion-require-holdout-pass) PROMOTION_REQUIRE_HOLDOUT_PASS=0; shift ;;
    --promotion-min-quality-streak) PROMOTION_MIN_QUALITY_STREAK="$2"; shift 2 ;;
    --quality-rollback-streak) QUALITY_ROLLBACK_STREAK="$2"; shift 2 ;;
    --quality-rollback-cooldown-steps) QUALITY_ROLLBACK_COOLDOWN_STEPS="$2"; shift 2 ;;
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
if ! [[ "$MIN_MANIFESTS" =~ ^[0-9]+$ ]] || ! [[ "$MIN_UNIQUE_INPUT_FILES" =~ ^[0-9]+$ ]] || ! [[ "$MIN_TRAIN_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "error: min-manifests, min-unique-input-files, and min-train-tokens must be integers >= 0" >&2
  exit 1
fi
if [[ "$MIN_MANIFESTS" -lt 0 || "$MIN_UNIQUE_INPUT_FILES" -lt 0 || "$MIN_TRAIN_TOKENS" -lt 0 ]]; then
  echo "error: min-manifests, min-unique-input-files, and min-train-tokens must be >= 0" >&2
  exit 1
fi
if [[ "$POLL_SECONDS" -le 0 ]]; then
  echo "error: poll-seconds must be > 0" >&2
  exit 1
fi
if [[ "$TRAIN_STALL_CHECK_SECONDS" -le 0 ]]; then
  echo "error: train-stall-check-seconds must be > 0" >&2
  exit 1
fi
if [[ "$TRAIN_STALL_KILL_SECONDS" -lt 0 ]]; then
  echo "error: train-stall-kill-seconds must be >= 0" >&2
  exit 1
fi
if [[ "$BATCH_SIZE" -le 0 || "$GRAD_ACCUM_STEPS" -le 0 ]]; then
  echo "error: batch-size and grad-accum-steps must be > 0" >&2
  exit 1
fi
if [[ "$CHECKPOINT_KEEP_LAST" -lt 0 || "$CHECKPOINT_KEEP_EVERY" -lt 0 ]]; then
  echo "error: checkpoint retention values must be >= 0" >&2
  exit 1
fi
if [[ "$GENERATION_EVERY_CHUNKS" -le 0 ]]; then
  echo "error: generation-every-chunks must be > 0" >&2
  exit 1
fi
if [[ "$HOLDOUT_GATE" -eq 1 && -z "$HOLDOUT_SUITE" ]]; then
  echo "error: holdout-suite must be set when holdout gate is enabled" >&2
  exit 1
fi
if [[ "$HOLDOUT_GATE" -eq 1 && ! -f "$HOLDOUT_SUITE" ]]; then
  echo "error: holdout-suite not found: $HOLDOUT_SUITE" >&2
  exit 1
fi
if [[ "$HOLDOUT_EVERY_CHUNKS" -le 0 ]]; then
  echo "error: holdout-every-chunks must be > 0" >&2
  exit 1
fi
if ! [[ "$QUALITY_ROLLBACK_STREAK" =~ ^[0-9]+$ ]] || ! [[ "$QUALITY_ROLLBACK_COOLDOWN_STEPS" =~ ^[0-9]+$ ]]; then
  echo "error: quality rollback values must be integers >= 0" >&2
  exit 1
fi
if ! [[ "$PROMOTION_MIN_QUALITY_STREAK" =~ ^[0-9]+$ ]]; then
  echo "error: promotion-min-quality-streak must be an integer >= 0" >&2
  exit 1
fi
if [[ "$DEDUPE_KEEP" != "newest" && "$DEDUPE_KEEP" != "oldest" ]]; then
  echo "error: dedupe-keep must be one of: newest, oldest" >&2
  exit 1
fi
if ! [[ "$DEDUPE_REPORT_KEEP" =~ ^[0-9]+$ ]]; then
  echo "error: dedupe-report-keep must be an integer >= 0" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$STATE_DIR" artifacts/reports/evals

LOCK_FILE="$STATE_DIR/supervisor.lock"
if ! command -v flock >/dev/null 2>&1; then
  echo "error: required command not found: flock" >&2
  exit 1
fi
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "error: another train_supervisor instance is already running ($LOCK_FILE)" >&2
  exit 3
fi

SUP_LOG="$STATE_DIR/supervisor_$(date +%Y%m%d_%H%M%S).log"
TRAIN_TREND_TSV="$STATE_DIR/train_trend.tsv"
EVAL_TREND_TSV="$STATE_DIR/eval_trend.tsv"
GENERATION_TREND_TSV="$STATE_DIR/generation_trend.tsv"
HOLDOUT_TREND_TSV="$STATE_DIR/holdout_trend.tsv"
HOLDOUT_BASELINE_FILE="$STATE_DIR/holdout_baseline_report.txt"
BEST_META_JSON="$STATE_DIR/best_checkpoint.json"
TRAINED_BATCHES_FILE="$STATE_DIR/trained_batch_names.txt"
LAST_QUALITY_ROLLBACK_STEP_FILE="$STATE_DIR/last_quality_rollback_step.txt"
touch "$SUP_LOG"
touch "$TRAINED_BATCHES_FILE"

if [[ ! -f "$TRAIN_TREND_TSV" ]]; then
  echo -e "run_tag\tstep_start\tstep_target\tstep_end\trc\tmanifests\tbatch_size\tgrad_accum\tbest_val_ppl\tgpu_avg_util\tgpu_max_mem_mib\tsampled_batches\tsampled_trace\ttrain_log" > "$TRAIN_TREND_TSV"
fi
if [[ ! -f "$EVAL_TREND_TSV" ]]; then
  echo -e "run_tag\tstep\teval_rc\tpass_rate\tcheck_pass_rate\tavg_case_score\tcases_passed\tcases_total\tregression_pass\tpromotion_pass\tfailed_checks\tbaseline_report\treport_json" > "$EVAL_TREND_TSV"
fi
if [[ ! -f "$GENERATION_TREND_TSV" ]]; then
  echo -e "run_tag\tstep\tgeneration_rc\tpass_rate\tcheck_pass_rate\tavg_case_score\tcases_passed\tcases_total\tregression_pass\tbaseline_report\treport_json" > "$GENERATION_TREND_TSV"
fi
if [[ ! -f "$HOLDOUT_TREND_TSV" ]]; then
  echo -e "run_tag\tstep\tholdout_rc\tpass_rate\tcheck_pass_rate\tavg_case_score\tcases_passed\tcases_total\tregression_pass\tbaseline_report\treport_json" > "$HOLDOUT_TREND_TSV"
fi

LAST_EVAL_RAN=0
LAST_EVAL_RC=0
LAST_EVAL_REPORT=""
LAST_EVAL_REGRESSION_PASS="NA"
LAST_EVAL_PROMOTION_PASS="NA"
LAST_GENERATION_GATE_RAN=0
LAST_GENERATION_GATE_RC=0
LAST_HOLDOUT_GATE_RAN=0
LAST_HOLDOUT_GATE_RC=0
QUALITY_PASS_STREAK=0

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$SUP_LOG"
}

find_latest_successful_baseline_for_suite() {
  local trend_tsv="$1"
  local suite_json="$2"
  if [[ ! -f "$trend_tsv" || ! -f "$suite_json" ]]; then
    echo ""
    return 0
  fi
  "$PYTHON_BIN" - <<'PY' "$trend_tsv" "$suite_json"
import json
import sys
from pathlib import Path

trend_tsv = Path(sys.argv[1])
suite_json = Path(sys.argv[2])

try:
    suite_obj = json.loads(suite_json.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

target_name = str(suite_obj.get("name", "")).strip()
try:
    target_resolved = suite_json.resolve()
except OSError:
    target_resolved = suite_json
target_basename = suite_json.name

rows: list[list[str]] = []
for line in trend_tsv.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line or line.startswith("run_tag\t"):
        continue
    rows.append(line.split("\t"))

for parts in reversed(rows):
    if len(parts) < 4:
        continue
    rc = parts[2].strip()
    report_path = parts[-1].strip()
    if rc != "0" or not report_path or report_path == "NA":
        continue
    report_file = Path(report_path)
    if not report_file.is_file():
        continue
    try:
        report_obj = json.loads(report_file.read_text(encoding="utf-8"))
    except Exception:
        continue

    report_suite_name = str(report_obj.get("suite_name", "")).strip()
    report_suite_path_raw = str(report_obj.get("suite_path", "")).strip()

    name_match = bool(target_name and report_suite_name and report_suite_name == target_name)
    path_match = False
    if report_suite_path_raw:
        report_suite_path = Path(report_suite_path_raw)
        if report_suite_path.name == target_basename:
            path_match = True
        else:
            try:
                if report_suite_path.resolve() == target_resolved:
                    path_match = True
            except OSError:
                path_match = False

    if name_match or path_match:
        print(str(report_file))
        raise SystemExit(0)

print("")
PY
}

is_truthy() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

fixed_holdout_baseline_report() {
  if [[ ! -f "$HOLDOUT_BASELINE_FILE" ]]; then
    echo ""
    return 0
  fi
  local report
  report="$(head -n 1 "$HOLDOUT_BASELINE_FILE" 2>/dev/null || true)"
  if [[ -n "$report" && -f "$report" ]]; then
    echo "$report"
    return 0
  fi
  echo ""
}

set_holdout_baseline_report() {
  local report="$1"
  if [[ -z "$report" || ! -f "$report" ]]; then
    return 0
  fi
  printf '%s\n' "$report" > "$HOLDOUT_BASELINE_FILE"
}

find_oldest_supervisor_pid() {
  ps -eo pid=,ppid=,etimes=,args= | awk -v target_state="$STATE_DIR" -v self_pid="$$" '
{
  pid = $1
  ppid = $2
  etimes = $3
  $1 = ""; $2 = ""; $3 = ""
  sub(/^ +/, "", $0)
  cmd = $0
  if (pid == self_pid || ppid == self_pid) {
    next
  }
  if (cmd !~ /^bash scripts\/train_supervisor_rtx5070_350bt\.sh( |$)/) {
    next
  }
  process_state = "artifacts/reports/train_supervisor_350bt"
  arg_count = split(cmd, args, /[[:space:]]+/)
  for (i = 1; i <= arg_count; i++) {
    if (args[i] == "--state-dir" && (i + 1) <= arg_count) {
      process_state = args[i + 1]
      break
    }
  }
  if (process_state != target_state) {
    next
  }
  if (best_pid == "" || etimes > best_etime || (etimes == best_etime && pid < best_pid)) {
    best_pid = pid
    best_etime = etimes
  }
}
END {
  if (best_pid != "") {
    print best_pid
    exit 0
  }
  exit 1
}'
}

ensure_single_supervisor_process() {
  local oldest_pid
  oldest_pid="$(find_oldest_supervisor_pid || true)"
  if [[ -z "$oldest_pid" ]]; then
    return 0
  fi
  if [[ "$oldest_pid" != "$$" ]]; then
    log "singleton_exit reason=older_supervisor_running same_state_dir=$STATE_DIR self=$$ keeper=$oldest_pid"
    exit 0
  fi
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

checkpoint_step() {
  local ckpt="$1"
  "$PYTHON_BIN" - <<'PY' "$ckpt"
import sys
import torch
obj = torch.load(sys.argv[1], map_location="cpu")
step = obj.get("step")
if not isinstance(step, int):
    raise SystemExit(2)
print(step)
PY
}

checkpoint_is_valid_for_resume() {
  local ckpt="$1"
  "$PYTHON_BIN" - <<'PY' "$ckpt"
import sys
import torch
obj = torch.load(sys.argv[1], map_location="cpu")
required = ["step", "model_state", "optimizer_state", "model_config", "tokenizer_path"]
for key in required:
    if key not in obj:
        raise SystemExit(2)
if not isinstance(obj.get("step"), int):
    raise SystemExit(2)
if not isinstance(obj.get("model_state"), dict):
    raise SystemExit(2)
if not isinstance(obj.get("optimizer_state"), dict):
    raise SystemExit(2)
if not isinstance(obj.get("model_config"), dict):
    raise SystemExit(2)
if not isinstance(obj.get("tokenizer_path"), str):
    raise SystemExit(2)
print("ok")
PY
}

quarantine_bad_checkpoint() {
  local ckpt="$1"
  local reason="$2"
  if [[ ! -f "$ckpt" ]]; then
    return 0
  fi
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local bad_path="${ckpt}.bad_${ts}"
  mv "$ckpt" "$bad_path"
  log "checkpoint_quarantined reason=$reason old=$ckpt new=$bad_path"
}

select_resume_checkpoint() {
  local candidates=()
  if [[ -f "$OUTPUT_DIR/last.pt" ]]; then
    candidates+=("$OUTPUT_DIR/last.pt")
  fi
  while IFS= read -r path; do
    candidates+=("$path")
  done < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'ckpt_step_*.pt' | sort -r)

  local ckpt
  for ckpt in "${candidates[@]}"; do
    if checkpoint_is_valid_for_resume "$ckpt" >/dev/null 2>&1; then
      echo "$ckpt"
      return 0
    fi
    quarantine_bad_checkpoint "$ckpt" "invalid_resume"
  done
  echo ""
}

log_has_resume_checkpoint_error() {
  local run_log="$1"
  grep -Eqi \
    "pytorchstreamreader|failed finding central directory|pickle data was truncated|unexpected eof|zip archive|invalid load key|optimizer_state|resume checkpoint|load_state_dict" \
    "$run_log"
}

latest_step_from_log() {
  local run_log="$1"
  if [[ ! -f "$run_log" ]]; then
    echo "0"
    return 0
  fi
  local step
  step="$(
    grep -oE 'step=[0-9]+' "$run_log" | tail -n 1 | cut -d= -f2 || true
  )"
  if [[ -z "$step" ]]; then
    echo "0"
    return 0
  fi
  echo "$step"
}

wait_for_train_with_stall_guard() {
  local train_pid="$1"
  local run_log="$2"
  local start_step="$3"
  local stall_flag_file="$4"
  : > "$stall_flag_file"

  local last_step="$start_step"
  local last_progress_epoch
  last_progress_epoch="$(date +%s)"

  while kill -0 "$train_pid" 2>/dev/null; do
    sleep "$TRAIN_STALL_CHECK_SECONDS"
    if ! kill -0 "$train_pid" 2>/dev/null; then
      break
    fi
    local observed_step
    observed_step="$(latest_step_from_log "$run_log")"
    if [[ "$observed_step" =~ ^[0-9]+$ ]] && [[ "$observed_step" -gt "$last_step" ]]; then
      last_step="$observed_step"
      last_progress_epoch="$(date +%s)"
      continue
    fi
    if [[ "$TRAIN_STALL_KILL_SECONDS" -le 0 ]]; then
      continue
    fi
    local now_epoch
    now_epoch="$(date +%s)"
    local stalled_for=$((now_epoch - last_progress_epoch))
    if [[ "$stalled_for" -lt "$TRAIN_STALL_KILL_SECONDS" ]]; then
      continue
    fi

    log "train_stall_detected stalled_for=${stalled_for}s kill_after=${TRAIN_STALL_KILL_SECONDS}s last_step=$last_step action=terminate_chunk"
    echo "1" > "$stall_flag_file"
    kill -TERM "$train_pid" 2>/dev/null || true
    sleep 5
    if kill -0 "$train_pid" 2>/dev/null; then
      kill -KILL "$train_pid" 2>/dev/null || true
    fi
    break
  done

  wait "$train_pid"
  return $?
}

manifest_count() {
  find "$SHARDS_PATH" -name manifest.json 2>/dev/null | wc -l | tr -d ' '
}

collect_manifest_batch_names() {
  "$PYTHON_BIN" - <<'PY' "$SHARDS_PATH"
from pathlib import Path
import sys

root = Path(sys.argv[1])
names: list[str] = []
for manifest_path in sorted(root.rglob("manifest.json")):
    names.append(manifest_path.parent.name)

for name in names:
    print(name)
PY
}

collect_sampled_batch_names() {
  local sampled_trace_json="$1"
  "$PYTHON_BIN" - <<'PY' "$sampled_trace_json"
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
if not trace_path.is_file():
    raise SystemExit(0)

try:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

rows = payload.get("sampled_shards", [])
if not isinstance(rows, list):
    raise SystemExit(0)

batches: set[str] = set()
for row in rows:
    if not isinstance(row, dict):
        continue
    raw = row.get("path")
    if not isinstance(raw, str) or not raw.strip():
        continue
    batch = Path(raw).resolve().parent.name
    if batch:
        batches.add(batch)

for batch in sorted(batches):
    print(batch)
PY
}

update_trained_batch_registry() {
  local sampled_batches_file="$1"
  local step="$2"
  local chunk_batches_count="$3"
  if [[ ! -f "$sampled_batches_file" ]]; then
    log "trained_batches_skip step=$step reason=missing_sampled_batches_file file=$sampled_batches_file"
    return 0
  fi
  local sampled_count
  sampled_count="$(wc -l < "$sampled_batches_file" | tr -d ' ')"
  if [[ "$sampled_count" -le 0 ]]; then
    log "trained_batches_skip step=$step reason=empty_sampled_batches sampled_file=$sampled_batches_file chunk_batches=$chunk_batches_count"
    return 0
  fi
  cat "$sampled_batches_file" >> "$TRAINED_BATCHES_FILE"
  sort -u -o "$TRAINED_BATCHES_FILE" "$TRAINED_BATCHES_FILE"
  local trained_count
  trained_count="$(wc -l < "$TRAINED_BATCHES_FILE" | tr -d ' ')"
  log "trained_batches_update step=$step sampled_batches=$sampled_count chunk_batches=$chunk_batches_count trained_batches=$trained_count registry=$TRAINED_BATCHES_FILE sampled_file=$sampled_batches_file"
}

backfill_trained_batch_registry() {
  "$PYTHON_BIN" - <<'PY' "$TRAIN_TREND_TSV" "$STATE_DIR" "$TRAINED_BATCHES_FILE" >> "$SUP_LOG" 2>&1
from pathlib import Path
import sys

trend_path = Path(sys.argv[1])
state_dir = Path(sys.argv[2])
registry_path = Path(sys.argv[3])

existing: set[str] = set()
if registry_path.exists():
    for line in registry_path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = line.strip()
        if value:
            existing.add(value)

added = 0
if trend_path.exists():
    for row in trend_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not row or row.startswith("run_tag\t"):
            continue
        parts = row.split("\t")
        if len(parts) < 5:
            continue
        run_tag = parts[0].strip()
        rc = parts[4].strip()
        if not run_tag or rc != "0":
            continue
        for sampled_file in sorted(state_dir.glob(f"sampled_batches_*_{run_tag}.txt")):
            for line in sampled_file.read_text(encoding="utf-8", errors="replace").splitlines():
                batch = line.strip()
                if not batch or batch in existing:
                    continue
                existing.add(batch)
                added += 1

registry_path.parent.mkdir(parents=True, exist_ok=True)
registry_path.write_text(
    "".join(f"{name}\n" for name in sorted(existing)),
    encoding="utf-8",
)
print(
    f"trained_batches_backfill added={added} total={len(existing)} "
    f"registry={registry_path}"
)
PY
}

manifest_coverage_counts() {
  "$PYTHON_BIN" - <<'PY' "$SHARDS_PATH"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
file_to_manifests: dict[str, set[str]] = {}
manifest_sets: list[set[str]] = []
train_tokens = 0

for manifest_path in sorted(root.rglob("manifest.json")):
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    raw = payload.get("input_files", [])
    if not isinstance(raw, list):
        continue
    names = {Path(str(item)).name for item in raw if str(item).strip()}
    if not names:
        continue
    manifest_sets.append(names)
    mref = str(manifest_path)
    for name in names:
        refs = file_to_manifests.setdefault(name, set())
        refs.add(mref)
    train_meta = payload.get("train")
    if isinstance(train_meta, dict):
        total = train_meta.get("total_tokens")
        if isinstance(total, int):
            train_tokens += total
            continue
        shards = train_meta.get("shards")
        if isinstance(shards, list):
            for shard in shards:
                if isinstance(shard, dict):
                    token_count = shard.get("tokens")
                    if isinstance(token_count, int):
                        train_tokens += token_count

overlap_inputs = sum(1 for refs in file_to_manifests.values() if len(refs) > 1)
overlap_manifests = sum(
    1 for names in manifest_sets if any(len(file_to_manifests.get(name, ())) > 1 for name in names)
)
print(len(file_to_manifests), overlap_inputs, overlap_manifests, train_tokens)
PY
}

run_hot_manifest_guard() {
  local guard_report="$STATE_DIR/hot_manifest_guard_latest.json"
  local guard_out
  set +e
  guard_out="$(
    PYTHONPATH=src \
    "$PYTHON_BIN" scripts/enforce_hot_only_manifests.py \
      --shards-root "$SHARDS_PATH" \
      --report-output "$guard_report" \
      2>&1
  )"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "hot_manifest_guard_failed rc=$rc detail=$(echo "$guard_out" | tr '\n' ' ')"
    return 0
  fi
  log "$guard_out"
}

run_manifest_dedupe() {
  if [[ "$DEDUPE_OVERLAP_MANIFESTS" -ne 1 ]]; then
    return 0
  fi

  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local dedupe_report="$STATE_DIR/manifest_dedupe_${ts}.json"
  local dedupe_log="$STATE_DIR/manifest_dedupe_${ts}.log"

  local -a extra_args=()
  if [[ "$DEDUPE_DRY_RUN" -eq 1 ]]; then
    extra_args+=(--dry-run)
  fi

  set +e
  PYTHONPATH=src \
  .venv/bin/python scripts/fineweb_manifest_dedupe.py \
    --shards-root "$SHARDS_PATH" \
    --report-output "$dedupe_report" \
    --keep "$DEDUPE_KEEP" \
    "${extra_args[@]}" \
    > "$dedupe_log" 2>&1
  local rc=$?
  set -e

  if [[ "$rc" -ne 0 ]]; then
    log "manifest_dedupe_failed rc=$rc log=$dedupe_log"
    return 0
  fi

  local summary
  summary="$(
    python3 - <<'PY' "$dedupe_report"
import json
import sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(
    obj.get("manifest_total", 0),
    obj.get("manifest_kept", 0),
    obj.get("manifest_overlap", 0),
    obj.get("unique_input_files", 0),
    len(obj.get("disabled", [])),
)
PY
  )"
  local total kept overlap unique disabled
  read -r total kept overlap unique disabled <<< "$summary"
  log "manifest_dedupe_done total=$total kept=$kept overlap=$overlap unique_inputs=$unique disabled=$disabled dry_run=$DEDUPE_DRY_RUN report=$dedupe_report"
}

prune_manifest_dedupe_artifacts() {
  local keep="$DEDUPE_REPORT_KEEP"
  if [[ "$keep" -le 0 ]]; then
    return
  fi

  "$PYTHON_BIN" - <<'PY' "$STATE_DIR" "$keep" >> "$SUP_LOG" 2>&1
from pathlib import Path
import sys

state_dir = Path(sys.argv[1])
keep = int(sys.argv[2])
patterns = ("manifest_dedupe_*.json", "manifest_dedupe_*.log")
deleted = 0

for pattern in patterns:
    files = sorted(
        state_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for extra in files[keep:]:
        try:
            extra.unlink()
            deleted += 1
        except OSError:
            pass

if deleted:
    print(f"manifest_dedupe_pruned deleted={deleted} keep={keep}")
PY
}

start_gpu_monitor() {
  local train_pid="$1"
  local monitor_file="$2"
  GPU_MONITOR_PID=""
  if ! command -v nvidia-smi >/dev/null 2>&1; then
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
  GPU_MONITOR_PID="$!"
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

promote_best_checkpoint_if_needed() {
  local step="$1"
  local eval_rc="$2"
  local eval_report="$3"
  if [[ "$eval_rc" -ne 0 ]]; then
    return 0
  fi
  if [[ ! -f "$eval_report" || ! -f "$OUTPUT_DIR/last.pt" ]]; then
    return 0
  fi

  local promoted=0
  local metric_value="0.0"
  local metric_name="pass_rate"
  if [[ -f "$BEST_META_JSON" ]]; then
    read -r promoted metric_value metric_name < <(
      python3 - <<'PY' "$eval_report" "$BEST_META_JSON"
import json
import sys

report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
best = json.load(open(sys.argv[2], "r", encoding="utf-8"))
summary = report.get("summary", {})
promotion = report.get("promotion", {})
is_promoted = bool(promotion.get("promoted")) if isinstance(promotion, dict) else False

if is_promoted:
    print(1, summary.get("pass_rate", 0.0), "policy")
    raise SystemExit(0)

current_pass = float(summary.get("pass_rate", 0.0))
best_pass = float(best.get("pass_rate", 0.0))
if current_pass >= best_pass:
    print(1, current_pass, "pass_rate")
else:
    print(0, current_pass, "pass_rate")
PY
    )
  else
    read -r promoted metric_value metric_name < <(
      python3 - <<'PY' "$eval_report"
import json
import sys

report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
summary = report.get("summary", {})
promotion = report.get("promotion", {})
is_promoted = bool(promotion.get("promoted")) if isinstance(promotion, dict) else False
if is_promoted:
    print(1, summary.get("pass_rate", 0.0), "policy")
else:
    print(1, summary.get("pass_rate", 0.0), "pass_rate")
PY
    )
  fi

  if [[ "$promoted" -ne 1 ]]; then
    log "best_skip step=$step reason=not_better metric=$metric_name value=$metric_value"
    return 0
  fi

  cp -f "$OUTPUT_DIR/last.pt" "$OUTPUT_DIR/best.pt"
  if [[ -f "$OUTPUT_DIR/last.safetensors" ]]; then
    cp -f "$OUTPUT_DIR/last.safetensors" "$OUTPUT_DIR/best.safetensors"
  fi
  if [[ -f "$OUTPUT_DIR/last_ema.safetensors" ]]; then
    cp -f "$OUTPUT_DIR/last_ema.safetensors" "$OUTPUT_DIR/best_ema.safetensors"
  fi
  cp -f "$eval_report" "$OUTPUT_DIR/best_eval_report.json"

  python3 - <<'PY' "$BEST_META_JSON" "$step" "$eval_report" "$metric_name" "$metric_value"
import json
import sys
from datetime import datetime, timezone

meta_path = sys.argv[1]
payload = {
    "best_step": int(sys.argv[2]),
    "best_eval_report": sys.argv[3],
    "metric_name": sys.argv[4],
    "pass_rate": float(sys.argv[5]),
    "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
}
with open(meta_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY
  log "best_promoted step=$step metric=$metric_name value=$metric_value checkpoint=$OUTPUT_DIR/best.pt"
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
  LAST_EVAL_RAN=0
  LAST_EVAL_RC=0
  LAST_EVAL_REPORT=""
  LAST_EVAL_REGRESSION_PASS="NA"
  LAST_EVAL_PROMOTION_PASS="NA"
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
  baseline_report="$(find_latest_successful_baseline_for_suite "$EVAL_TREND_TSV" "$EVAL_SUITE")"
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
  LAST_EVAL_RAN=1
  LAST_EVAL_RC="$eval_rc"
  LAST_EVAL_REPORT="$eval_report"
  LAST_EVAL_REGRESSION_PASS="$regression_pass"
  LAST_EVAL_PROMOTION_PASS="$promotion_pass"

  echo -e "${run_tag}\t${step}\t${eval_rc}\t${pass_rate}\t${check_pass_rate}\t${avg_case_score}\t${cases_passed}\t${cases_total}\t${regression_pass}\t${promotion_pass}\t${failed_checks}\t${baseline_report:-NA}\t${eval_report}" >> "$EVAL_TREND_TSV"
  log "eval_done rc=$eval_rc pass_rate=$pass_rate check_pass_rate=$check_pass_rate avg_case_score=$avg_case_score regression_pass=$regression_pass promotion_pass=$promotion_pass baseline=${baseline_report:-none} report=$eval_report"
  if [[ "$RENDER_EVAL_DASHBOARD" -eq 1 ]]; then
    if PYTHONPATH=src .venv/bin/python scripts/render_eval_trend_dashboard.py \
      --input-tsv "$EVAL_TREND_TSV" \
      --output-html "$STATE_DIR/eval_dashboard.html" \
      --output-json "$STATE_DIR/eval_dashboard_summary.json" \
      >> "$eval_log" 2>&1; then
      log "eval_dashboard_updated html=$STATE_DIR/eval_dashboard.html"
    else
      log "eval_dashboard_update_failed"
    fi
  fi
}

run_generation_gate() {
  local run_tag="$1"
  local step="$2"
  local successful_chunks="$3"
  LAST_GENERATION_GATE_RAN=0
  LAST_GENERATION_GATE_RC=0
  if [[ "$GENERATION_GATE" -ne 1 ]]; then
    return 0
  fi
  if (( successful_chunks % GENERATION_EVERY_CHUNKS != 0 )); then
    log "generation_gate_skip reason=interval step=$step successful_chunks=$successful_chunks interval=$GENERATION_EVERY_CHUNKS"
    return 0
  fi
  if [[ ! -f "$OUTPUT_DIR/last.pt" ]]; then
    log "generation_gate_skip reason=no_last_checkpoint"
    return 0
  fi

  local gen_report="artifacts/reports/evals/gen_gate_step$(printf '%07d' "$step")_${run_tag}.json"
  local gen_log="$STATE_DIR/generation_gate_${step}_${run_tag}.log"
  local baseline_report=""
  baseline_report="$(find_latest_successful_baseline_for_suite "$GENERATION_TREND_TSV" "$GENERATION_SUITE")"
  if [[ -n "$baseline_report" && ! -f "$baseline_report" ]]; then
    baseline_report=""
  fi
  log "generation_gate_start step=$step suite=$GENERATION_SUITE report=$gen_report baseline=${baseline_report:-none}"

  local -a gen_extra_args=()
  if [[ -n "$baseline_report" ]]; then
    gen_extra_args+=(--baseline-report "$baseline_report")
    gen_extra_args+=(--max-pass-rate-drop "$GENERATION_MAX_PASS_RATE_DROP")
    gen_extra_args+=(--max-check-pass-rate-drop "$GENERATION_MAX_CHECK_PASS_RATE_DROP")
    gen_extra_args+=(--max-avg-case-score-drop "$GENERATION_MAX_AVG_CASE_SCORE_DROP")
    if [[ "$GENERATION_FAIL_ON_REGRESSION" -eq 1 ]]; then
      gen_extra_args+=(--fail-on-regression)
    fi
  fi
  if [[ -n "$GENERATION_FAIL_BELOW_PASS_RATE" ]]; then
    gen_extra_args+=(--fail-below-pass-rate "$GENERATION_FAIL_BELOW_PASS_RATE")
  fi

  set +e
  PYTHONPATH=src \
  .venv/bin/python scripts/eval_checkpoint_prompts.py \
    --checkpoint "$OUTPUT_DIR/last.pt" \
    --suite "$GENERATION_SUITE" \
    --output "$gen_report" \
    --device "$GENERATION_DEVICE" \
    --max-new-tokens "$GENERATION_MAX_NEW_TOKENS" \
    --temperature "$GENERATION_TEMPERATURE" \
    --top-k "$GENERATION_TOP_K" \
    --seed "$GENERATION_SEED" \
    --seed-stride "$GENERATION_SEED_STRIDE" \
    "${gen_extra_args[@]}" \
    > "$gen_log" 2>&1
  local gen_rc=$?
  set -e
  LAST_GENERATION_GATE_RAN=1
  LAST_GENERATION_GATE_RC="$gen_rc"

  local pass_rate="NA"
  local check_pass_rate="NA"
  local avg_case_score="NA"
  local cases_passed="NA"
  local cases_total="NA"
  local regression_pass="NA"
  if [[ -f "$gen_report" ]]; then
    read -r pass_rate check_pass_rate avg_case_score cases_passed cases_total regression_pass < <(
      python3 - <<'PY' "$gen_report"
import json
import sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
s = obj.get("summary", {})
r = obj.get("regression", {})
print(
    s.get("pass_rate", "NA"),
    s.get("check_pass_rate", "NA"),
    s.get("avg_case_score", "NA"),
    s.get("cases_passed", "NA"),
    s.get("cases_total", "NA"),
    r.get("pass", "NA") if isinstance(r, dict) else "NA",
)
PY
    )
  fi

  echo -e "${run_tag}\t${step}\t${gen_rc}\t${pass_rate}\t${check_pass_rate}\t${avg_case_score}\t${cases_passed}\t${cases_total}\t${regression_pass}\t${baseline_report:-NA}\t${gen_report}" >> "$GENERATION_TREND_TSV"
  log "generation_gate_done rc=$gen_rc pass_rate=$pass_rate check_pass_rate=$check_pass_rate avg_case_score=$avg_case_score regression_pass=$regression_pass baseline=${baseline_report:-none} report=$gen_report"
  return "$gen_rc"
}

run_holdout_gate() {
  local run_tag="$1"
  local step="$2"
  local successful_chunks="$3"
  LAST_HOLDOUT_GATE_RAN=0
  LAST_HOLDOUT_GATE_RC=0
  if [[ "$HOLDOUT_GATE" -ne 1 ]]; then
    return 0
  fi
  if (( successful_chunks % HOLDOUT_EVERY_CHUNKS != 0 )); then
    log "holdout_gate_skip reason=interval step=$step successful_chunks=$successful_chunks interval=$HOLDOUT_EVERY_CHUNKS"
    return 0
  fi
  if [[ ! -f "$OUTPUT_DIR/last.pt" ]]; then
    log "holdout_gate_skip reason=no_last_checkpoint"
    return 0
  fi

  local holdout_report="artifacts/reports/evals/holdout_gate_step$(printf '%07d' "$step")_${run_tag}.json"
  local holdout_log="$STATE_DIR/holdout_gate_${step}_${run_tag}.log"
  local baseline_report=""
  baseline_report="$(fixed_holdout_baseline_report)"
  if [[ -n "$baseline_report" && ! -f "$baseline_report" ]]; then
    baseline_report=""
  fi
  log "holdout_gate_start step=$step suite=$HOLDOUT_SUITE report=$holdout_report baseline=${baseline_report:-none}"

  local -a holdout_extra_args=()
  if [[ -n "$baseline_report" ]]; then
    holdout_extra_args+=(--baseline-report "$baseline_report")
    holdout_extra_args+=(--max-pass-rate-drop "$HOLDOUT_MAX_PASS_RATE_DROP")
    holdout_extra_args+=(--max-check-pass-rate-drop "$HOLDOUT_MAX_CHECK_PASS_RATE_DROP")
    holdout_extra_args+=(--max-avg-case-score-drop "$HOLDOUT_MAX_AVG_CASE_SCORE_DROP")
    if [[ "$HOLDOUT_FAIL_ON_REGRESSION" -eq 1 ]]; then
      holdout_extra_args+=(--fail-on-regression)
    fi
  fi
  if [[ -n "$HOLDOUT_FAIL_BELOW_PASS_RATE" ]]; then
    holdout_extra_args+=(--fail-below-pass-rate "$HOLDOUT_FAIL_BELOW_PASS_RATE")
  fi

  set +e
  PYTHONPATH=src \
  .venv/bin/python scripts/eval_checkpoint_prompts.py \
    --checkpoint "$OUTPUT_DIR/last.pt" \
    --suite "$HOLDOUT_SUITE" \
    --output "$holdout_report" \
    --device "$HOLDOUT_DEVICE" \
    --max-new-tokens "$HOLDOUT_MAX_NEW_TOKENS" \
    --temperature "$HOLDOUT_TEMPERATURE" \
    --top-k "$HOLDOUT_TOP_K" \
    --seed "$HOLDOUT_SEED" \
    --seed-stride "$HOLDOUT_SEED_STRIDE" \
    "${holdout_extra_args[@]}" \
    > "$holdout_log" 2>&1
  local holdout_rc=$?
  set -e
  LAST_HOLDOUT_GATE_RAN=1
  LAST_HOLDOUT_GATE_RC="$holdout_rc"

  local pass_rate="NA"
  local check_pass_rate="NA"
  local avg_case_score="NA"
  local cases_passed="NA"
  local cases_total="NA"
  local regression_pass="NA"
  if [[ -f "$holdout_report" ]]; then
    read -r pass_rate check_pass_rate avg_case_score cases_passed cases_total regression_pass < <(
      python3 - <<'PY' "$holdout_report"
import json
import sys
obj = json.load(open(sys.argv[1], "r", encoding="utf-8"))
s = obj.get("summary", {})
r = obj.get("regression", {})
print(
    s.get("pass_rate", "NA"),
    s.get("check_pass_rate", "NA"),
    s.get("avg_case_score", "NA"),
    s.get("cases_passed", "NA"),
    s.get("cases_total", "NA"),
    r.get("pass", "NA") if isinstance(r, dict) else "NA",
)
PY
    )
  fi

  echo -e "${run_tag}\t${step}\t${holdout_rc}\t${pass_rate}\t${check_pass_rate}\t${avg_case_score}\t${cases_passed}\t${cases_total}\t${regression_pass}\t${baseline_report:-NA}\t${holdout_report}" >> "$HOLDOUT_TREND_TSV"
  log "holdout_gate_done rc=$holdout_rc pass_rate=$pass_rate check_pass_rate=$check_pass_rate avg_case_score=$avg_case_score regression_pass=$regression_pass baseline=${baseline_report:-none} report=$holdout_report"

  if [[ -z "$baseline_report" && "$holdout_rc" -eq 0 && -f "$holdout_report" ]]; then
    set_holdout_baseline_report "$holdout_report"
    log "holdout_baseline_set report=$holdout_report"
  fi
  return "$holdout_rc"
}

evaluate_quality_gates_and_maybe_promote() {
  local step="$1"
  local quality_ok=1
  local -a reasons=()

  if [[ "$LAST_EVAL_RAN" -ne 1 ]]; then
    quality_ok=0
    reasons+=("eval_not_run")
  elif [[ "$LAST_EVAL_RC" -ne 0 ]]; then
    quality_ok=0
    reasons+=("eval_failed")
  fi
  if [[ "$LAST_EVAL_REGRESSION_PASS" == "False" || "$LAST_EVAL_REGRESSION_PASS" == "0" ]]; then
    quality_ok=0
    reasons+=("eval_regressed")
  fi
  if [[ "$PROMOTION_REQUIRE_POLICY_PASS" -eq 1 ]]; then
    if ! is_truthy "$LAST_EVAL_PROMOTION_PASS"; then
      quality_ok=0
      reasons+=("policy_not_promoted")
    fi
  fi
  if [[ "$PROMOTION_REQUIRE_GENERATION_PASS" -eq 1 && "$GENERATION_GATE" -eq 1 ]]; then
    if [[ "$LAST_GENERATION_GATE_RAN" -ne 1 ]]; then
      quality_ok=0
      reasons+=("generation_not_run")
    elif [[ "$LAST_GENERATION_GATE_RC" -ne 0 ]]; then
      quality_ok=0
      reasons+=("generation_failed")
    fi
  fi
  if [[ "$PROMOTION_REQUIRE_HOLDOUT_PASS" -eq 1 && "$HOLDOUT_GATE" -eq 1 ]]; then
    if [[ "$LAST_HOLDOUT_GATE_RAN" -ne 1 ]]; then
      quality_ok=0
      reasons+=("holdout_not_run")
    elif [[ "$LAST_HOLDOUT_GATE_RC" -ne 0 ]]; then
      quality_ok=0
      reasons+=("holdout_failed")
    fi
  fi

  if [[ "$quality_ok" -eq 1 ]]; then
    QUALITY_PASS_STREAK=$((QUALITY_PASS_STREAK + 1))
  else
    QUALITY_PASS_STREAK=0
  fi

  local reason_text="none"
  if [[ "${#reasons[@]}" -gt 0 ]]; then
    reason_text="$(IFS=,; echo "${reasons[*]}")"
  fi

  log "quality_gate_result step=$step pass=$quality_ok streak=$QUALITY_PASS_STREAK required_streak=$PROMOTION_MIN_QUALITY_STREAK reasons=$reason_text"

  if [[ "$quality_ok" -ne 1 ]]; then
    log "best_skip step=$step reason=quality_gate_failed reasons=$reason_text"
    return 0
  fi
  if [[ "$PROMOTION_MIN_QUALITY_STREAK" -gt 0 && "$QUALITY_PASS_STREAK" -lt "$PROMOTION_MIN_QUALITY_STREAK" ]]; then
    log "best_skip step=$step reason=quality_streak_not_met streak=$QUALITY_PASS_STREAK required=$PROMOTION_MIN_QUALITY_STREAK"
    return 0
  fi

  promote_best_checkpoint_if_needed "$step" "$LAST_EVAL_RC" "$LAST_EVAL_REPORT"
}

quality_failure_tail_summary() {
  "$PYTHON_BIN" - <<'PY' "$EVAL_TREND_TSV" "$GENERATION_TREND_TSV" "$HOLDOUT_TREND_TSV"
from pathlib import Path
import sys

eval_path = Path(sys.argv[1])
gen_path = Path(sys.argv[2])
holdout_path = Path(sys.argv[3])

step_fail: dict[int, bool] = {}
step_sources: dict[int, set[str]] = {}

def parse_rows(path: Path, source: str, rc_idx: int, regression_idx: int) -> None:
    if not path.exists():
        return
    for row in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not row or row.startswith("run_tag\t"):
            continue
        parts = row.split("\t")
        if len(parts) <= max(rc_idx, regression_idx):
            continue
        step_raw = parts[1].strip()
        if not step_raw.isdigit():
            continue
        step = int(step_raw)
        rc = parts[rc_idx].strip()
        regression = parts[regression_idx].strip() if regression_idx >= 0 else ""
        failed = rc != "0" or regression in {"False", "0"}
        if not failed:
            step_fail.setdefault(step, False)
            continue
        step_fail[step] = True
        sources = step_sources.setdefault(step, set())
        sources.add(source)

parse_rows(eval_path, "eval", rc_idx=2, regression_idx=8)
parse_rows(gen_path, "generation", rc_idx=2, regression_idx=8)
parse_rows(holdout_path, "holdout", rc_idx=2, regression_idx=8)

if not step_fail:
    print("0\t0\tnone")
    raise SystemExit(0)

ordered_steps = sorted(step_fail.keys())
streak = 0
for step in reversed(ordered_steps):
    if step_fail.get(step):
        streak += 1
    else:
        break

latest = ordered_steps[-1]
sources = ",".join(sorted(step_sources.get(latest, set()))) or "none"
print(f"{streak}\t{latest}\t{sources}")
PY
}

best_checkpoint_step() {
  if [[ -f "$BEST_META_JSON" ]]; then
    local best_step
    best_step="$("$PYTHON_BIN" - <<'PY' "$BEST_META_JSON"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)
step = payload.get("best_step")
print(str(step) if isinstance(step, int) else "")
PY
)"
    if [[ "$best_step" =~ ^[0-9]+$ ]]; then
      echo "$best_step"
      return 0
    fi
  fi
  if [[ -f "$OUTPUT_DIR/best.pt" ]]; then
    checkpoint_step "$OUTPUT_DIR/best.pt" || echo "0"
    return 0
  fi
  echo "0"
}

maybe_auto_rollback_on_quality_regression() {
  local current_step="$1"
  if [[ "$QUALITY_ROLLBACK_STREAK" -le 0 ]]; then
    return 0
  fi
  if [[ ! -f "$OUTPUT_DIR/best.pt" ]]; then
    return 0
  fi

  local tail_streak=0
  local latest_step=0
  local latest_sources="none"
  read -r tail_streak latest_step latest_sources < <(quality_failure_tail_summary)
  if [[ "$tail_streak" -lt "$QUALITY_ROLLBACK_STREAK" ]]; then
    return 0
  fi

  local best_step
  best_step="$(best_checkpoint_step)"
  if [[ -z "$best_step" || "$best_step" -le 0 ]]; then
    log "quality_rollback_skip reason=missing_best_step streak=$tail_streak latest_step=$latest_step"
    return 0
  fi
  if [[ "$best_step" -ge "$current_step" ]]; then
    log "quality_rollback_skip reason=best_not_older current_step=$current_step best_step=$best_step streak=$tail_streak"
    return 0
  fi

  local last_rollback_step=-1
  if [[ -f "$LAST_QUALITY_ROLLBACK_STEP_FILE" ]]; then
    local raw
    raw="$(cat "$LAST_QUALITY_ROLLBACK_STEP_FILE" 2>/dev/null || true)"
    if [[ "$raw" =~ ^[0-9]+$ ]]; then
      last_rollback_step="$raw"
    fi
  fi
  if [[ "$last_rollback_step" -ge 0 ]]; then
    local delta=$((current_step - last_rollback_step))
    if [[ "$delta" -lt "$QUALITY_ROLLBACK_COOLDOWN_STEPS" ]]; then
      log "quality_rollback_skip reason=cooldown current_step=$current_step last_rollback_step=$last_rollback_step cooldown_steps=$QUALITY_ROLLBACK_COOLDOWN_STEPS streak=$tail_streak"
      return 0
    fi
  fi

  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  if [[ -f "$OUTPUT_DIR/last.pt" ]]; then
    cp -f "$OUTPUT_DIR/last.pt" "$OUTPUT_DIR/last_pre_quality_rollback_${ts}.pt"
  fi
  cp -f "$OUTPUT_DIR/best.pt" "$OUTPUT_DIR/last.pt"
  if [[ -f "$OUTPUT_DIR/best.safetensors" ]]; then
    cp -f "$OUTPUT_DIR/best.safetensors" "$OUTPUT_DIR/last.safetensors"
  fi
  if [[ -f "$OUTPUT_DIR/best_ema.safetensors" ]]; then
    cp -f "$OUTPUT_DIR/best_ema.safetensors" "$OUTPUT_DIR/last_ema.safetensors"
  fi
  echo "$current_step" > "$LAST_QUALITY_ROLLBACK_STEP_FILE"
  log "quality_rollback_applied current_step=$current_step best_step=$best_step streak=$tail_streak latest_quality_step=$latest_step latest_sources=$latest_sources cooldown_steps=$QUALITY_ROLLBACK_COOLDOWN_STEPS"
}

clamp_batch_size
if [[ "$TARGET_EFFECTIVE_BATCH" -gt 0 ]]; then
  set_grad_accum_from_target
fi

failure_streak=0
successful_chunks=0
log "supervisor_start shards_path=$SHARDS_PATH output_dir=$OUTPUT_DIR step_chunk=$STEP_CHUNK"
log "tuning_start batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS auto_tune=$AUTO_TUNE target_effective_batch=$TARGET_EFFECTIVE_BATCH train_fail_on_eval_regression=$TRAIN_FAIL_ON_EVAL_REGRESSION checkpoint_keep_last=$CHECKPOINT_KEEP_LAST checkpoint_keep_every=$CHECKPOINT_KEEP_EVERY allow_context_extension=$ALLOW_CONTEXT_EXTENSION ema_decay=$EMA_DECAY ema_update_every=$EMA_UPDATE_EVERY ema_start_step=$EMA_START_STEP generation_gate=$GENERATION_GATE generation_every_chunks=$GENERATION_EVERY_CHUNKS generation_stop_on_fail=$GENERATION_STOP_ON_FAIL holdout_gate=$HOLDOUT_GATE holdout_suite=${HOLDOUT_SUITE:-none} holdout_every_chunks=$HOLDOUT_EVERY_CHUNKS holdout_stop_on_fail=$HOLDOUT_STOP_ON_FAIL promotion_require_policy_pass=$PROMOTION_REQUIRE_POLICY_PASS promotion_require_generation_pass=$PROMOTION_REQUIRE_GENERATION_PASS promotion_require_holdout_pass=$PROMOTION_REQUIRE_HOLDOUT_PASS promotion_min_quality_streak=$PROMOTION_MIN_QUALITY_STREAK quality_rollback_streak=$QUALITY_ROLLBACK_STREAK quality_rollback_cooldown_steps=$QUALITY_ROLLBACK_COOLDOWN_STEPS dedupe_overlap_manifests=$DEDUPE_OVERLAP_MANIFESTS dedupe_keep=$DEDUPE_KEEP dedupe_dry_run=$DEDUPE_DRY_RUN min_train_tokens=$MIN_TRAIN_TOKENS train_stall_check_seconds=$TRAIN_STALL_CHECK_SECONDS train_stall_kill_seconds=$TRAIN_STALL_KILL_SECONDS"
backfill_trained_batch_registry
ensure_single_supervisor_process

while true; do
  ensure_single_supervisor_process
  run_hot_manifest_guard
  run_manifest_dedupe
  prune_manifest_dedupe_artifacts
  mcount="$(manifest_count)"
  if [[ "$mcount" -lt "$MIN_MANIFESTS" ]]; then
    log "waiting_for_manifests have=$mcount need=$MIN_MANIFESTS sleep=${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
    continue
  fi
  read -r unique_inputs overlap_inputs overlap_manifests train_tokens < <(manifest_coverage_counts)
  if [[ "$MIN_UNIQUE_INPUT_FILES" -gt 0 && "$unique_inputs" -lt "$MIN_UNIQUE_INPUT_FILES" ]]; then
    log "waiting_for_unique_inputs have=$unique_inputs need=$MIN_UNIQUE_INPUT_FILES overlap_inputs=$overlap_inputs overlap_manifests=$overlap_manifests sleep=${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
    continue
  fi
  if [[ "$MIN_TRAIN_TOKENS" -gt 0 && "$train_tokens" -lt "$MIN_TRAIN_TOKENS" ]]; then
    log "waiting_for_train_tokens have_tokens=$train_tokens need_tokens=$MIN_TRAIN_TOKENS unique_inputs=$unique_inputs overlap_inputs=$overlap_inputs overlap_manifests=$overlap_manifests sleep=${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
    continue
  fi

  resume_ckpt="$(select_resume_checkpoint)"
  if [[ -n "$resume_ckpt" ]]; then
    step_now="$(checkpoint_step "$resume_ckpt" || echo 0)"
  else
    step_now="0"
  fi
  target_step=$((step_now + STEP_CHUNK))
  run_tag="$(date +%Y%m%d_%H%M%S)"
  run_log="$STATE_DIR/train_${step_now}_to_${target_step}_${run_tag}.log"
  gpu_log="$STATE_DIR/gpu_${step_now}_to_${target_step}_${run_tag}.csv"
  chunk_batches_file="$STATE_DIR/chunk_batches_${step_now}_to_${target_step}_${run_tag}.txt"
  sampled_shards_trace="$STATE_DIR/sampled_shards_${step_now}_to_${target_step}_${run_tag}.json"
  sampled_batches_file="$STATE_DIR/sampled_batches_${step_now}_to_${target_step}_${run_tag}.txt"
  collect_manifest_batch_names > "$chunk_batches_file"
  chunk_batches_count="$(wc -l < "$chunk_batches_file" | tr -d ' ')"
  resume_args=()
  if [[ -n "$resume_ckpt" ]]; then
    resume_args=(--resume-from "$resume_ckpt")
  fi

  log "train_launch manifests=$mcount unique_inputs=$unique_inputs train_tokens=$train_tokens overlap_inputs=$overlap_inputs overlap_manifests=$overlap_manifests step_now=$step_now target_step=$target_step batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS resume=${resume_ckpt:-none} chunk_batches=$chunk_batches_count chunk_batches_file=$chunk_batches_file sampled_trace=$sampled_shards_trace run_log=$run_log"

  train_gate_args=()
  if [[ "$TRAIN_FAIL_ON_EVAL_REGRESSION" -eq 1 ]]; then
    train_gate_args+=(--fail-on-eval-regression)
    train_gate_args+=(--eval-regression-tolerance 0.20)
  fi
  if [[ "$ALLOW_CONTEXT_EXTENSION" -eq 1 ]]; then
    train_gate_args+=(--allow-context-extension)
  fi

  set +e
  stall_flag_file="$STATE_DIR/train_stall_${run_tag}.flag"
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
      "${train_gate_args[@]}" \
      --log-interval "$LOG_INTERVAL" \
      --precision "$PRECISION" \
      --checkpoint-keep-last "$CHECKPOINT_KEEP_LAST" \
      --checkpoint-keep-every "$CHECKPOINT_KEEP_EVERY" \
      --sampled-shards-trace "$sampled_shards_trace" \
      --sampled-shards-trace-min-rows 1 \
      --ema-decay "$EMA_DECAY" \
      --ema-update-every "$EMA_UPDATE_EVERY" \
      --ema-start-step "$EMA_START_STEP" \
      --export-safetensors \
      "${resume_args[@]}"
  ) > >(tee -a "$run_log") 2>&1 &
  train_pid=$!
  start_gpu_monitor "$train_pid" "$gpu_log"
  monitor_pid="${GPU_MONITOR_PID:-}"
  wait_for_train_with_stall_guard "$train_pid" "$run_log" "$step_now" "$stall_flag_file"
  rc=$?
  stalled_chunk=0
  if [[ -f "$stall_flag_file" ]] && [[ "$(cat "$stall_flag_file" 2>/dev/null || true)" == "1" ]]; then
    stalled_chunk=1
  fi
  rm -f "$stall_flag_file" || true
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  set -e

  new_resume_ckpt="$(select_resume_checkpoint)"
  if [[ -n "$new_resume_ckpt" ]]; then
    new_step="$(checkpoint_step "$new_resume_ckpt" || echo 0)"
  else
    new_step="0"
  fi
  best_val_ppl="$(best_val_ppl_from_log "$run_log")"
  read -r gpu_avg_util gpu_max_mem < <(gpu_summary "$gpu_log")
  sampled_batches_count="NA"

  if [[ "$rc" -eq 0 ]]; then
    failure_streak=0
    successful_chunks=$((successful_chunks + 1))
    collect_sampled_batch_names "$sampled_shards_trace" > "$sampled_batches_file"
    sampled_batches_count="$(wc -l < "$sampled_batches_file" | tr -d ' ')"
    log "train_done rc=0 step_now=$new_step best_val_ppl=$best_val_ppl gpu_avg_util=$gpu_avg_util gpu_max_mem=$gpu_max_mem sampled_batches=$sampled_batches_count sampled_batches_file=$sampled_batches_file"
    update_trained_batch_registry "$sampled_batches_file" "$new_step" "$chunk_batches_count"
    run_post_chunk_eval "$run_tag" "$new_step"
    if ! run_generation_gate "$run_tag" "$new_step" "$successful_chunks"; then
      log "generation_gate_failed step=$new_step successful_chunks=$successful_chunks"
      if [[ "$GENERATION_STOP_ON_FAIL" -eq 1 ]]; then
        log "supervisor_stop reason=generation_gate_failed"
        exit 11
      fi
    fi
    if ! run_holdout_gate "$run_tag" "$new_step" "$successful_chunks"; then
      log "holdout_gate_failed step=$new_step successful_chunks=$successful_chunks"
      if [[ "$HOLDOUT_STOP_ON_FAIL" -eq 1 ]]; then
        log "supervisor_stop reason=holdout_gate_failed"
        exit 12
      fi
    fi
    evaluate_quality_gates_and_maybe_promote "$new_step"
    maybe_auto_rollback_on_quality_regression "$new_step"
  else
    if [[ -n "$resume_ckpt" ]] && log_has_resume_checkpoint_error "$run_log"; then
      quarantine_bad_checkpoint "$resume_ckpt" "resume_failure"
    fi
    QUALITY_PASS_STREAK=0
    failure_streak=$((failure_streak + 1))
    if [[ "$stalled_chunk" -eq 1 ]]; then
      log "train_failed rc=$rc failure_streak=$failure_streak reason=stall_killed best_val_ppl=$best_val_ppl gpu_avg_util=$gpu_avg_util gpu_max_mem=$gpu_max_mem"
    else
      log "train_failed rc=$rc failure_streak=$failure_streak best_val_ppl=$best_val_ppl gpu_avg_util=$gpu_avg_util gpu_max_mem=$gpu_max_mem"
    fi
    if [[ "$MAX_FAILURE_STREAK" -gt 0 && "$failure_streak" -ge "$MAX_FAILURE_STREAK" ]]; then
      log "supervisor_stop reason=max_failure_streak_reached"
      exit 10
    fi
    maybe_auto_rollback_on_quality_regression "$new_step"
  fi

  auto_tune_after_chunk "$rc" "$run_log" "$gpu_avg_util" "$gpu_max_mem"
  echo -e "${run_tag}\t${step_now}\t${target_step}\t${new_step}\t${rc}\t${mcount}\t${BATCH_SIZE}\t${GRAD_ACCUM_STEPS}\t${best_val_ppl}\t${gpu_avg_util}\t${gpu_max_mem}\t${sampled_batches_count}\t${sampled_shards_trace}\t${run_log}" >> "$TRAIN_TREND_TSV"
  sleep "$POLL_SECONDS"
done

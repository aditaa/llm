#!/usr/bin/env bash
set -euo pipefail

WARM_PARQUET_DIR="/mnt/ceph/llm/data/fineweb/sample-350BT/sample/350BT"
HOT_PARQUET_DIR="data/fineweb/sample-350BT/sample/350BT"
SHARDS_ROOT="data/shards_global/fineweb-global-bpe-v1"
TOKENIZER_PATH="artifacts/tokenizer/fineweb-global-bpe-v1.json"
STATE_DIR="artifacts/reports/fineweb_stage_shard_loop"
WARM_ROOT="/mnt/ceph/llm/data"

FIELD="text"
BATCH_SIZE=8192
SHARD_SIZE_TOKENS=5000000
VAL_RATIO=0.01
SEED=42
MIN_CHARS=80
MAX_CHARS=0
MAX_ROWS_PER_FILE=0

STAGE_MAX_FILES=10
STAGE_MAX_GIB=0
STAGE_MIN_AGE_SECONDS=180
PROCESS_MAX_FILES=10

SLEEP_SECONDS=120
ITERATIONS=0
SYNC_TO_WARM=1
PURGE_HOT=1
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/fineweb_stage_shard_loop.sh [options]

Continuously roll FineWeb parquet files from warm -> hot, shard them with a
stable tokenizer, verify shard integrity, sync results back to warm storage,
and purge processed hot parquet files.

Options:
  --warm-parquet-dir DIR      Warm source parquet dir
  --hot-parquet-dir DIR       Hot local parquet dir
  --shards-root DIR           Root for new shard batch directories
  --tokenizer-path FILE       Shared tokenizer path (train once, then reuse)
  --state-dir DIR             State/log directory
  --warm-root DIR             Warm storage root (for shard/tokenizer sync)

  --stage-max-files N         Max parquet files to stage per cycle (default: 10)
  --stage-max-gib N           Max GiB to stage per cycle (default: 0 unlimited)
  --stage-min-age-seconds N   Skip recently modified source files (default: 180)
  --process-max-files N       Max staged files to shard per batch (default: 10)

  --field NAME                Parquet text field (default: text)
  --batch-size N              Parquet read batch size (default: 8192)
  --shard-size-tokens N       Tokens per output shard (default: 5000000)
  --val-ratio X               Validation ratio (default: 0.01)
  --seed N                    RNG seed (default: 42)
  --min-chars N               Drop short rows (default: 80)
  --max-chars N               Truncate row chars (default: 0 disabled)
  --max-rows-per-file N       Row cap per parquet file (default: 0 all)

  --sleep-seconds N           Idle poll sleep (default: 120)
  --iterations N              Number of successful batches (default: 0 infinite)
  --no-sync-to-warm           Skip syncing shard outputs to warm storage
  --no-purge-hot              Keep processed parquet files on hot storage
  --dry-run                   Print actions without executing shard build/deletes
  -h, --help                  Show help

Example:
  bash scripts/fineweb_stage_shard_loop.sh \
    --stage-max-files 8 \
    --process-max-files 8 \
    --sleep-seconds 90
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --warm-parquet-dir)
      WARM_PARQUET_DIR="$2"
      shift 2
      ;;
    --hot-parquet-dir)
      HOT_PARQUET_DIR="$2"
      shift 2
      ;;
    --shards-root)
      SHARDS_ROOT="$2"
      shift 2
      ;;
    --tokenizer-path)
      TOKENIZER_PATH="$2"
      shift 2
      ;;
    --state-dir)
      STATE_DIR="$2"
      shift 2
      ;;
    --warm-root)
      WARM_ROOT="$2"
      shift 2
      ;;
    --stage-max-files)
      STAGE_MAX_FILES="$2"
      shift 2
      ;;
    --stage-max-gib)
      STAGE_MAX_GIB="$2"
      shift 2
      ;;
    --stage-min-age-seconds)
      STAGE_MIN_AGE_SECONDS="$2"
      shift 2
      ;;
    --process-max-files)
      PROCESS_MAX_FILES="$2"
      shift 2
      ;;
    --field)
      FIELD="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --shard-size-tokens)
      SHARD_SIZE_TOKENS="$2"
      shift 2
      ;;
    --val-ratio)
      VAL_RATIO="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --min-chars)
      MIN_CHARS="$2"
      shift 2
      ;;
    --max-chars)
      MAX_CHARS="$2"
      shift 2
      ;;
    --max-rows-per-file)
      MAX_ROWS_PER_FILE="$2"
      shift 2
      ;;
    --sleep-seconds)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --no-sync-to-warm)
      SYNC_TO_WARM=0
      shift
      ;;
    --no-purge-hot)
      PURGE_HOT=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "$WARM_PARQUET_DIR" ]]; then
  echo "error: warm parquet dir not found: $WARM_PARQUET_DIR" >&2
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "error: .venv/bin/python not found; run make setup-train first" >&2
  exit 1
fi

mkdir -p "$HOT_PARQUET_DIR" "$SHARDS_ROOT" "$STATE_DIR"
PROCESSED_FILE="$STATE_DIR/processed_parquet_files.txt"
touch "$PROCESSED_FILE"

LOCK_FILE="$STATE_DIR/loop.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "error: another fineweb_stage_shard_loop is already running" >&2
  exit 1
fi

LOG_FILE="$STATE_DIR/loop_$(date +%Y%m%d_%H%M%S).log"
log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

warm_shards_root="$WARM_ROOT/shards_global/$(basename "$SHARDS_ROOT")"
warm_tokenizer_dir="$WARM_ROOT/tokenizer"
warm_reports_dir="$WARM_ROOT/reports/fineweb_stage_shard_loop"

stage_once() {
  log "stage_start max_files=$STAGE_MAX_FILES max_gib=$STAGE_MAX_GIB"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    bash scripts/stage_fineweb_from_warm.sh \
      --src-dir "$WARM_PARQUET_DIR" \
      --dest-dir "$HOT_PARQUET_DIR" \
      --max-files "$STAGE_MAX_FILES" \
      --max-gib "$STAGE_MAX_GIB" \
      --min-age-seconds "$STAGE_MIN_AGE_SECONDS" \
      --dry-run | tee -a "$LOG_FILE"
  else
    bash scripts/stage_fineweb_from_warm.sh \
      --src-dir "$WARM_PARQUET_DIR" \
      --dest-dir "$HOT_PARQUET_DIR" \
      --max-files "$STAGE_MAX_FILES" \
      --max-gib "$STAGE_MAX_GIB" \
      --min-age-seconds "$STAGE_MIN_AGE_SECONDS" | tee -a "$LOG_FILE"
  fi
  log "stage_done"
}

select_unprocessed_files() {
  local -n out_ref="$1"
  out_ref=()
  mapfile -t all_files < <(find "$HOT_PARQUET_DIR" -maxdepth 1 -type f -name '*.parquet' -printf '%f\n' | sort)
  for name in "${all_files[@]}"; do
    if grep -Fqx "$name" "$PROCESSED_FILE"; then
      continue
    fi
    out_ref+=("$name")
    if [[ "$PROCESS_MAX_FILES" -gt 0 && "${#out_ref[@]}" -ge "$PROCESS_MAX_FILES" ]]; then
      break
    fi
  done
}

sync_batch_to_warm() {
  local batch_dir="$1"
  local report_path="$2"
  mkdir -p "$warm_shards_root" "$warm_tokenizer_dir" "$warm_reports_dir"
  rsync -ah "$batch_dir/" "$warm_shards_root/$(basename "$batch_dir")/"
  if [[ -f "$TOKENIZER_PATH" ]]; then
    rsync -ah "$TOKENIZER_PATH" "$warm_tokenizer_dir/"
  fi
  rsync -ah "$report_path" "$warm_reports_dir/"
  rsync -ah "$LOG_FILE" "$warm_reports_dir/"
}

process_batch() {
  local batch_index="$1"
  shift
  local files=("$@")
  if [[ "${#files[@]}" -eq 0 ]]; then
    return 0
  fi

  local batch_id
  batch_id="fw350bt_$(printf '%04d' "$batch_index")_$(date +%Y%m%d_%H%M%S)"
  local files_list="$STATE_DIR/${batch_id}.files.txt"
  local report_json="$STATE_DIR/${batch_id}.report.json"
  local output_dir="$SHARDS_ROOT/$batch_id"

  : > "$files_list"
  for name in "${files[@]}"; do
    printf '%s/%s\n' "$HOT_PARQUET_DIR" "$name" >> "$files_list"
  done

  local tok_arg_flag
  local tok_arg_path
  if [[ -f "$TOKENIZER_PATH" ]]; then
    tok_arg_flag="--tokenizer-in"
    tok_arg_path="$TOKENIZER_PATH"
  else
    tok_arg_flag="--tokenizer-out"
    tok_arg_path="$TOKENIZER_PATH"
    mkdir -p "$(dirname "$TOKENIZER_PATH")"
  fi

  log "batch_start id=$batch_id files=${#files[@]} tokenizer_arg=$tok_arg_flag"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry_run_shard_build output_dir=$output_dir files_list=$files_list"
  else
    PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py \
      --input-dir "$HOT_PARQUET_DIR" \
      --files-list "$files_list" \
      --output-dir "$output_dir" \
      "$tok_arg_flag" "$tok_arg_path" \
      --field "$FIELD" \
      --batch-size "$BATCH_SIZE" \
      --shard-size-tokens "$SHARD_SIZE_TOKENS" \
      --val-ratio "$VAL_RATIO" \
      --seed "$SEED" \
      --min-chars "$MIN_CHARS" \
      --max-chars "$MAX_CHARS" \
      --max-rows-per-file "$MAX_ROWS_PER_FILE" \
      --report-output "$report_json" | tee -a "$LOG_FILE"

    PYTHONPATH=src .venv/bin/python -m llm.cli verify-shards \
      --path "$output_dir" | tee -a "$LOG_FILE"

    if [[ "$SYNC_TO_WARM" -eq 1 ]]; then
      sync_batch_to_warm "$output_dir" "$report_json"
      log "batch_synced id=$batch_id warm_shards_root=$warm_shards_root"
    fi
  fi

  for name in "${files[@]}"; do
    printf '%s\n' "$name" >> "$PROCESSED_FILE"
  done
  sort -u "$PROCESSED_FILE" -o "$PROCESSED_FILE"

  if [[ "$PURGE_HOT" -eq 1 ]]; then
    for name in "${files[@]}"; do
      local hot_file="$HOT_PARQUET_DIR/$name"
      if [[ -f "$hot_file" ]]; then
        rm -f "$hot_file"
      fi
    done
    log "batch_hot_purged id=$batch_id purged_files=${#files[@]}"
  fi

  log "batch_done id=$batch_id"
}

log "loop_start warm_parquet_dir=$WARM_PARQUET_DIR hot_parquet_dir=$HOT_PARQUET_DIR"
log "loop_config shards_root=$SHARDS_ROOT tokenizer_path=$TOKENIZER_PATH iterations=$ITERATIONS"

completed_batches=0
while true; do
  stage_once

  selected=()
  select_unprocessed_files selected

  if [[ "${#selected[@]}" -eq 0 ]]; then
    if [[ "$ITERATIONS" -gt 0 && "$completed_batches" -ge "$ITERATIONS" ]]; then
      log "loop_done reason=iterations_reached completed_batches=$completed_batches"
      break
    fi
    log "idle_no_unprocessed_files sleep_seconds=$SLEEP_SECONDS"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  process_batch "$((completed_batches + 1))" "${selected[@]}"
  completed_batches=$((completed_batches + 1))

  if [[ "$ITERATIONS" -gt 0 && "$completed_batches" -ge "$ITERATIONS" ]]; then
    log "loop_done reason=iterations_reached completed_batches=$completed_batches"
    break
  fi
done

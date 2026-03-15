#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="/media/llm/data/fineweb/fineweb-edu-full/data"
SHARDS_ROOT="/media/llm/data/shards_global/fineweb-global-bpe-v1"
TOKENIZER_PATH="/media/llm/data/tokenizer/fineweb-global-bpe-v1.json"
STATE_DIR="/media/llm/data/reports/fineweb_edu_shard_loop"
PYTHON_BIN=".venv/bin/python"
JOB_PREFIX="fwedu"

FIELD="text"
BATCH_SIZE=8192
ENCODE_BATCH_SIZE=1024
TOKENIZER_THREADS=10
SHARD_JOBS=1
SHARD_SIZE_TOKENS=20000000
VAL_RATIO=0.01
SEED=42
MIN_CHARS=80
MAX_CHARS=0
MAX_ROWS_PER_FILE=0

PROCESS_MAX_FILES=12
MIN_AGE_SECONDS=180
SLEEP_SECONDS=60
ITERATIONS=0
KEEP_STAGE_DIRS=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/fineweb_edu_shard_loop.sh [options]

Continuously shard nested FineWeb-Edu parquet files with an existing tokenizer
into the shared fineweb-global-bpe-v1 shard root.

This script keeps tokenizer compatibility with existing FineWeb sample shards
and avoids basename collisions by creating deterministic per-file aliases.

Options:
  --source-root DIR          Source root with nested parquet files
  --shards-root DIR          Output shard root
  --tokenizer-path FILE      Existing tokenizer JSON to reuse
  --state-dir DIR            State/log directory
  --python-bin FILE          Python executable (default: .venv/bin/python)
  --job-prefix STR           Output batch prefix (default: fwedu)

  --field NAME               Parquet text field (default: text)
  --batch-size N             Parquet read batch size (default: 8192)
  --encode-batch-size N      Tokenizer encode batch size (default: 1024)
  --tokenizer-threads N      RAYON_NUM_THREADS for tokenizer (default: 10)
  --shard-jobs N             Parallel shard jobs per batch (default: 1)
  --shard-size-tokens N      Tokens per output shard (default: 20000000)
  --val-ratio X              Validation ratio (default: 0.01)
  --seed N                   RNG seed (default: 42)
  --min-chars N              Drop short rows (default: 80)
  --max-chars N              Truncate row chars (default: 0 disabled)
  --max-rows-per-file N      Optional row cap (default: 0 all rows)

  --process-max-files N      Max parquet files per batch (default: 12)
  --min-age-seconds N        Skip files newer than N seconds (default: 180)
  --sleep-seconds N          Idle retry sleep (default: 60)
  --iterations N             Successful batches before exit (default: 0 infinite)
  --keep-stage-dirs          Keep staged symlink dirs for debugging
  --dry-run                  Print actions without sharding
  -h, --help                 Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="$2"
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
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --job-prefix)
      JOB_PREFIX="$2"
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
    --encode-batch-size)
      ENCODE_BATCH_SIZE="$2"
      shift 2
      ;;
    --tokenizer-threads)
      TOKENIZER_THREADS="$2"
      shift 2
      ;;
    --shard-jobs)
      SHARD_JOBS="$2"
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
    --process-max-files)
      PROCESS_MAX_FILES="$2"
      shift 2
      ;;
    --min-age-seconds)
      MIN_AGE_SECONDS="$2"
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
    --keep-stage-dirs)
      KEEP_STAGE_DIRS=1
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
      usage >&2
      exit 2
      ;;
  esac
done

require_nonneg_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "error: $name must be an integer >= 0 (got: $value)" >&2
    exit 2
  fi
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: $name must be a positive integer (got: $value)" >&2
    exit 2
  fi
}

require_positive_int "batch-size" "$BATCH_SIZE"
require_positive_int "encode-batch-size" "$ENCODE_BATCH_SIZE"
require_positive_int "tokenizer-threads" "$TOKENIZER_THREADS"
require_positive_int "shard-jobs" "$SHARD_JOBS"
require_positive_int "shard-size-tokens" "$SHARD_SIZE_TOKENS"
require_nonneg_int "min-chars" "$MIN_CHARS"
require_nonneg_int "max-chars" "$MAX_CHARS"
require_nonneg_int "max-rows-per-file" "$MAX_ROWS_PER_FILE"
require_positive_int "process-max-files" "$PROCESS_MAX_FILES"
require_nonneg_int "min-age-seconds" "$MIN_AGE_SECONDS"
require_positive_int "sleep-seconds" "$SLEEP_SECONDS"
require_nonneg_int "iterations" "$ITERATIONS"

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "error: source-root not found: $SOURCE_ROOT" >&2
  exit 1
fi
if [[ ! -f "$TOKENIZER_PATH" ]]; then
  echo "error: tokenizer-path not found: $TOKENIZER_PATH" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: python-bin not executable: $PYTHON_BIN" >&2
  exit 1
fi

if ! PYTHONPATH=src "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import pyarrow.parquet  # noqa: F401
import tokenizers  # noqa: F401
PY
then
  echo "error: missing python deps (pyarrow/tokenizers); run make setup-train" >&2
  exit 1
fi

mkdir -p "$SHARDS_ROOT" "$STATE_DIR"
PROCESSED_FILE="$STATE_DIR/processed_relpaths.txt"
BAD_FILE="$STATE_DIR/bad_relpaths.txt"
mkdir -p "$STATE_DIR/staged"
touch "$PROCESSED_FILE" "$BAD_FILE"

LOCK_FILE="$STATE_DIR/loop.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "error: another fineweb_edu_shard_loop is already running" >&2
  exit 3
fi

LOG_FILE="$STATE_DIR/loop_$(date +%Y%m%d_%H%M%S).log"
log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

is_rel_recorded() {
  local rel="$1"
  local file="$2"
  grep -Fqx "$rel" "$file"
}

mark_bad() {
  local rel="$1"
  printf '%s\n' "$rel" >> "$BAD_FILE"
  sort -u "$BAD_FILE" -o "$BAD_FILE"
}

mark_processed_many() {
  local rel
  for rel in "$@"; do
    printf '%s\n' "$rel" >> "$PROCESSED_FILE"
  done
  sort -u "$PROCESSED_FILE" -o "$PROCESSED_FILE"
}

alias_for_rel() {
  local rel="$1"
  local parent base digest prefix
  parent="$(dirname "$rel")"
  base="$(basename "$rel")"
  if [[ "$parent" == "." ]]; then
    prefix="root"
  else
    prefix="${parent//\//__}"
  fi
  digest="$(printf '%s' "$rel" | sha1sum | awk '{print substr($1, 1, 10)}')"
  printf '%s__%s__%s' "$prefix" "$base" "$digest"
}

validate_parquet_file() {
  local file_path="$1"
  local field_name="$2"
  (
    exec 9>&-
    PYTHONPATH=src "$PYTHON_BIN" - "$file_path" "$field_name" <<'PY'
import sys
from pyarrow import parquet as pq

path = sys.argv[1]
field = sys.argv[2]
table = pq.ParquetFile(path)
meta = table.metadata
if meta is None or meta.num_row_groups <= 0:
    raise RuntimeError("missing row groups")
if meta.num_rows <= 0:
    raise RuntimeError("no rows")
if field not in table.schema.names:
    raise RuntimeError(f"missing field '{field}'")
PY
  )
}

select_candidates() {
  local now rel abs mtime age
  local -n out_ref="$1"
  out_ref=()
  now="$(date +%s)"

  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    if is_rel_recorded "$rel" "$PROCESSED_FILE"; then
      continue
    fi
    if is_rel_recorded "$rel" "$BAD_FILE"; then
      continue
    fi
    abs="$SOURCE_ROOT/$rel"
    [[ -f "$abs" ]] || continue
    mtime="$(stat -c%Y "$abs" || echo 0)"
    age=$((now - mtime))
    if [[ "$age" -lt "$MIN_AGE_SECONDS" ]]; then
      continue
    fi
    out_ref+=("$rel")
    if [[ "${#out_ref[@]}" -ge "$PROCESS_MAX_FILES" ]]; then
      break
    fi
  done < <(find "$SOURCE_ROOT" -type f -name '*.parquet' -printf '%P\n' | sort)
  return 0
}

process_batch() {
  local batch_index="$1"
  shift
  local rels=("$@")
  local batch_id rel abs alias
  local -a valid_rels=()
  local -a valid_aliases=()
  local -a job_ids=()
  local -a job_files_lists=()
  local -a job_rel_lists=()
  local -a job_stage_dirs=()
  local -a active_pids=()
  local -a active_job_rel_lists=()
  local -a succeeded_job_rel_lists=()
  local idx=0
  local j

  batch_id="${JOB_PREFIX}_$(printf '%04d' "$batch_index")_$(date +%Y%m%d_%H%M%S)"

  for rel in "${rels[@]}"; do
    abs="$SOURCE_ROOT/$rel"
    if ! validate_parquet_file "$abs" "$FIELD" >> "$LOG_FILE" 2>&1; then
      log "preflight_bad rel=$rel action=mark_bad"
      mark_bad "$rel"
      continue
    fi
    alias="$(alias_for_rel "$rel")"
    valid_rels+=("$rel")
    valid_aliases+=("$alias")
  done

  if [[ "${#valid_rels[@]}" -eq 0 ]]; then
    log "batch_skip id=$batch_id reason=no_valid_files_after_preflight"
    return 0
  fi

  local job_count="$SHARD_JOBS"
  if [[ "$job_count" -gt "${#valid_rels[@]}" ]]; then
    job_count="${#valid_rels[@]}"
  fi
  if [[ "$job_count" -lt 1 ]]; then
    job_count=1
  fi

  for ((j = 0; j < job_count; j++)); do
    local jid
    if [[ "$job_count" -eq 1 ]]; then
      jid="$batch_id"
    else
      jid="${batch_id}_j$(printf '%02d' $((j + 1)))"
    fi
    local stage_dir="$STATE_DIR/staged/$jid"
    local files_list="$STATE_DIR/${jid}.files.txt"
    local rel_list="$STATE_DIR/${jid}.relpaths.txt"
    mkdir -p "$stage_dir"
    : > "$files_list"
    : > "$rel_list"
    job_ids+=("$jid")
    job_stage_dirs+=("$stage_dir")
    job_files_lists+=("$files_list")
    job_rel_lists+=("$rel_list")
  done

  for ((idx = 0; idx < ${#valid_rels[@]}; idx++)); do
    local bucket=$((idx % job_count))
    rel="${valid_rels[$idx]}"
    alias="${valid_aliases[$idx]}"
    abs="$SOURCE_ROOT/$rel"
    ln -sfn "$abs" "${job_stage_dirs[$bucket]}/$alias"
    printf '%s\n' "$alias" >> "${job_files_lists[$bucket]}"
    printf '%s\n' "$rel" >> "${job_rel_lists[$bucket]}"
  done

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry_run_batch id=$batch_id files=${#valid_rels[@]} shard_jobs=$job_count"
    if [[ "$KEEP_STAGE_DIRS" -ne 1 ]]; then
      rm -rf "${job_stage_dirs[@]}"
    fi
    rm -f "${job_files_lists[@]}" "${job_rel_lists[@]}"
    return 0
  fi

  log "batch_start id=$batch_id files=${#valid_rels[@]} shard_jobs=$job_count tokenizer_threads=$TOKENIZER_THREADS"
  for ((j = 0; j < job_count; j++)); do
    local jid="${job_ids[$j]}"
    local stage_dir="${job_stage_dirs[$j]}"
    local files_list="${job_files_lists[$j]}"
    local rel_list="${job_rel_lists[$j]}"
    local output_dir="$SHARDS_ROOT/$jid"
    local report_json="$STATE_DIR/${jid}.report.json"
    if [[ ! -s "$files_list" ]]; then
      continue
    fi

    (
      set +e
      (
        exec 9>&-
        TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS="$TOKENIZER_THREADS" PYTHONPATH=src "$PYTHON_BIN" scripts/fineweb_parquet_to_shards.py \
          --input-dir "$stage_dir" \
          --files-list "$files_list" \
          --output-dir "$output_dir" \
          --tokenizer-in "$TOKENIZER_PATH" \
          --field "$FIELD" \
          --batch-size "$BATCH_SIZE" \
          --encode-batch-size "$ENCODE_BATCH_SIZE" \
          --shard-size-tokens "$SHARD_SIZE_TOKENS" \
          --val-ratio "$VAL_RATIO" \
          --seed "$SEED" \
          --min-chars "$MIN_CHARS" \
          --max-chars "$MAX_CHARS" \
          --max-rows-per-file "$MAX_ROWS_PER_FILE" \
          --report-output "$report_json" >> "$LOG_FILE" 2>&1
      )
      rc=$?
      set -e
      if [[ "$rc" -ne 0 ]]; then
        log "job_fail id=$jid rc=$rc phase=shard_build"
        exit "$rc"
      fi

      set +e
      (
        exec 9>&-
        PYTHONPATH=src "$PYTHON_BIN" -m llm.cli verify-shards --path "$output_dir" >> "$LOG_FILE" 2>&1
      )
      rc=$?
      set -e
      if [[ "$rc" -ne 0 ]]; then
        log "job_fail id=$jid rc=$rc phase=verify"
        exit "$rc"
      fi
      log "job_done id=$jid"
    ) &
    active_pids+=("$!")
    active_job_rel_lists+=("$rel_list")
  done

  for ((j = 0; j < ${#active_pids[@]}; j++)); do
    if wait "${active_pids[$j]}"; then
      succeeded_job_rel_lists+=("${active_job_rel_lists[$j]}")
    fi
  done

  if [[ "${#succeeded_job_rel_lists[@]}" -eq 0 ]]; then
    log "batch_fail id=$batch_id reason=no_successful_jobs"
    if [[ "$KEEP_STAGE_DIRS" -ne 1 ]]; then
      rm -rf "${job_stage_dirs[@]}"
    fi
    rm -f "${job_files_lists[@]}" "${job_rel_lists[@]}"
    return 1
  fi

  local -a processed_rels=()
  local -A seen=()
  local path one_rel
  for path in "${succeeded_job_rel_lists[@]}"; do
    while IFS= read -r one_rel; do
      [[ -z "$one_rel" ]] && continue
      if [[ -n "${seen[$one_rel]+x}" ]]; then
        continue
      fi
      seen["$one_rel"]=1
      processed_rels+=("$one_rel")
    done < "$path"
  done
  mark_processed_many "${processed_rels[@]}"
  log "batch_done id=$batch_id processed=${#processed_rels[@]} successful_jobs=${#succeeded_job_rel_lists[@]}/${#active_pids[@]}"

  if [[ "$KEEP_STAGE_DIRS" -ne 1 ]]; then
    rm -rf "${job_stage_dirs[@]}"
  fi
  rm -f "${job_files_lists[@]}" "${job_rel_lists[@]}"
  return 0
}

log "loop_start source_root=$SOURCE_ROOT shards_root=$SHARDS_ROOT tokenizer_path=$TOKENIZER_PATH state_dir=$STATE_DIR"
log "loop_config job_prefix=$JOB_PREFIX process_max_files=$PROCESS_MAX_FILES min_age_seconds=$MIN_AGE_SECONDS shard_jobs=$SHARD_JOBS shard_size_tokens=$SHARD_SIZE_TOKENS tokenizer_threads=$TOKENIZER_THREADS sleep_seconds=$SLEEP_SECONDS iterations=$ITERATIONS"

completed=0
while true; do
  if [[ "$ITERATIONS" -gt 0 && "$completed" -ge "$ITERATIONS" ]]; then
    log "loop_done reason=iterations_reached completed=$completed"
    break
  fi

  selected=()
  select_candidates selected
  if [[ "${#selected[@]}" -eq 0 ]]; then
    log "idle_no_candidates sleep_seconds=$SLEEP_SECONDS"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  if process_batch "$((completed + 1))" "${selected[@]}"; then
    completed=$((completed + 1))
  else
    log "batch_retry_later sleep_seconds=$SLEEP_SECONDS"
    sleep "$SLEEP_SECONDS"
  fi
done

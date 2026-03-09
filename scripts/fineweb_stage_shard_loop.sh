#!/usr/bin/env bash
set -euo pipefail

WARM_PARQUET_DIR="/mnt/ceph/llm/data/fineweb/sample-350BT/sample/350BT"
HOT_PARQUET_DIR="data/fineweb/sample-350BT/sample/350BT"
SHARDS_ROOT="data/shards_global/fineweb-global-bpe-v1"
TOKENIZER_PATH="artifacts/tokenizer/fineweb-global-bpe-v1.json"
STATE_DIR="artifacts/reports/fineweb_stage_shard_loop"
WARM_ROOT="/mnt/ceph/llm/data"
BAD_PARQUET_FILE=""
QUARANTINE_DIR=""

FIELD="text"
BATCH_SIZE=8192
ENCODE_BATCH_SIZE=1024
TOKENIZER_THREADS=10
SHARD_JOBS=1
SHARD_RETRY_ON_OOM=1
SHARD_MIN_BATCH_SIZE=512
SHARD_SIZE_TOKENS=5000000
VAL_RATIO=0.01
SEED=42
MIN_CHARS=80
MAX_CHARS=0
MAX_ROWS_PER_FILE=0

STAGE_MAX_FILES=10
STAGE_MAX_GIB=0
STAGE_MIN_AGE_SECONDS=180
HOT_QUEUE_MIN_FILES=8
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
  --bad-parquet-file FILE     State file of known-bad parquet basenames
                              (default: <state-dir>/bad_parquet_files.txt)
  --quarantine-dir DIR        Move preflight-failed hot parquet files here
                              (default: <state-dir>/quarantine_bad_parquet)

  --stage-max-files N         Max parquet files to stage per cycle (default: 10)
  --stage-max-gib N           Max GiB to stage per cycle (default: 0 unlimited)
  --stage-min-age-seconds N   Skip recently modified source files (default: 180)
  --hot-queue-min-files N     Try to keep at least N parquet files staged on hot disk
                              (default: 8)
  --process-max-files N       Max staged files to shard per batch (default: 10)

  --field NAME                Parquet text field (default: text)
  --batch-size N              Parquet read batch size (default: 8192)
  --encode-batch-size N       Tokenizer encode batch size (default: 1024)
  --tokenizer-threads N       Tokenizer worker threads via RAYON_NUM_THREADS (default: 10)
  --shard-jobs N              Parallel shard jobs per batch (default: 1)
  --no-shard-retry-on-oom     Disable automatic shard build retry on OOM/memory errors
  --shard-min-batch-size N    Lower bound for batch-size backoff on OOM (default: 512)
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
    --bad-parquet-file)
      BAD_PARQUET_FILE="$2"
      shift 2
      ;;
    --quarantine-dir)
      QUARANTINE_DIR="$2"
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
    --hot-queue-min-files)
      HOT_QUEUE_MIN_FILES="$2"
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
    --no-shard-retry-on-oom)
      SHARD_RETRY_ON_OOM=0
      shift
      ;;
    --shard-min-batch-size)
      SHARD_MIN_BATCH_SIZE="$2"
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
if ! PYTHONPATH=src .venv/bin/python - <<'PY' >/dev/null 2>&1
import pyarrow.parquet  # noqa: F401
PY
then
  echo "error: pyarrow is required for parquet preflight; run make setup-train first" >&2
  exit 1
fi

mkdir -p "$HOT_PARQUET_DIR" "$SHARDS_ROOT" "$STATE_DIR"
PROCESSED_FILE="$STATE_DIR/processed_parquet_files.txt"
touch "$PROCESSED_FILE"
STAGE_SKIP_FILE="$STATE_DIR/stage_skip_list.txt"
touch "$STAGE_SKIP_FILE"

if [[ -z "$BAD_PARQUET_FILE" ]]; then
  BAD_PARQUET_FILE="$STATE_DIR/bad_parquet_files.txt"
fi
if [[ -z "$QUARANTINE_DIR" ]]; then
  QUARANTINE_DIR="$STATE_DIR/quarantine_bad_parquet"
fi
touch "$BAD_PARQUET_FILE"
mkdir -p "$QUARANTINE_DIR"

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

is_bad_parquet_name() {
  local name="$1"
  grep -Fqx "$name" "$BAD_PARQUET_FILE"
}

is_processed_parquet_name() {
  local name="$1"
  grep -Fqx "$name" "$PROCESSED_FILE"
}

refresh_stage_skip_list() {
  : > "$STAGE_SKIP_FILE"
  if [[ -s "$BAD_PARQUET_FILE" ]]; then
    cat "$BAD_PARQUET_FILE" >> "$STAGE_SKIP_FILE"
  fi
  if [[ -s "$PROCESSED_FILE" ]]; then
    cat "$PROCESSED_FILE" >> "$STAGE_SKIP_FILE"
  fi
  if [[ -s "$STAGE_SKIP_FILE" ]]; then
    sort -u "$STAGE_SKIP_FILE" -o "$STAGE_SKIP_FILE"
  fi
}

bootstrap_processed_from_manifests() {
  local before_count
  before_count="$(awk 'NF {c+=1} END {print c+0}' "$PROCESSED_FILE")"
  PYTHONPATH=src .venv/bin/python - "$SHARDS_ROOT" "$PROCESSED_FILE" <<'PY' >> "$LOG_FILE" 2>&1
import json
import sys
from pathlib import Path

shards_root = Path(sys.argv[1])
processed_path = Path(sys.argv[2])

names: set[str] = set()
if processed_path.exists():
    for raw in processed_path.read_text(encoding="utf-8", errors="replace").splitlines():
        value = raw.strip()
        if value:
            names.add(value)

for manifest_path in sorted(shards_root.rglob("manifest.json")):
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    for raw in manifest.get("input_files", []):
        name = Path(str(raw)).name
        if name:
            names.add(name)

processed_path.parent.mkdir(parents=True, exist_ok=True)
processed_path.write_text(
    "".join(f"{name}\n" for name in sorted(names)),
    encoding="utf-8",
)
print(f"bootstrap_processed total={len(names)}")
PY
  local after_count
  after_count="$(awk 'NF {c+=1} END {print c+0}' "$PROCESSED_FILE")"
  local added=$((after_count - before_count))
  if [[ "$added" -gt 0 ]]; then
    log "processed_bootstrap_from_manifests added=$added total=$after_count"
  else
    log "processed_bootstrap_from_manifests added=0 total=$after_count"
  fi
}

purge_hot_known_files() {
  local removed=0
  mapfile -t hot_files < <(find "$HOT_PARQUET_DIR" -maxdepth 1 -type f -name '*.parquet' -printf '%f\n' | sort)
  for name in "${hot_files[@]}"; do
    if is_bad_parquet_name "$name" || is_processed_parquet_name "$name"; then
      local hot_file="$HOT_PARQUET_DIR/$name"
      if [[ -f "$hot_file" ]]; then
        rm -f "$hot_file"
        removed=$((removed + 1))
      fi
    fi
  done
  if [[ "$removed" -gt 0 ]]; then
    log "hot_cleanup_removed_known_files removed=$removed"
  fi
}

count_hot_eligible_files() {
  local count=0
  mapfile -t hot_files < <(find "$HOT_PARQUET_DIR" -maxdepth 1 -type f -name '*.parquet' -printf '%f\n' | sort)
  for name in "${hot_files[@]}"; do
    if is_bad_parquet_name "$name" || is_processed_parquet_name "$name"; then
      continue
    fi
    count=$((count + 1))
  done
  echo "$count"
}

mark_bad_parquet() {
  local parquet_path="$1"
  local reason="$2"
  local name
  name="$(basename "$parquet_path")"
  printf '%s\n' "$name" >> "$BAD_PARQUET_FILE"
  sort -u "$BAD_PARQUET_FILE" -o "$BAD_PARQUET_FILE"

  if [[ -f "$parquet_path" ]]; then
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local quarantine_path="$QUARANTINE_DIR/${name}.bad_${ts}"
    mv -f "$parquet_path" "$quarantine_path"
    log "parquet_quarantined file=$name reason=$reason quarantine_path=$quarantine_path"
  else
    log "parquet_marked_bad file=$name reason=$reason action=record_only"
  fi
}

reconcile_bad_parquet_from_warm() {
  if [[ ! -s "$BAD_PARQUET_FILE" ]]; then
    return
  fi

  local tmp_file
  tmp_file="$(mktemp)"
  local name
  local retained=0
  local reinstated=0
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    local warm_path="$WARM_PARQUET_DIR/$name"
    if [[ -f "$warm_path" ]] && validate_parquet_file "$warm_path" "$FIELD" >> "$LOG_FILE" 2>&1; then
      log "bad_parquet_reinstated file=$name reason=warm_source_valid"
      reinstated=$((reinstated + 1))
      continue
    fi
    printf '%s\n' "$name" >> "$tmp_file"
    retained=$((retained + 1))
  done < "$BAD_PARQUET_FILE"
  sort -u "$tmp_file" -o "$tmp_file"
  mv "$tmp_file" "$BAD_PARQUET_FILE"
  log "bad_parquet_reconcile retained=$retained reinstated=$reinstated"
}

validate_parquet_file() {
  local parquet_path="$1"
  local field_name="$2"
  PYTHONPATH=src .venv/bin/python - "$parquet_path" "$field_name" <<'PY'
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
}

preflight_selected_files() {
  local -n in_ref="$1"
  local -n out_ref="$2"
  out_ref=()

  for name in "${in_ref[@]}"; do
    if is_bad_parquet_name "$name"; then
      log "preflight_skip_known_bad file=$name"
      continue
    fi
    local hot_path="$HOT_PARQUET_DIR/$name"
    if [[ ! -f "$hot_path" ]]; then
      log "preflight_skip_missing file=$name"
      continue
    fi
    if validate_parquet_file "$hot_path" "$FIELD" >> "$LOG_FILE" 2>&1; then
      out_ref+=("$name")
    else
      mark_bad_parquet "$hot_path" "preflight_validation_failed"
    fi
  done
}

validate_job_artifacts() {
  local job_id="$1"
  local files_list="$2"
  local report_json="$STATE_DIR/${job_id}.report.json"
  local output_dir="$SHARDS_ROOT/$job_id"

  if [[ ! -f "$report_json" ]]; then
    log "guardrail_fail id=$job_id reason=missing_report report=$report_json"
    return 1
  fi
  if [[ ! -f "$output_dir/manifest.json" ]]; then
    log "guardrail_fail id=$job_id reason=missing_manifest manifest=$output_dir/manifest.json"
    return 1
  fi

  PYTHONPATH=src .venv/bin/python -m llm.fineweb_guardrails \
    --job-id "$job_id" \
    --report-json "$report_json" \
    --output-dir "$output_dir" \
    --files-list "$files_list"
}

validate_batch_guardrails() {
  local batch_id="$1"
  local -n ids_ref="$2"
  local -n lists_ref="$3"
  local failed=0
  local idx
  for ((idx = 0; idx < ${#ids_ref[@]}; idx++)); do
    if ! validate_job_artifacts "${ids_ref[$idx]}" "${lists_ref[$idx]}" | tee -a "$LOG_FILE"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    log "batch_guardrails_failed id=$batch_id"
    return 1
  fi
  log "batch_guardrails_ok id=$batch_id jobs=${#ids_ref[@]}"
}

stage_once() {
  refresh_stage_skip_list

  local hot_count
  local eligible_hot_count
  hot_count="$(find "$HOT_PARQUET_DIR" -maxdepth 1 -type f -name '*.parquet' | wc -l | tr -d ' ')"
  eligible_hot_count="$(count_hot_eligible_files)"
  local stage_files="$STAGE_MAX_FILES"

  if [[ "$HOT_QUEUE_MIN_FILES" -gt 0 ]]; then
    if [[ "$eligible_hot_count" -ge "$HOT_QUEUE_MIN_FILES" ]]; then
      log "stage_skip reason=queue_satisfied hot_count=$hot_count eligible_hot_count=$eligible_hot_count queue_min=$HOT_QUEUE_MIN_FILES"
      return 0
    fi
    local needed
    needed=$((HOT_QUEUE_MIN_FILES - eligible_hot_count))
    if [[ "$stage_files" -le 0 || "$needed" -lt "$stage_files" ]]; then
      stage_files="$needed"
    fi
  fi

  if [[ "$stage_files" -le 0 ]]; then
    log "stage_skip reason=stage_files_nonpositive hot_count=$hot_count eligible_hot_count=$eligible_hot_count queue_min=$HOT_QUEUE_MIN_FILES"
    return 0
  fi

  log "stage_start max_files=$stage_files max_gib=$STAGE_MAX_GIB hot_count=$hot_count eligible_hot_count=$eligible_hot_count queue_min=$HOT_QUEUE_MIN_FILES"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    (
      # Prevent staging subprocess from inheriting the loop lock FD.
      exec 9>&-
      bash scripts/stage_fineweb_from_warm.sh \
        --src-dir "$WARM_PARQUET_DIR" \
        --dest-dir "$HOT_PARQUET_DIR" \
        --max-files "$stage_files" \
        --max-gib "$STAGE_MAX_GIB" \
        --min-age-seconds "$STAGE_MIN_AGE_SECONDS" \
        --skip-list "$STAGE_SKIP_FILE" \
        --dry-run
    ) | tee -a "$LOG_FILE"
  else
    (
      # Prevent staging subprocess from inheriting the loop lock FD.
      exec 9>&-
      bash scripts/stage_fineweb_from_warm.sh \
        --src-dir "$WARM_PARQUET_DIR" \
        --dest-dir "$HOT_PARQUET_DIR" \
        --max-files "$stage_files" \
        --max-gib "$STAGE_MAX_GIB" \
        --min-age-seconds "$STAGE_MIN_AGE_SECONDS" \
        --skip-list "$STAGE_SKIP_FILE"
    ) | tee -a "$LOG_FILE"
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
    if is_bad_parquet_name "$name"; then
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
  (
    exec 9>&-
    rsync -ah "$batch_dir/" "$warm_shards_root/$(basename "$batch_dir")/"
  )
  if [[ -f "$TOKENIZER_PATH" ]]; then
    (
      exec 9>&-
      rsync -ah "$TOKENIZER_PATH" "$warm_tokenizer_dir/"
    )
  fi
  (
    exec 9>&-
    rsync -ah "$report_path" "$warm_reports_dir/"
  )
  (
    exec 9>&-
    rsync -ah "$LOG_FILE" "$warm_reports_dir/"
  )
}

run_shard_job() {
  local job_id="$1"
  local files_list="$2"
  local tok_arg_flag="$3"
  local tok_arg_path="$4"
  local report_json="$STATE_DIR/${job_id}.report.json"
  local output_dir="$SHARDS_ROOT/$job_id"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry_run_shard_build id=$job_id output_dir=$output_dir files_list=$files_list"
    return 0
  fi

  local shard_batch_size
  shard_batch_size="$BATCH_SIZE"
  local shard_attempt=1
  local shard_rc=0
  while true; do
    local shard_attempt_log="$STATE_DIR/${job_id}.shard_attempt_${shard_attempt}.log"
    log "shard_build_attempt id=$job_id attempt=$shard_attempt batch_size=$shard_batch_size encode_batch_size=$ENCODE_BATCH_SIZE tokenizer_threads=$TOKENIZER_THREADS"

    set +e
    (
      # Prevent shard subprocesses from inheriting the loop lock FD.
      exec 9>&-
      TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS="$TOKENIZER_THREADS" PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py \
        --input-dir "$HOT_PARQUET_DIR" \
        --files-list "$files_list" \
        --output-dir "$output_dir" \
        "$tok_arg_flag" "$tok_arg_path" \
        --field "$FIELD" \
        --batch-size "$shard_batch_size" \
        --encode-batch-size "$ENCODE_BATCH_SIZE" \
        --shard-size-tokens "$SHARD_SIZE_TOKENS" \
        --val-ratio "$VAL_RATIO" \
        --seed "$SEED" \
        --min-chars "$MIN_CHARS" \
        --max-chars "$MAX_CHARS" \
        --max-rows-per-file "$MAX_ROWS_PER_FILE" \
        --report-output "$report_json"
    ) 2>&1 | tee -a "$LOG_FILE" "$shard_attempt_log"
    shard_rc=${PIPESTATUS[0]}
    set -e

    if [[ "$shard_rc" -eq 0 ]]; then
      break
    fi

    if [[ "$SHARD_RETRY_ON_OOM" -ne 1 ]]; then
      log "shard_build_failed id=$job_id attempt=$shard_attempt rc=$shard_rc retry_on_oom=0"
      return "$shard_rc"
    fi

    if ! grep -Eqi 'out of memory|memoryerror|arrowmemoryerror|bad_alloc|cannot allocate memory|std::bad_alloc' "$shard_attempt_log"; then
      log "shard_build_failed id=$job_id attempt=$shard_attempt rc=$shard_rc reason=non_oom_error"
      return "$shard_rc"
    fi

    if [[ "$shard_batch_size" -le "$SHARD_MIN_BATCH_SIZE" ]]; then
      log "shard_build_failed id=$job_id attempt=$shard_attempt rc=$shard_rc reason=min_batch_reached batch_size=$shard_batch_size"
      return "$shard_rc"
    fi

    local next_batch_size
    next_batch_size=$((shard_batch_size / 2))
    if [[ "$next_batch_size" -lt "$SHARD_MIN_BATCH_SIZE" ]]; then
      next_batch_size="$SHARD_MIN_BATCH_SIZE"
    fi
    if [[ "$next_batch_size" -eq "$shard_batch_size" ]]; then
      log "shard_build_failed id=$job_id attempt=$shard_attempt rc=$shard_rc reason=no_batch_change_possible batch_size=$shard_batch_size"
      return "$shard_rc"
    fi

    log "shard_build_retry id=$job_id from_batch_size=$shard_batch_size to_batch_size=$next_batch_size reason=oom_detected"
    shard_batch_size="$next_batch_size"
    shard_attempt=$((shard_attempt + 1))
  done

  (
    # Prevent verify subprocess from inheriting the loop lock FD.
    exec 9>&-
    PYTHONPATH=src .venv/bin/python -m llm.cli verify-shards \
      --path "$output_dir"
  ) | tee -a "$LOG_FILE"

  if [[ "$SYNC_TO_WARM" -eq 1 ]]; then
    sync_batch_to_warm "$output_dir" "$report_json"
    log "batch_synced id=$job_id warm_shards_root=$warm_shards_root"
  fi
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

  log "batch_start id=$batch_id files=${#files[@]} tokenizer_arg=$tok_arg_flag shard_jobs=$SHARD_JOBS"

  local job_count="$SHARD_JOBS"
  local -a guard_job_ids=()
  local -a guard_job_lists=()
  if [[ "$tok_arg_flag" == "--tokenizer-out" && "$job_count" -gt 1 ]]; then
    log "batch_warn id=$batch_id reason=tokenizer_train_requires_single_job requested_jobs=$job_count forcing_jobs=1"
    job_count=1
  fi
  if [[ "$job_count" -gt "${#files[@]}" ]]; then
    job_count="${#files[@]}"
  fi

  if [[ "$job_count" -le 1 ]]; then
    local files_list="$STATE_DIR/${batch_id}.files.txt"
    : > "$files_list"
    for name in "${files[@]}"; do
      printf '%s\n' "$name" >> "$files_list"
    done
    run_shard_job "$batch_id" "$files_list" "$tok_arg_flag" "$tok_arg_path"
    guard_job_ids+=("$batch_id")
    guard_job_lists+=("$files_list")
  else
    local -a job_lists=()
    local -a job_ids=()
    local -a pids=()
    local -a active_job_ids=()
    local -a active_job_lists=()
    local idx=0
    local j
    for ((j = 0; j < job_count; j++)); do
      local jid="${batch_id}_j$(printf '%02d' $((j + 1)))"
      local jlist="$STATE_DIR/${jid}.files.txt"
      : > "$jlist"
      job_lists+=("$jlist")
      job_ids+=("$jid")
    done
    for name in "${files[@]}"; do
      local bucket=$((idx % job_count))
      printf '%s\n' "$name" >> "${job_lists[$bucket]}"
      idx=$((idx + 1))
    done

    for ((j = 0; j < job_count; j++)); do
      if [[ ! -s "${job_lists[$j]}" ]]; then
        continue
      fi
      run_shard_job "${job_ids[$j]}" "${job_lists[$j]}" "$tok_arg_flag" "$tok_arg_path" &
      pids+=("$!")
      active_job_ids+=("${job_ids[$j]}")
      active_job_lists+=("${job_lists[$j]}")
    done

    local failed=0
    for ((j = 0; j < ${#pids[@]}; j++)); do
      if ! wait "${pids[$j]}"; then
        log "batch_job_failed id=${active_job_ids[$j]}"
        failed=1
      fi
    done
    if [[ "$failed" -ne 0 ]]; then
      log "batch_failed id=$batch_id reason=one_or_more_parallel_jobs_failed"
      return 1
    fi
    guard_job_ids=("${active_job_ids[@]}")
    guard_job_lists=("${active_job_lists[@]}")
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "batch_done_dry_run id=$batch_id"
    return 0
  fi
  validate_batch_guardrails "$batch_id" guard_job_ids guard_job_lists

  for name in "${files[@]}"; do
    printf '%s\n' "$name" >> "$PROCESSED_FILE"
  done
  sort -u "$PROCESSED_FILE" -o "$PROCESSED_FILE"
  refresh_stage_skip_list

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
log "loop_config shards_root=$SHARDS_ROOT tokenizer_path=$TOKENIZER_PATH iterations=$ITERATIONS hot_queue_min_files=$HOT_QUEUE_MIN_FILES bad_parquet_file=$BAD_PARQUET_FILE quarantine_dir=$QUARANTINE_DIR"
reconcile_bad_parquet_from_warm
bootstrap_processed_from_manifests
refresh_stage_skip_list

completed_batches=0
while true; do
  purge_hot_known_files
  selected=()
  select_unprocessed_files selected

  if [[ "${#selected[@]}" -eq 0 ]]; then
    stage_once
    selected=()
    select_unprocessed_files selected
  fi

  if [[ "${#selected[@]}" -eq 0 ]]; then
    if [[ "$ITERATIONS" -gt 0 && "$completed_batches" -ge "$ITERATIONS" ]]; then
      log "loop_done reason=iterations_reached completed_batches=$completed_batches"
      break
    fi
    log "idle_no_unprocessed_files sleep_seconds=$SLEEP_SECONDS"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  valid_selected=()
  preflight_selected_files selected valid_selected
  if [[ "${#valid_selected[@]}" -eq 0 ]]; then
    log "batch_skip reason=no_valid_parquet_after_preflight selected=${#selected[@]}"
    stage_once
    sleep 2
    continue
  fi
  if [[ "${#valid_selected[@]}" -lt "${#selected[@]}" ]]; then
    log "preflight_filtered selected=${#selected[@]} valid=${#valid_selected[@]}"
  fi

  process_batch "$((completed_batches + 1))" "${valid_selected[@]}"
  completed_batches=$((completed_batches + 1))
  stage_once

  if [[ "$ITERATIONS" -gt 0 && "$completed_batches" -ge "$ITERATIONS" ]]; then
    log "loop_done reason=iterations_reached completed_batches=$completed_batches"
    break
  fi
done

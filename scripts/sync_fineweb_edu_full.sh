#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sync_fineweb_edu_full.sh [local_dir]

Defaults:
  local_dir: /mnt/pve/cephfs/llm/data/fineweb/fineweb-edu-full

Environment overrides:
  MAX_WORKERS=10
  RETRY_DELAY_SECONDS=30
  ATTEMPT_TIMEOUT_SECONDS=10800
  HF_HUB_DOWNLOAD_TIMEOUT=120   # per-file read timeout (seconds)
  ENABLE_TARGETED_TIMEOUT_RETRY=1
  TARGETED_MAX_FILES=12
  TARGETED_RETRIES_PER_FILE=3
  TARGETED_ATTEMPT_TIMEOUT_SECONDS=1800
  TARGETED_HF_HUB_DOWNLOAD_TIMEOUT=300
  MAX_RETRIES=0                  # 0 = retry forever
  HF_HUB_ENABLE_HF_TRANSFER=1    # set 0 to disable
  HF_TOKEN=hf_xxx                # optional for public dataset, recommended for rate limits

Notes:
  - Downloads full HuggingFaceFW/fineweb-edu parquet set (resumable).
  - Safe to rerun; it resumes existing partial files.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: command not found: $1" >&2
    exit 1
  fi
}

LOCAL_DIR="${1:-/mnt/pve/cephfs/llm/data/fineweb/fineweb-edu-full}"
DATASET="HuggingFaceFW/fineweb-edu"
INCLUDE_PATTERN="data/*/*.parquet"
MAX_WORKERS="${MAX_WORKERS:-10}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-30}"
ATTEMPT_TIMEOUT_SECONDS="${ATTEMPT_TIMEOUT_SECONDS:-10800}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
ENABLE_TARGETED_TIMEOUT_RETRY="${ENABLE_TARGETED_TIMEOUT_RETRY:-1}"
TARGETED_MAX_FILES="${TARGETED_MAX_FILES:-12}"
TARGETED_RETRIES_PER_FILE="${TARGETED_RETRIES_PER_FILE:-3}"
TARGETED_ATTEMPT_TIMEOUT_SECONDS="${TARGETED_ATTEMPT_TIMEOUT_SECONDS:-1800}"
TARGETED_HF_HUB_DOWNLOAD_TIMEOUT="${TARGETED_HF_HUB_DOWNLOAD_TIMEOUT:-300}"
MAX_RETRIES="${MAX_RETRIES:-0}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
LOG_FILE="${LOCAL_DIR}/fineweb_edu_sync.log"

HF_BIN="hf"
if ! command -v "$HF_BIN" >/dev/null 2>&1; then
  HF_BIN="huggingface-cli"
fi

require_cmd "$HF_BIN"
require_cmd find
require_cmd date
require_cmd awk
require_cmd sleep

mkdir -p "$LOCAL_DIR"

LOCK_FILE="${LOCAL_DIR}/.fineweb_edu_sync.lock"
if [[ -f "$LOCK_FILE" ]]; then
  prev_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  if [[ -n "${prev_pid:-}" ]] && kill -0 "$prev_pid" 2>/dev/null; then
    echo "error: sync already running (pid=$prev_pid)" >&2
    exit 3
  fi
fi
echo "$$" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '[%s] %s\n' "$ts" "$1" | tee -a "$LOG_FILE"
}

run_download_cmd() {
  local attempt_timeout="$1"
  local read_timeout="$2"
  shift 2
  local -a dl_cmd=("$@")
  if command -v timeout >/dev/null 2>&1; then
    HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" HF_HUB_DOWNLOAD_TIMEOUT="$read_timeout" \
      timeout "$attempt_timeout" "${dl_cmd[@]}" >> "$LOG_FILE" 2>&1
    return $?
  fi
  HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" HF_HUB_DOWNLOAD_TIMEOUT="$read_timeout" \
    "${dl_cmd[@]}" >> "$LOG_FILE" 2>&1
}

collect_timeout_candidates() {
  local start_line="$1"
  sed -n "$((start_line + 1)),\$p" "$LOG_FILE" \
    | sed -nE 's#.*Error while downloading from https://[^ ]+/data/([^:]+\.parquet):.*#data/\1#p' \
    | sort \
    | uniq -c \
    | sort -nr \
    | awk '{print $2}' \
    | head -n "$TARGETED_MAX_FILES"
}

run_targeted_retries() {
  local start_line="$1"
  local -a targets=()
  local target_path retry rc target_full
  mapfile -t targets < <(collect_timeout_candidates "$start_line")
  if [[ "${#targets[@]}" -eq 0 ]]; then
    log "targeted_retry: no timeout candidates in this attempt window"
    return 0
  fi
  log "targeted_retry: candidates=${#targets[@]} retries_per_file=${TARGETED_RETRIES_PER_FILE}"
  for target_path in "${targets[@]}"; do
    target_full="${LOCAL_DIR}/${target_path}"
    if [[ -f "$target_full" ]]; then
      continue
    fi
    for ((retry = 1; retry <= TARGETED_RETRIES_PER_FILE; retry++)); do
      local -a target_cmd=(
        "$HF_BIN" download "$DATASET"
        --repo-type dataset
        --include "$target_path"
        --local-dir "$LOCAL_DIR"
        --max-workers 1
      )
      set +e
      run_download_cmd "$TARGETED_ATTEMPT_TIMEOUT_SECONDS" "$TARGETED_HF_HUB_DOWNLOAD_TIMEOUT" "${target_cmd[@]}"
      rc=$?
      set -e
      if [[ "$rc" -eq 0 && -f "$target_full" ]]; then
        log "targeted_retry_ok path=${target_path} attempt=${retry}"
        break
      fi
      log "targeted_retry_fail path=${target_path} attempt=${retry} rc=${rc}"
      sleep 2
    done
  done
}

if [[ -z "${HF_TOKEN:-}" ]]; then
  log "warning: HF_TOKEN is not set; authenticated requests may reduce timeout/rate-limit pressure"
fi

attempt=1
while true; do
  if [[ "$MAX_RETRIES" -gt 0 && "$attempt" -gt "$MAX_RETRIES" ]]; then
    log "max retries reached (${MAX_RETRIES}); exiting with failure"
    exit 10
  fi

  parquet_count="$(find "$LOCAL_DIR" -type f -name '*.parquet' | wc -l | tr -d ' ')"
  incomplete_count="$(find "$LOCAL_DIR" -type f -name '*.incomplete' | wc -l | tr -d ' ')"
  log "attempt=${attempt} parquet_files=${parquet_count} incomplete_files=${incomplete_count}"

  cmd=(
    "$HF_BIN" download "$DATASET"
    --repo-type dataset
    --include "$INCLUDE_PATTERN"
    --local-dir "$LOCAL_DIR"
    --max-workers "$MAX_WORKERS"
  )

  attempt_log_start_line="$(wc -l < "$LOG_FILE" | tr -d ' ')"
  set +e
  run_download_cmd "$ATTEMPT_TIMEOUT_SECONDS" "$HF_HUB_DOWNLOAD_TIMEOUT" "${cmd[@]}"
  rc=$?
  set -e

  parquet_count="$(find "$LOCAL_DIR" -type f -name '*.parquet' | wc -l | tr -d ' ')"
  incomplete_count="$(find "$LOCAL_DIR" -type f -name '*.incomplete' | wc -l | tr -d ' ')"

  if [[ "$ENABLE_TARGETED_TIMEOUT_RETRY" -eq 1 && ( "$rc" -ne 0 || "$incomplete_count" -ne 0 ) ]]; then
    run_targeted_retries "$attempt_log_start_line"
    parquet_count="$(find "$LOCAL_DIR" -type f -name '*.parquet' | wc -l | tr -d ' ')"
    incomplete_count="$(find "$LOCAL_DIR" -type f -name '*.incomplete' | wc -l | tr -d ' ')"
  fi

  if [[ "$rc" -eq 0 && "$incomplete_count" -eq 0 ]]; then
    log "completed successfully parquet_files=${parquet_count}"
    exit 0
  fi

  log "attempt=${attempt} rc=${rc} parquet_files=${parquet_count} incomplete_files=${incomplete_count}; retry in ${RETRY_DELAY_SECONDS}s"
  sleep "$RETRY_DELAY_SECONDS"
  attempt=$((attempt + 1))
done

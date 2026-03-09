#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/hf_download_watchdog.sh \
    --dataset HuggingFaceFW/fineweb \
    --repo-type dataset \
    --include "sample/350BT/*.parquet" \
    --local-dir /mnt/ceph/llm/data/fineweb/sample-350BT \
    [--max-workers 4] \
    [--enable-hf-transfer | --disable-hf-transfer] \
    [--skip-dry-run] \
    [--dry-run-timeout-seconds 180] \
    [--attempt-timeout-seconds 5400] \
    [--retry-delay-seconds 30] \
    [--max-retries 0] \
    [--worker-log-file artifacts/reports/fineweb_350bt_download_resumable.log] \
    [--watchdog-log-file artifacts/reports/hf_download_watchdog.log] \
    [--check-interval-seconds 120] \
    [--stall-seconds 1200] \
    [--exit-on-complete --expected-parquet-files 510 --expected-bytes 1061360917731]

Notes:
  - Wraps `hf_download_resumable.sh` and keeps it running.
  - Restarts the worker if it exits unexpectedly.
  - Detects stalls from unchanged parquet/incomplete bytes+counts and restarts.
  - Optional completion exit can stop watchdog once expected file/byte targets are reached.
USAGE
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command not found: $cmd" >&2
    exit 1
  fi
}

DATASET=""
REPO_TYPE="dataset"
INCLUDE_PATTERN=""
LOCAL_DIR=""
MAX_WORKERS=4
ENABLE_HF_TRANSFER="auto"
SKIP_DRY_RUN=0
DRY_RUN_TIMEOUT_SECONDS=180
ATTEMPT_TIMEOUT_SECONDS=5400
RETRY_DELAY_SECONDS=30
MAX_RETRIES=0
WORKER_LOG_FILE="artifacts/reports/fineweb_350bt_download_resumable.log"
WATCHDOG_LOG_FILE="artifacts/reports/hf_download_watchdog.log"
CHECK_INTERVAL_SECONDS=120
STALL_SECONDS=1200
RESTART_DELAY_SECONDS=5
EXIT_ON_COMPLETE=0
EXPECTED_PARQUET_FILES=0
EXPECTED_BYTES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    --repo-type)
      REPO_TYPE="${2:-}"
      shift 2
      ;;
    --include)
      INCLUDE_PATTERN="${2:-}"
      shift 2
      ;;
    --local-dir)
      LOCAL_DIR="${2:-}"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="${2:-}"
      shift 2
      ;;
    --enable-hf-transfer)
      ENABLE_HF_TRANSFER="1"
      shift
      ;;
    --disable-hf-transfer)
      ENABLE_HF_TRANSFER="0"
      shift
      ;;
    --skip-dry-run)
      SKIP_DRY_RUN=1
      shift
      ;;
    --dry-run-timeout-seconds)
      DRY_RUN_TIMEOUT_SECONDS="${2:-}"
      shift 2
      ;;
    --attempt-timeout-seconds)
      ATTEMPT_TIMEOUT_SECONDS="${2:-}"
      shift 2
      ;;
    --retry-delay-seconds)
      RETRY_DELAY_SECONDS="${2:-}"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="${2:-}"
      shift 2
      ;;
    --worker-log-file)
      WORKER_LOG_FILE="${2:-}"
      shift 2
      ;;
    --watchdog-log-file)
      WATCHDOG_LOG_FILE="${2:-}"
      shift 2
      ;;
    --check-interval-seconds)
      CHECK_INTERVAL_SECONDS="${2:-}"
      shift 2
      ;;
    --stall-seconds)
      STALL_SECONDS="${2:-}"
      shift 2
      ;;
    --exit-on-complete)
      EXIT_ON_COMPLETE=1
      shift
      ;;
    --expected-parquet-files)
      EXPECTED_PARQUET_FILES="${2:-}"
      shift 2
      ;;
    --expected-bytes)
      EXPECTED_BYTES="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATASET" || -z "$INCLUDE_PATTERN" || -z "$LOCAL_DIR" ]]; then
  echo "error: --dataset, --include, and --local-dir are required" >&2
  usage >&2
  exit 2
fi

if ! [[ "$CHECK_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$CHECK_INTERVAL_SECONDS" -le 0 ]]; then
  echo "error: --check-interval-seconds must be a positive integer" >&2
  exit 2
fi
if ! [[ "$STALL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$STALL_SECONDS" -le 0 ]]; then
  echo "error: --stall-seconds must be a positive integer" >&2
  exit 2
fi
if ! [[ "$EXPECTED_PARQUET_FILES" =~ ^[0-9]+$ ]]; then
  echo "error: --expected-parquet-files must be an integer >= 0" >&2
  exit 2
fi
if ! [[ "$EXPECTED_BYTES" =~ ^[0-9]+$ ]]; then
  echo "error: --expected-bytes must be an integer >= 0" >&2
  exit 2
fi
if [[ "$EXIT_ON_COMPLETE" -eq 1 && "$EXPECTED_PARQUET_FILES" -le 0 && "$EXPECTED_BYTES" -le 0 ]]; then
  echo "error: --exit-on-complete requires --expected-parquet-files and/or --expected-bytes" >&2
  exit 2
fi

require_cmd awk
require_cmd date
require_cmd find
require_cmd flock
require_cmd kill
require_cmd mkdir
require_cmd sleep

mkdir -p "$LOCAL_DIR" "$(dirname "$WORKER_LOG_FILE")" "$(dirname "$WATCHDOG_LOG_FILE")"

WATCHDOG_LOCK_FILE="$LOCAL_DIR/.hf_download_watchdog.lock"
exec 8>"$WATCHDOG_LOCK_FILE"
if ! flock -n 8; then
  echo "error: another hf_download_watchdog instance is already running" >&2
  exit 3
fi

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '[%s] %s\n' "$ts" "$1" | tee -a "$WATCHDOG_LOG_FILE"
}

progress_snapshot() {
  local parquet_count incomplete_count total_bytes
  parquet_count="$(find "$LOCAL_DIR" -type f -name '*.parquet' | wc -l | tr -d ' ')"
  incomplete_count="$(find "$LOCAL_DIR" -type f -name '*.incomplete' | wc -l | tr -d ' ')"
  total_bytes="$(find "$LOCAL_DIR" -type f \( -name '*.parquet' -o -name '*.incomplete' \) -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
  printf '%s:%s:%s' "$parquet_count" "$incomplete_count" "$total_bytes"
}

parse_snapshot() {
  local snapshot="$1"
  local parquet_count incomplete_count total_bytes
  IFS=':' read -r parquet_count incomplete_count total_bytes <<< "$snapshot"
  echo "${parquet_count:-0} ${incomplete_count:-0} ${total_bytes:-0}"
}

clear_stale_lock() {
  local lock_file="$LOCAL_DIR/.hf_download_resumable.lock"
  if [[ ! -f "$lock_file" ]]; then
    return
  fi
  local lock_pid
  lock_pid="$(cat "$lock_file" 2>/dev/null || true)"
  if [[ -z "$lock_pid" ]]; then
    rm -f "$lock_file"
    log "removed empty worker lock file"
    return
  fi
  if ! kill -0 "$lock_pid" 2>/dev/null; then
    rm -f "$lock_file"
    log "removed stale worker lock pid=$lock_pid"
  fi
}

WORKER_PID=""

start_worker() {
  clear_stale_lock
  local -a cmd=(
    bash scripts/hf_download_resumable.sh
    --dataset "$DATASET"
    --repo-type "$REPO_TYPE"
    --include "$INCLUDE_PATTERN"
    --local-dir "$LOCAL_DIR"
    --max-workers "$MAX_WORKERS"
    --dry-run-timeout-seconds "$DRY_RUN_TIMEOUT_SECONDS"
    --attempt-timeout-seconds "$ATTEMPT_TIMEOUT_SECONDS"
    --retry-delay-seconds "$RETRY_DELAY_SECONDS"
    --max-retries "$MAX_RETRIES"
    --log-file "$WORKER_LOG_FILE"
  )

  if [[ "$ENABLE_HF_TRANSFER" == "1" ]]; then
    cmd+=(--enable-hf-transfer)
  elif [[ "$ENABLE_HF_TRANSFER" == "0" ]]; then
    cmd+=(--disable-hf-transfer)
  fi
  if [[ "$SKIP_DRY_RUN" -eq 1 ]]; then
    cmd+=(--skip-dry-run)
  fi

  # Launch worker in its own session so stop_worker can terminate the full process group.
  if command -v setsid >/dev/null 2>&1; then
    setsid "${cmd[@]}" >> "$WATCHDOG_LOG_FILE" 2>&1 &
  else
    "${cmd[@]}" >> "$WATCHDOG_LOG_FILE" 2>&1 &
  fi
  WORKER_PID="$!"
  log "worker_started pid=$WORKER_PID"
}

stop_worker() {
  local reason="$1"
  if [[ -z "$WORKER_PID" ]]; then
    return
  fi
  if ! kill -0 "$WORKER_PID" 2>/dev/null; then
    WORKER_PID=""
    return
  fi

  log "worker_stop pid=$WORKER_PID reason=$reason"
  kill -TERM "-$WORKER_PID" 2>/dev/null || kill "$WORKER_PID" 2>/dev/null || true
  sleep 10
  if kill -0 "$WORKER_PID" 2>/dev/null; then
    kill -KILL "-$WORKER_PID" 2>/dev/null || kill -9 "$WORKER_PID" 2>/dev/null || true
  fi
  wait "$WORKER_PID" 2>/dev/null || true
  WORKER_PID=""
  clear_stale_lock
}

cleanup() {
  stop_worker "watchdog_exit"
}
on_signal() {
  cleanup
  exit 0
}
trap cleanup EXIT
trap on_signal INT TERM

start_worker
last_snapshot="$(progress_snapshot)"
last_progress_epoch="$(date +%s)"
log "watchdog_start interval_seconds=$CHECK_INTERVAL_SECONDS stall_seconds=$STALL_SECONDS snapshot=$last_snapshot"

while true; do
  sleep "$CHECK_INTERVAL_SECONDS"

  current_snapshot="$(progress_snapshot)"
  now_epoch="$(date +%s)"

  if [[ "$current_snapshot" != "$last_snapshot" ]]; then
    last_snapshot="$current_snapshot"
    last_progress_epoch="$now_epoch"
    log "progress snapshot=$current_snapshot"
  fi

  if [[ "$EXIT_ON_COMPLETE" -eq 1 ]]; then
    read -r parquet_count incomplete_count total_bytes <<< "$(parse_snapshot "$current_snapshot")"
    complete=1
    if [[ "$EXPECTED_PARQUET_FILES" -gt 0 && "$parquet_count" -lt "$EXPECTED_PARQUET_FILES" ]]; then
      complete=0
    fi
    if [[ "$incomplete_count" -ne 0 ]]; then
      complete=0
    fi
    if [[ "$EXPECTED_BYTES" -gt 0 && "$total_bytes" -lt "$EXPECTED_BYTES" ]]; then
      complete=0
    fi
    if [[ "$complete" -eq 1 ]]; then
      log "completion_reached snapshot=$current_snapshot expected_parquet=$EXPECTED_PARQUET_FILES expected_bytes=$EXPECTED_BYTES exiting"
      stop_worker "completion_reached"
      exit 0
    fi
  fi

  if [[ -n "$WORKER_PID" ]] && ! kill -0 "$WORKER_PID" 2>/dev/null; then
    wait "$WORKER_PID" 2>/dev/null || true
    clear_stale_lock
    log "worker_exited snapshot=$current_snapshot restarting_after=${RESTART_DELAY_SECONDS}s"
    sleep "$RESTART_DELAY_SECONDS"
    start_worker
    last_snapshot="$(progress_snapshot)"
    last_progress_epoch="$(date +%s)"
    continue
  fi

  idle_seconds=$((now_epoch - last_progress_epoch))
  if [[ "$idle_seconds" -ge "$STALL_SECONDS" ]]; then
    log "stall_detected idle_seconds=$idle_seconds snapshot=$current_snapshot"
    stop_worker "stall_detected"
    sleep "$RESTART_DELAY_SECONDS"
    start_worker
    last_snapshot="$(progress_snapshot)"
    last_progress_epoch="$(date +%s)"
  fi
done

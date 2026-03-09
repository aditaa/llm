#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/fineweb_stage_shard_watchdog.sh [options]

Restart watchdog for scripts/fineweb_stage_shard_loop.sh.
- Restarts worker if it exits.
- Restarts worker if progress snapshot is unchanged for too long.

Options:
  --worker-args "..."            Args passed to fineweb_stage_shard_loop.sh
  --watchdog-log-file FILE       Watchdog log output file
  --check-interval-seconds N     Health/progress check interval (default: 120)
  --stall-seconds N              Restart worker after N seconds with no progress (default: 1800)
  --hot-parquet-dir DIR          Hot parquet directory for progress snapshot
  --shards-root DIR              Shards root for progress snapshot
  --processed-file FILE          Stage-loop processed parquet state file
  --no-cleanup-stale-workers     Do not terminate stale loop/shard workers before relaunch
  -h, --help                     Show help
USAGE
}

WORKER_ARGS="--hot-queue-min-files 10 --stage-max-files 8 --stage-copy-jobs 4 --stage-min-free-gib 80 --process-max-files 15 --shard-jobs 2 --auto-tune-shard-jobs --auto-tune-min-shard-jobs 2 --auto-tune-max-shard-jobs 3 --auto-tune-low-load-pct 80 --auto-tune-high-load-pct 95 --auto-tune-min-batch-seconds 300 --tokenizer-threads 10 --encode-batch-size 1024 --shard-size-tokens 20000000 --sync-background --sync-max-inflight 2 --sleep-seconds 60 --shard-min-batch-size 512"
WATCHDOG_LOG_FILE="artifacts/reports/fineweb_stage_shard_loop/watchdog.log"
CHECK_INTERVAL_SECONDS=120
STALL_SECONDS=1800
HOT_PARQUET_DIR="data/fineweb/sample-350BT/sample/350BT"
SHARDS_ROOT="data/shards_global/fineweb-global-bpe-v1"
PROCESSED_FILE="artifacts/reports/fineweb_stage_shard_loop/processed_parquet_files.txt"
CLEANUP_STALE_WORKERS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worker-args)
      WORKER_ARGS="${2:-}"
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
    --hot-parquet-dir)
      HOT_PARQUET_DIR="${2:-}"
      shift 2
      ;;
    --shards-root)
      SHARDS_ROOT="${2:-}"
      shift 2
      ;;
    --processed-file)
      PROCESSED_FILE="${2:-}"
      shift 2
      ;;
    --no-cleanup-stale-workers)
      CLEANUP_STALE_WORKERS=0
      shift
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

mkdir -p "$(dirname "$WATCHDOG_LOG_FILE")"
if ! command -v flock >/dev/null 2>&1; then
  echo "error: required command not found: flock" >&2
  exit 1
fi

LOCK_FILE="${WATCHDOG_LOG_FILE}.lock"
exec 8>"$LOCK_FILE"
if ! flock -n 8; then
  echo "error: another fineweb_stage_shard_watchdog instance is already running" >&2
  exit 3
fi

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '[%s] %s\n' "$ts" "$1" | tee -a "$WATCHDOG_LOG_FILE"
}

progress_snapshot() {
  local hot_count manifest_count processed_count
  hot_count="$(find "$HOT_PARQUET_DIR" -maxdepth 1 -type f -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
  manifest_count="$(find "$SHARDS_ROOT" -name manifest.json 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -f "$PROCESSED_FILE" ]]; then
    processed_count="$(awk 'NF {c+=1} END {print c+0}' "$PROCESSED_FILE")"
  else
    processed_count=0
  fi
  printf '%s:%s:%s' "$hot_count" "$manifest_count" "$processed_count"
}

WORKER_PID=""

terminate_pid_list() {
  local reason="$1"
  shift
  local pids=("$@")
  local pid
  for pid in "${pids[@]}"; do
    [[ -z "$pid" ]] && continue
    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if [[ "$pid" -eq "$$" ]]; then
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      log "stale_worker_stop pid=$pid reason=$reason"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${pids[@]}"; do
    [[ -z "$pid" ]] && continue
    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if [[ "$pid" -eq "$$" ]]; then
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
      log "stale_worker_killed pid=$pid reason=$reason"
    fi
  done
}

cleanup_stale_workers() {
  if [[ "$CLEANUP_STALE_WORKERS" -ne 1 ]]; then
    return
  fi
  local -a loop_pids=()
  local -a shard_pids=()
  local line pid

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    pid="${line%% *}"
    [[ "$pid" == "$$" ]] && continue
    loop_pids+=("$pid")
  done < <(pgrep -af "scripts/fineweb_stage_shard_loop.sh" || true)

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    pid="${line%% *}"
    [[ "$pid" == "$$" ]] && continue
    shard_pids+=("$pid")
  done < <(pgrep -af "scripts/fineweb_parquet_to_shards.py --input-dir $HOT_PARQUET_DIR" || true)

  if [[ "${#loop_pids[@]}" -gt 0 ]]; then
    terminate_pid_list "watchdog_cleanup_loop" "${loop_pids[@]}"
  fi
  if [[ "${#shard_pids[@]}" -gt 0 ]]; then
    terminate_pid_list "watchdog_cleanup_sharder" "${shard_pids[@]}"
  fi
}

start_worker() {
  cleanup_stale_workers
  # Launch worker in its own session so stop_worker can terminate the full process group.
  (
    # Prevent watchdog lock FD from being inherited by worker descendants.
    exec 8>&-
    if command -v setsid >/dev/null 2>&1; then
      exec setsid bash scripts/fineweb_stage_shard_loop.sh $WORKER_ARGS >> "$WATCHDOG_LOG_FILE" 2>&1
    else
      exec bash scripts/fineweb_stage_shard_loop.sh $WORKER_ARGS >> "$WATCHDOG_LOG_FILE" 2>&1
    fi
  ) &
  WORKER_PID="$!"
  log "worker_started pid=$WORKER_PID args=\"$WORKER_ARGS\""
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
  sleep 5
  if kill -0 "$WORKER_PID" 2>/dev/null; then
    kill -KILL "-$WORKER_PID" 2>/dev/null || kill -9 "$WORKER_PID" 2>/dev/null || true
  fi
  wait "$WORKER_PID" 2>/dev/null || true
  WORKER_PID=""
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

  if [[ -n "$WORKER_PID" ]] && ! kill -0 "$WORKER_PID" 2>/dev/null; then
    wait "$WORKER_PID" 2>/dev/null || true
    log "worker_exited snapshot=$current_snapshot restarting_after=5s"
    sleep 5
    start_worker
    last_snapshot="$(progress_snapshot)"
    last_progress_epoch="$(date +%s)"
    continue
  fi

  if (( now_epoch - last_progress_epoch >= STALL_SECONDS )); then
    log "worker_stalled elapsed=$((now_epoch - last_progress_epoch))s snapshot=$current_snapshot restarting"
    stop_worker "stall_timeout"
    sleep 5
    start_worker
    last_snapshot="$(progress_snapshot)"
    last_progress_epoch="$(date +%s)"
  fi
done

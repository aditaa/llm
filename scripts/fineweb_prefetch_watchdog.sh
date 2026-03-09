#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:-artifacts/reports/fineweb_prefetch_hot_queue/watchdog.log}"
QUEUE_MIN="${QUEUE_MIN:-40}"
STAGE_MAX="${STAGE_MAX:-20}"
COPY_JOBS="${COPY_JOBS:-12}"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"
SLEEP_SECONDS="${SLEEP_SECONDS:-20}"
CHECK_SECONDS="${CHECK_SECONDS:-30}"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

start_worker() {
  nohup bash scripts/fineweb_prefetch_hot_queue.sh \
    --queue-min-files "$QUEUE_MIN" \
    --stage-max-files "$STAGE_MAX" \
    --stage-copy-jobs "$COPY_JOBS" \
    --stage-min-age-seconds 120 \
    --min-free-gib "$MIN_FREE_GIB" \
    --sleep-seconds "$SLEEP_SECONDS" \
    --auto-skip-state-dir artifacts/reports/fineweb_stage_shard_loop \
    >> "$LOG_FILE" 2>&1 &
  echo $!
}

log "prefetch_watchdog_start queue_min=$QUEUE_MIN stage_max=$STAGE_MAX copy_jobs=$COPY_JOBS"
while true; do
  if ! pgrep -af '^bash scripts/fineweb_prefetch_hot_queue.sh' >/dev/null 2>&1; then
    pid="$(start_worker)"
    log "prefetch_worker_started pid=$pid"
  fi
  sleep "$CHECK_SECONDS"
done

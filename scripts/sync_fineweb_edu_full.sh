#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sync_fineweb_edu_full.sh [local_dir]

Defaults:
  local_dir: /mnt/pve/cephfs/llm/data/fineweb/fineweb-edu-full

Environment overrides:
  MAX_WORKERS=6
  RETRY_DELAY_SECONDS=30
  ATTEMPT_TIMEOUT_SECONDS=10800
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
MAX_WORKERS="${MAX_WORKERS:-6}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-30}"
ATTEMPT_TIMEOUT_SECONDS="${ATTEMPT_TIMEOUT_SECONDS:-10800}"
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

  set +e
  if command -v timeout >/dev/null 2>&1; then
    HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
      timeout "$ATTEMPT_TIMEOUT_SECONDS" "${cmd[@]}" >> "$LOG_FILE" 2>&1
    rc=$?
  else
    HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER="$HF_HUB_ENABLE_HF_TRANSFER" \
      "${cmd[@]}" >> "$LOG_FILE" 2>&1
    rc=$?
  fi
  set -e

  parquet_count="$(find "$LOCAL_DIR" -type f -name '*.parquet' | wc -l | tr -d ' ')"
  incomplete_count="$(find "$LOCAL_DIR" -type f -name '*.incomplete' | wc -l | tr -d ' ')"

  if [[ "$rc" -eq 0 && "$incomplete_count" -eq 0 ]]; then
    log "completed successfully parquet_files=${parquet_count}"
    exit 0
  fi

  log "attempt=${attempt} rc=${rc} parquet_files=${parquet_count} incomplete_files=${incomplete_count}; retry in ${RETRY_DELAY_SECONDS}s"
  sleep "$RETRY_DELAY_SECONDS"
  attempt=$((attempt + 1))
done

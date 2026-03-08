#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/hf_download_resumable.sh \
    --dataset HuggingFaceFW/fineweb \
    --repo-type dataset \
    --include "sample/350BT/*.parquet" \
    --local-dir /mnt/ceph/llm/data/fineweb/sample-350BT \
    [--max-workers 2] \
    [--enable-hf-transfer | --disable-hf-transfer] \
    [--skip-dry-run] \
    [--dry-run-timeout-seconds 180] \
    [--attempt-timeout-seconds 5400] \
    [--retry-delay-seconds 30] \
    [--max-retries 0] \
    [--log-file artifacts/reports/hf_download_resumable.log]

Notes:
  - Uses `hf download` and retries automatically on failures.
  - Resumes from partially downloaded files in `--local-dir`.
  - `--max-retries 0` means retry forever.
  - Auto-enables hf_transfer when installed (override with enable/disable flags).
  - Set `--skip-dry-run` if dry-run metadata is slow/unreliable.
  - `--attempt-timeout-seconds` bounds a single `hf download` call before retry.
  - If `HF_TOKEN` is set, `hf download` uses it from environment.
EOF
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
MAX_WORKERS=2
ENABLE_HF_TRANSFER="auto"
SKIP_DRY_RUN=0
DRY_RUN_TIMEOUT_SECONDS=180
ATTEMPT_TIMEOUT_SECONDS=5400
RETRY_DELAY_SECONDS=30
MAX_RETRIES=0
LOG_FILE="artifacts/reports/hf_download_resumable.log"

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
    --log-file)
      LOG_FILE="${2:-}"
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

require_cmd awk
require_cmd date
require_cmd find
require_cmd grep
require_cmd mkdir
require_cmd sed
require_cmd sleep

HF_BIN=".venv/bin/hf"
if [[ ! -x "$HF_BIN" ]]; then
  HF_BIN="hf"
fi
require_cmd "$HF_BIN"

PY_CMD=".venv/bin/python"
if [[ ! -x "$PY_CMD" ]]; then
  PY_CMD="python3"
fi
require_cmd "$PY_CMD"

mkdir -p "$(dirname "$LOG_FILE")" "$LOCAL_DIR"

HF_TRANSFER_ON=0
if [[ "$ENABLE_HF_TRANSFER" == "1" ]]; then
  HF_TRANSFER_ON=1
elif [[ "$ENABLE_HF_TRANSFER" == "auto" ]]; then
  if "$PY_CMD" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("hf_transfer") else 1)
PY
  then
    HF_TRANSFER_ON=1
  fi
fi

LOCK_FILE="${LOCAL_DIR}/.hf_download_resumable.lock"
if [[ -f "$LOCK_FILE" ]]; then
  prev_pid="$(cat "$LOCK_FILE" 2>/dev/null || true)"
  if [[ -n "${prev_pid:-}" ]] && kill -0 "$prev_pid" 2>/dev/null; then
    echo "error: another resumable download worker is active (pid=$prev_pid)" >&2
    exit 3
  fi
fi
echo "$$" > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

log() {
  local msg="$1"
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '[%s] %s\n' "$ts" "$msg" | tee -a "$LOG_FILE"
}
log "hf_transfer_enabled=$HF_TRANSFER_ON mode=$ENABLE_HF_TRANSFER"

expected_parquet_count=0
if [[ "$SKIP_DRY_RUN" -eq 0 ]]; then
  dry_run_cmd=(
    "$HF_BIN" download "$DATASET"
    --repo-type "$REPO_TYPE"
    --include "$INCLUDE_PATTERN"
    --local-dir "$LOCAL_DIR"
    --max-workers "$MAX_WORKERS"
    --dry-run
  )
  log "starting dry-run to estimate expected file count (timeout=${DRY_RUN_TIMEOUT_SECONDS}s)"
  if command -v timeout >/dev/null 2>&1; then
    set +e
    dry_out="$(timeout "$DRY_RUN_TIMEOUT_SECONDS" "${dry_run_cmd[@]}" 2>&1)"
    dry_rc=$?
    set -e
  else
    set +e
    dry_out="$("${dry_run_cmd[@]}" 2>&1)"
    dry_rc=$?
    set -e
  fi

  if [[ "$dry_rc" -eq 0 ]]; then
    expected_parquet_count="$(printf '%s\n' "$dry_out" | awk '/\* to download:/ {print $4}' | tail -n 1)"
    if [[ -z "$expected_parquet_count" || ! "$expected_parquet_count" =~ ^[0-9]+$ ]]; then
      expected_parquet_count=0
      log "dry-run completed but expected count was not parseable; will rely on hf exit status"
    else
      log "dry-run reports expected parquet files: $expected_parquet_count"
    fi
  elif [[ "$dry_rc" -eq 124 ]]; then
    log "dry-run timed out; continuing without expected file count"
    expected_parquet_count=0
  else
    log "dry-run failed; proceeding with retry loop anyway"
    expected_parquet_count=0
  fi
else
  log "skipping dry-run by request"
fi

attempt=1
while true; do
  if [[ "$MAX_RETRIES" -gt 0 && "$attempt" -gt "$MAX_RETRIES" ]]; then
    log "max retries reached ($MAX_RETRIES), exiting with failure"
    exit 10
  fi

  current_count="$(find "$LOCAL_DIR" -type f -name '*.parquet' | wc -l | tr -d ' ')"
  incomplete_count="$(find "$LOCAL_DIR" -type f -name '*.incomplete' | wc -l | tr -d ' ')"
  log "attempt=$attempt precheck parquet_files=$current_count incomplete_files=$incomplete_count"

  run_cmd=(
    "$HF_BIN" download "$DATASET"
    --repo-type "$REPO_TYPE"
    --include "$INCLUDE_PATTERN"
    --local-dir "$LOCAL_DIR"
    --max-workers "$MAX_WORKERS"
  )
  set +e
  if command -v timeout >/dev/null 2>&1; then
    HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER="$HF_TRANSFER_ON" timeout "$ATTEMPT_TIMEOUT_SECONDS" "${run_cmd[@]}" >> "$LOG_FILE" 2>&1
    rc=$?
  else
    HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER="$HF_TRANSFER_ON" "${run_cmd[@]}" >> "$LOG_FILE" 2>&1
    rc=$?
  fi
  set -e

  current_count="$(find "$LOCAL_DIR" -type f -name '*.parquet' | wc -l | tr -d ' ')"
  incomplete_count="$(find "$LOCAL_DIR" -type f -name '*.incomplete' | wc -l | tr -d ' ')"

  if [[ "$rc" -eq 0 ]]; then
    if [[ "$expected_parquet_count" -eq 0 ]]; then
      log "download command succeeded; parquet_files=$current_count incomplete_files=$incomplete_count"
      if [[ "$incomplete_count" -eq 0 ]]; then
        log "completed successfully"
        exit 0
      fi
      log "command succeeded but incomplete files remain; continuing"
    else
      if [[ "$current_count" -ge "$expected_parquet_count" && "$incomplete_count" -eq 0 ]]; then
        log "completed successfully: parquet_files=$current_count expected=$expected_parquet_count"
        exit 0
      fi
      log "command succeeded but completion check failed: parquet_files=$current_count expected=$expected_parquet_count incomplete_files=$incomplete_count"
    fi
  else
    if [[ "$rc" -eq 124 ]]; then
      log "download attempt timed out after ${ATTEMPT_TIMEOUT_SECONDS}s parquet_files=$current_count incomplete_files=$incomplete_count"
    else
      log "download failed with exit_code=$rc parquet_files=$current_count incomplete_files=$incomplete_count"
    fi
  fi

  delay=$(( RETRY_DELAY_SECONDS * attempt ))
  if [[ "$delay" -gt 900 ]]; then
    delay=900
  fi
  log "sleeping ${delay}s before retry"
  sleep "$delay"
  attempt=$((attempt + 1))
done

#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/mnt/ceph/llm/data/fineweb/sample-350BT/sample/350BT"
DEST_DIR="data/fineweb/sample-350BT/sample/350BT"
MAX_FILES=10
MAX_GIB=0
MIN_AGE_SECONDS=180
COPY_JOBS=1
MIN_FREE_GIB=0
LOCK_WAIT_SECONDS=0
RSYNC_RETRIES=3
RSYNC_CONNECT_TIMEOUT_SECONDS=30
RSYNC_TIMEOUT_SECONDS=900
SKIP_LIST=""
DRY_RUN=0
RSYNC_USE_CONTIMEOUT=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/stage_fineweb_from_warm.sh [options]

Copy completed FineWeb parquet chunks from warm storage to hot storage.
Only files older than --min-age-seconds are considered to avoid copying
actively written files.

Options:
  --src-dir DIR            Source parquet directory
                           (default: /mnt/ceph/llm/data/fineweb/sample-350BT/sample/350BT)
  --dest-dir DIR           Destination parquet directory
                           (default: data/fineweb/sample-350BT/sample/350BT)
  --max-files N            Max files to copy this run (default: 10)
  --max-gib N              Max total GiB to copy this run (default: 0 = unlimited)
  --min-age-seconds N      Ignore source files modified more recently than this
                           (default: 180)
  --copy-jobs N            Parallel copy workers for selected files (default: 1)
  --lock-wait-seconds N    Wait up to N seconds for stage lock (default: 0 skip if busy)
  --rsync-retries N        Retry attempts per file copy (default: 3)
  --rsync-connect-timeout-seconds N
                           Rsync connect timeout (default: 30)
  --rsync-timeout-seconds N
                           Rsync I/O timeout (default: 900)
  --min-free-gib N         Keep at least N GiB free on destination filesystem
                           after this staging run (default: 0 disabled)
  --skip-list FILE         Optional newline-separated parquet basenames to skip
  --dry-run                Print what would be copied without copying
  -h, --help               Show this help

Example:
  bash scripts/stage_fineweb_from_warm.sh --max-files 4 --max-gib 8 --copy-jobs 2 --min-free-gib 80
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-dir)
      SRC_DIR="$2"
      shift 2
      ;;
    --dest-dir)
      DEST_DIR="$2"
      shift 2
      ;;
    --max-files)
      MAX_FILES="$2"
      shift 2
      ;;
    --max-gib)
      MAX_GIB="$2"
      shift 2
      ;;
    --min-age-seconds)
      MIN_AGE_SECONDS="$2"
      shift 2
      ;;
    --copy-jobs)
      COPY_JOBS="$2"
      shift 2
      ;;
    --min-free-gib)
      MIN_FREE_GIB="$2"
      shift 2
      ;;
    --lock-wait-seconds)
      LOCK_WAIT_SECONDS="$2"
      shift 2
      ;;
    --rsync-retries)
      RSYNC_RETRIES="$2"
      shift 2
      ;;
    --rsync-connect-timeout-seconds)
      RSYNC_CONNECT_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --rsync-timeout-seconds)
      RSYNC_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --skip-list)
      SKIP_LIST="$2"
      shift 2
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

if [[ ! -d "$SRC_DIR" ]]; then
  echo "source directory not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR" artifacts/reports
log="artifacts/reports/stage_fineweb_from_warm_$(date +%Y%m%d_%H%M%S).log"
lock_key="$(printf '%s' "$DEST_DIR" | sha256sum | awk '{print $1}')"
lock_file="artifacts/reports/stage_fineweb_from_warm_${lock_key}.lock"
exec 9>"$lock_file"
if [[ "$LOCK_WAIT_SECONDS" -gt 0 ]]; then
  if ! flock -w "$LOCK_WAIT_SECONDS" 9; then
    echo "[$(date -Iseconds)] stage_lock_busy lock_file=$lock_file wait_seconds=$LOCK_WAIT_SECONDS action=skip" | tee -a "$log"
    exit 0
  fi
else
  if ! flock -n 9; then
    echo "[$(date -Iseconds)] stage_lock_busy lock_file=$lock_file wait_seconds=0 action=skip" | tee -a "$log"
    exit 0
  fi
fi

echo "[$(date -Iseconds)] source=$SRC_DIR" | tee -a "$log"
echo "[$(date -Iseconds)] dest=$DEST_DIR" | tee -a "$log"
echo "[$(date -Iseconds)] max_files=$MAX_FILES max_gib=$MAX_GIB min_age_seconds=$MIN_AGE_SECONDS copy_jobs=$COPY_JOBS min_free_gib=$MIN_FREE_GIB lock_wait_seconds=$LOCK_WAIT_SECONDS rsync_retries=$RSYNC_RETRIES rsync_connect_timeout_seconds=$RSYNC_CONNECT_TIMEOUT_SECONDS rsync_timeout_seconds=$RSYNC_TIMEOUT_SECONDS dry_run=$DRY_RUN skip_list=${SKIP_LIST:-none}" | tee -a "$log"

if [[ -n "$SKIP_LIST" && ! -f "$SKIP_LIST" ]]; then
  echo "skip-list not found: $SKIP_LIST" >&2
  exit 1
fi
if ! [[ "$COPY_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "copy-jobs must be a positive integer: $COPY_JOBS" >&2
  exit 2
fi
if ! [[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]]; then
  echo "min-free-gib must be an integer >= 0: $MIN_FREE_GIB" >&2
  exit 2
fi
for tuple in \
  "lock-wait-seconds:$LOCK_WAIT_SECONDS" \
  "rsync-retries:$RSYNC_RETRIES" \
  "rsync-connect-timeout-seconds:$RSYNC_CONNECT_TIMEOUT_SECONDS" \
  "rsync-timeout-seconds:$RSYNC_TIMEOUT_SECONDS"
do
  key="${tuple%%:*}"
  value="${tuple##*:}"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$key must be an integer >= 0: $value" >&2
    exit 2
  fi
done

if [[ "$SRC_DIR" == rsync://* || "$DEST_DIR" == rsync://* || "$SRC_DIR" == *"::"* || "$DEST_DIR" == *"::"* ]]; then
  RSYNC_USE_CONTIMEOUT=1
fi

now_epoch="$(date +%s)"
copied_files=0
copied_bytes=0
skipped_recent=0
skipped_existing=0
skipped_blocklist=0
considered=0
selected_entries=()
disk_guardrail_hit=0
free_start_bytes="$(df -PB1 "$DEST_DIR" | awk 'NR==2 {print $4}')"
min_free_bytes="$((MIN_FREE_GIB * 1024 * 1024 * 1024))"

copy_one() {
  local src_path="$1"
  local name="$2"
  local src_size="$3"
  local dst_path="$DEST_DIR/$name"
  local tmp_path="$DEST_DIR/${name}.incomplete"
  local tmp_size
  local attempt
  local max_attempts=$((RSYNC_RETRIES + 1))
  local -a rsync_args=(
    -ah
    --partial
    --inplace
    --timeout="$RSYNC_TIMEOUT_SECONDS"
  )
  if [[ "$RSYNC_USE_CONTIMEOUT" -eq 1 ]]; then
    rsync_args+=(--contimeout="$RSYNC_CONNECT_TIMEOUT_SECONDS")
  fi

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    rm -f "$tmp_path"
    if rsync "${rsync_args[@]}" "$src_path" "$tmp_path" >> "$log" 2>&1; then
      break
    fi
    rm -f "$tmp_path"
    if [[ "$attempt" -lt "$max_attempts" ]]; then
      echo "RETRY copy file=$name attempt=$attempt/$max_attempts" | tee -a "$log"
      sleep 2
      continue
    fi
    echo "FAIL copy error file=$name attempt=$attempt/$max_attempts" | tee -a "$log"
    return 1
  done

  tmp_size="$(stat -c%s "$tmp_path" || echo -1)"
  if [[ "$tmp_size" -ne "$src_size" ]]; then
    rm -f "$tmp_path"
    echo "FAIL size mismatch after temp copy file=$name src=$src_size tmp=$tmp_size" | tee -a "$log"
    return 1
  fi

  mv -f "$tmp_path" "$dst_path"
  echo "COPY ok $name size_bytes=$src_size" | tee -a "$log"
}

while IFS= read -r src_path; do
  [[ -z "$src_path" ]] && continue
  considered=$((considered + 1))

  name="$(basename "$src_path")"
  dst_path="$DEST_DIR/$name"
  src_size="$(stat -c%s "$src_path")"
  src_mtime="$(stat -c%Y "$src_path")"
  age="$((now_epoch - src_mtime))"

  if [[ -n "$SKIP_LIST" ]] && grep -Fqx "$name" "$SKIP_LIST"; then
    skipped_blocklist=$((skipped_blocklist + 1))
    continue
  fi

  if [[ "$age" -lt "$MIN_AGE_SECONDS" ]]; then
    skipped_recent=$((skipped_recent + 1))
    continue
  fi

  if [[ -f "$dst_path" ]]; then
    dst_size="$(stat -c%s "$dst_path" || echo -1)"
    if [[ "$dst_size" -eq "$src_size" ]]; then
      skipped_existing=$((skipped_existing + 1))
      continue
    fi
  fi

  if [[ "$MAX_FILES" -gt 0 && "$copied_files" -ge "$MAX_FILES" ]]; then
    break
  fi
  if [[ "$MAX_GIB" -gt 0 ]]; then
    max_bytes="$((MAX_GIB * 1024 * 1024 * 1024))"
    if [[ $((copied_bytes + src_size)) -gt "$max_bytes" ]]; then
      break
    fi
  fi
  if [[ "$MIN_FREE_GIB" -gt 0 ]]; then
    projected_free="$((free_start_bytes - copied_bytes - src_size))"
    if [[ "$projected_free" -lt "$min_free_bytes" ]]; then
      disk_guardrail_hit=1
      break
    fi
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN copy $name size_bytes=$src_size age_seconds=$age" | tee -a "$log"
    copied_files=$((copied_files + 1))
    copied_bytes=$((copied_bytes + src_size))
    continue
  fi

  copied_files=$((copied_files + 1))
  copied_bytes=$((copied_bytes + src_size))
  selected_entries+=("$src_path"$'\t'"$name"$'\t'"$src_size")
done < <(find "$SRC_DIR" -maxdepth 1 -type f -name '*.parquet' | sort)

if [[ "$DRY_RUN" -ne 1 && "$copied_files" -gt 0 ]]; then
  running=0
  failed=0
  for entry in "${selected_entries[@]}"; do
    IFS=$'\t' read -r src_path name src_size <<< "$entry"
    copy_one "$src_path" "$name" "$src_size" &
    running=$((running + 1))
    if [[ "$running" -ge "$COPY_JOBS" ]]; then
      if ! wait -n; then
        failed=1
      fi
      running=$((running - 1))
      if [[ "$failed" -ne 0 ]]; then
        break
      fi
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    kill $(jobs -p) 2>/dev/null || true
    while [[ "$running" -gt 0 ]]; do
      wait -n || true
      running=$((running - 1))
    done
    exit 1
  fi

  while [[ "$running" -gt 0 ]]; do
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  done

  if [[ "$failed" -ne 0 ]]; then
    exit 1
  fi
fi

copied_gib="$(awk -v b="$copied_bytes" 'BEGIN { printf "%.2f", b/1024/1024/1024 }')"
echo "[$(date -Iseconds)] considered=$considered copied_files=$copied_files copied_gib=$copied_gib skipped_recent=$skipped_recent skipped_existing=$skipped_existing skipped_blocklist=$skipped_blocklist" | tee -a "$log"
if [[ "$disk_guardrail_hit" -eq 1 ]]; then
  free_end_bytes="$(df -PB1 "$DEST_DIR" | awk 'NR==2 {print $4}')"
  free_end_gib="$(awk -v b="$free_end_bytes" 'BEGIN { printf "%.2f", b/1024/1024/1024 }')"
  echo "[$(date -Iseconds)] stop_reason=min_free_guardrail min_free_gib=$MIN_FREE_GIB free_gib_now=$free_end_gib" | tee -a "$log"
fi
echo "log_file=$log" | tee -a "$log"

#!/usr/bin/env bash
set -euo pipefail

WARM_PARQUET_DIR="/mnt/ceph/llm/data/fineweb/sample-350BT/sample/350BT"
HOT_PARQUET_DIR="data/fineweb/sample-350BT/sample/350BT"
QUEUE_MIN_FILES=12
STAGE_MAX_FILES=8
STAGE_MAX_GIB=0
STAGE_MIN_AGE_SECONDS=180
STAGE_COPY_JOBS=1
MIN_FREE_GIB=50
SLEEP_SECONDS=60
ITERATIONS=0
SKIP_LIST=""
AUTO_SKIP_STATE_DIR="artifacts/reports/fineweb_stage_shard_loop"
AUTO_SKIP=1
DRY_RUN=0
STATE_DIR="artifacts/reports/fineweb_prefetch_hot_queue"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/fineweb_prefetch_hot_queue.sh [options]

Maintain a warm->hot parquet queue by staging additional files when hot queue
count drops below --queue-min-files.

Options:
  --warm-parquet-dir DIR      Warm source parquet directory
  --hot-parquet-dir DIR       Hot parquet directory
  --queue-min-files N         Target minimum number of hot parquet files
                              (default: 12)
  --stage-max-files N         Max files to stage in one cycle (default: 8)
  --stage-copy-jobs N         Parallel warm->hot copy workers per stage cycle
                              (default: 1)
  --stage-max-gib N           Max GiB to stage in one cycle (default: 0 unlimited)
  --stage-min-age-seconds N   Ignore warm files newer than this age (default: 180)
  --min-free-gib N            Skip staging if hot filesystem free GiB is below this
                              threshold (default: 50)
  --sleep-seconds N           Sleep between cycles (default: 60)
  --iterations N              Number of cycles (default: 0 infinite)
  --skip-list FILE            Optional parquet basename skip list
  --auto-skip-state-dir DIR   Stage-loop state dir for auto skip list
                              (default: artifacts/reports/fineweb_stage_shard_loop)
  --no-auto-skip              Disable auto skip-list discovery/merge
  --state-dir DIR             Log/state directory
  --dry-run                   Print actions without copying files
  -h, --help                  Show help

Example:
  bash scripts/fineweb_prefetch_hot_queue.sh \
    --queue-min-files 16 \
    --stage-max-files 10 \
    --sleep-seconds 45
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
    --queue-min-files)
      QUEUE_MIN_FILES="$2"
      shift 2
      ;;
    --stage-max-files)
      STAGE_MAX_FILES="$2"
      shift 2
      ;;
    --stage-copy-jobs)
      STAGE_COPY_JOBS="$2"
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
    --min-free-gib)
      MIN_FREE_GIB="$2"
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
    --skip-list)
      SKIP_LIST="$2"
      shift 2
      ;;
    --auto-skip-state-dir)
      AUTO_SKIP_STATE_DIR="$2"
      shift 2
      ;;
    --no-auto-skip)
      AUTO_SKIP=0
      shift
      ;;
    --state-dir)
      STATE_DIR="$2"
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

if [[ ! -d "$WARM_PARQUET_DIR" ]]; then
  echo "error: warm parquet dir not found: $WARM_PARQUET_DIR" >&2
  exit 1
fi
if [[ ! -d "$HOT_PARQUET_DIR" ]]; then
  echo "error: hot parquet dir not found: $HOT_PARQUET_DIR" >&2
  exit 1
fi
if [[ "$QUEUE_MIN_FILES" -lt 1 ]]; then
  echo "error: queue-min-files must be >= 1" >&2
  exit 1
fi
if ! [[ "$STAGE_COPY_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: stage-copy-jobs must be a positive integer" >&2
  exit 1
fi
if [[ -n "$SKIP_LIST" && ! -f "$SKIP_LIST" ]]; then
  echo "error: skip-list not found: $SKIP_LIST" >&2
  exit 1
fi
if [[ "$SLEEP_SECONDS" -lt 1 ]]; then
  echo "error: sleep-seconds must be >= 1" >&2
  exit 1
fi

mkdir -p "$STATE_DIR"
LOCK_FILE="$STATE_DIR/prefetch.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "error: another fineweb_prefetch_hot_queue instance is already running" >&2
  exit 1
fi

LOG_FILE="$STATE_DIR/prefetch_$(date +%Y%m%d_%H%M%S).log"
AUTO_SKIP_FILE="$STATE_DIR/auto_skip_list.txt"
log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG_FILE"
}

hot_count() {
  find "$HOT_PARQUET_DIR" -maxdepth 1 -type f -name '*.parquet' | wc -l | tr -d ' '
}

hot_free_gib() {
  local avail_kib
  avail_kib="$(df -Pk "$HOT_PARQUET_DIR" | awk 'NR==2 {print $4}')"
  awk -v kib="$avail_kib" 'BEGIN { printf "%.2f", kib/1024/1024 }'
}

resolve_skip_list() {
  if [[ -n "$SKIP_LIST" ]]; then
    echo "$SKIP_LIST"
    return
  fi
  if [[ "$AUTO_SKIP" -ne 1 ]]; then
    echo ""
    return
  fi

  local direct_file="$AUTO_SKIP_STATE_DIR/stage_skip_list.txt"
  if [[ -f "$direct_file" ]]; then
    echo "$direct_file"
    return
  fi

  : > "$AUTO_SKIP_FILE"
  local src
  for src in \
    "$AUTO_SKIP_STATE_DIR/processed_parquet_files.txt" \
    "$AUTO_SKIP_STATE_DIR/bad_parquet_files.txt"
  do
    if [[ -s "$src" ]]; then
      cat "$src" >> "$AUTO_SKIP_FILE"
    fi
  done
  if [[ -s "$AUTO_SKIP_FILE" ]]; then
    sort -u "$AUTO_SKIP_FILE" -o "$AUTO_SKIP_FILE"
    echo "$AUTO_SKIP_FILE"
    return
  fi
  echo ""
}

cycle=0
log "prefetch_start warm=$WARM_PARQUET_DIR hot=$HOT_PARQUET_DIR queue_min_files=$QUEUE_MIN_FILES stage_max_files=$STAGE_MAX_FILES stage_copy_jobs=$STAGE_COPY_JOBS stage_max_gib=$STAGE_MAX_GIB min_free_gib=$MIN_FREE_GIB auto_skip=$AUTO_SKIP auto_skip_state_dir=$AUTO_SKIP_STATE_DIR dry_run=$DRY_RUN"

while true; do
  cycle=$((cycle + 1))
  current_hot="$(hot_count)"
  free_gib="$(hot_free_gib)"

  if awk "BEGIN {exit !($free_gib < $MIN_FREE_GIB)}"; then
    log "cycle=$cycle action=skip reason=low_free_space free_gib=$free_gib threshold=$MIN_FREE_GIB hot_files=$current_hot"
  elif [[ "$current_hot" -lt "$QUEUE_MIN_FILES" ]]; then
    need=$((QUEUE_MIN_FILES - current_hot))
    stage_files="$STAGE_MAX_FILES"
    if [[ "$stage_files" -gt "$need" ]]; then
      stage_files="$need"
    fi

    stage_cmd=(bash scripts/stage_fineweb_from_warm.sh
      --src-dir "$WARM_PARQUET_DIR"
      --dest-dir "$HOT_PARQUET_DIR"
      --max-files "$stage_files"
      --copy-jobs "$STAGE_COPY_JOBS"
      --max-gib "$STAGE_MAX_GIB"
      --min-age-seconds "$STAGE_MIN_AGE_SECONDS"
      --min-free-gib "$MIN_FREE_GIB"
    )
    resolved_skip_list="$(resolve_skip_list)"
    if [[ -n "$resolved_skip_list" ]]; then
      stage_cmd+=(--skip-list "$resolved_skip_list")
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
      stage_cmd+=(--dry-run)
    fi

    log "cycle=$cycle action=stage hot_files=$current_hot need=$need stage_files=$stage_files stage_copy_jobs=$STAGE_COPY_JOBS free_gib=$free_gib skip_list=${resolved_skip_list:-none}"
    "${stage_cmd[@]}" >> "$LOG_FILE" 2>&1 || {
      log "cycle=$cycle action=stage_failed"
    }
  else
    log "cycle=$cycle action=idle hot_files=$current_hot free_gib=$free_gib"
  fi

  if [[ "$ITERATIONS" -gt 0 && "$cycle" -ge "$ITERATIONS" ]]; then
    log "prefetch_done cycles=$cycle"
    break
  fi

  sleep "$SLEEP_SECONDS"
done

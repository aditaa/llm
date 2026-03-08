#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/mnt/ceph/llm/data/fineweb/sample-350BT/sample/350BT"
DEST_DIR="data/fineweb/sample-350BT/sample/350BT"
MAX_FILES=10
MAX_GIB=0
MIN_AGE_SECONDS=180
SKIP_LIST=""
DRY_RUN=0

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
  --skip-list FILE         Optional newline-separated parquet basenames to skip
  --dry-run                Print what would be copied without copying
  -h, --help               Show this help

Example:
  bash scripts/stage_fineweb_from_warm.sh --max-files 4 --max-gib 8
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

echo "[$(date -Iseconds)] source=$SRC_DIR" | tee -a "$log"
echo "[$(date -Iseconds)] dest=$DEST_DIR" | tee -a "$log"
echo "[$(date -Iseconds)] max_files=$MAX_FILES max_gib=$MAX_GIB min_age_seconds=$MIN_AGE_SECONDS dry_run=$DRY_RUN skip_list=${SKIP_LIST:-none}" | tee -a "$log"

if [[ -n "$SKIP_LIST" && ! -f "$SKIP_LIST" ]]; then
  echo "skip-list not found: $SKIP_LIST" >&2
  exit 1
fi

now_epoch="$(date +%s)"
copied_files=0
copied_bytes=0
skipped_recent=0
skipped_existing=0
skipped_blocklist=0
considered=0

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

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN copy $name size_bytes=$src_size age_seconds=$age" | tee -a "$log"
    copied_files=$((copied_files + 1))
    copied_bytes=$((copied_bytes + src_size))
    continue
  fi

  rsync -ah --partial --inplace "$src_path" "$DEST_DIR/" >> "$log" 2>&1
  dst_size="$(stat -c%s "$dst_path" || echo -1)"
  if [[ "$dst_size" -ne "$src_size" ]]; then
    echo "FAIL size mismatch after copy file=$name src=$src_size dst=$dst_size" | tee -a "$log"
    exit 1
  fi

  copied_files=$((copied_files + 1))
  copied_bytes=$((copied_bytes + src_size))
  echo "COPY ok $name size_bytes=$src_size" | tee -a "$log"
done < <(find "$SRC_DIR" -maxdepth 1 -type f -name '*.parquet' | sort)

copied_gib="$(awk -v b="$copied_bytes" 'BEGIN { printf "%.2f", b/1024/1024/1024 }')"
echo "[$(date -Iseconds)] considered=$considered copied_files=$copied_files copied_gib=$copied_gib skipped_recent=$skipped_recent skipped_existing=$skipped_existing skipped_blocklist=$skipped_blocklist" | tee -a "$log"
echo "log_file=$log" | tee -a "$log"

#!/usr/bin/env bash
set -u

SRC_DIR="${1:-data/raw_zim}"
DST_DIR="${2:-/mnt/ceph/llm/data/raw_zim}"
SLEEP_SECS="${3:-120}"

mkdir -p "$SRC_DIR" "$DST_DIR"

while true; do
  found=0
  while IFS= read -r zim_path; do
    found=1
    zim_name="$(basename "$zim_path")"
    echo "$(date -Is) offload_start $zim_name"
    if rsync -ah --partial --inplace --append-verify --remove-source-files \
      "$zim_path" "$DST_DIR/$zim_name"; then
      echo "$(date -Is) offload_done $zim_name"
    else
      echo "$(date -Is) offload_fail $zim_name"
    fi
  done < <(find "$SRC_DIR" -maxdepth 1 -type f -name "*.zim" | sort)

  if [ "$found" -eq 0 ]; then
    echo "$(date -Is) idle_no_local_zims"
    sleep "$SLEEP_SECS"
  fi

  find "$SRC_DIR" -maxdepth 1 -type f -name "*.zim" -size 0 -delete || true
done

#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-/mnt/ceph/llm/data}"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "error: source mount does not exist: ${SRC_ROOT}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync is required but not installed" >&2
  exit 1
fi

mkdir -p \
  data/raw_zim \
  data/fineweb \
  data/cleaned \
  data/extracted \
  data/shards \
  data/shards_global \
  artifacts/tokenizer \
  artifacts/checkpoints \
  artifacts/reports

sync_dir() {
  local src="$1"
  local dst="$2"
  if [[ -d "${src}" ]]; then
    echo "hydrate ${src} -> ${dst}"
    rsync -ah --info=stats2,progress2 "${src}/" "${dst}/"
  else
    echo "skip ${src} (not found)"
  fi
}

sync_dir "${SRC_ROOT}/raw_zim" "data/raw_zim"
sync_dir "${SRC_ROOT}/fineweb" "data/fineweb"
sync_dir "${SRC_ROOT}/cleaned" "data/cleaned"
sync_dir "${SRC_ROOT}/extracted" "data/extracted"
sync_dir "${SRC_ROOT}/shards" "data/shards"
sync_dir "${SRC_ROOT}/shards_global" "data/shards_global"
sync_dir "${SRC_ROOT}/tokenizer" "artifacts/tokenizer"
sync_dir "${SRC_ROOT}/checkpoints" "artifacts/checkpoints"
sync_dir "${SRC_ROOT}/reports" "artifacts/reports"

echo "hydrate complete"

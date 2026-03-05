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

mkdir -p data/extracted data/shards artifacts/tokenizer data/raw_zim

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
sync_dir "${SRC_ROOT}/extracted" "data/extracted"
sync_dir "${SRC_ROOT}/shards" "data/shards"
sync_dir "${SRC_ROOT}/tokenizer" "artifacts/tokenizer"

echo "hydrate complete"

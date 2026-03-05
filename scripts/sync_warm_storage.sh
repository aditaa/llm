#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT="${1:-/mnt/ceph/llm/data}"

if [[ ! -d "${DEST_ROOT}" ]]; then
  echo "error: destination mount does not exist: ${DEST_ROOT}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync is required but not installed" >&2
  exit 1
fi

echo "sync destination: ${DEST_ROOT}"
mkdir -p \
  "${DEST_ROOT}/raw_zim" \
  "${DEST_ROOT}/extracted" \
  "${DEST_ROOT}/shards" \
  "${DEST_ROOT}/tokenizer" \
  "${DEST_ROOT}/logs"

sync_dir() {
  local src="$1"
  local dst="$2"
  local extra_args="${3:-}"
  if [[ -d "${src}" ]]; then
    echo "sync ${src} -> ${dst}"
    # shellcheck disable=SC2086
    rsync -ah --info=stats2,progress2 ${extra_args} "${src}/" "${dst}/"
  else
    echo "skip ${src} (not found)"
  fi
}

if find data -maxdepth 2 -type f -name "*.zim" | grep -q .; then
  echo "sync raw .zim files -> ${DEST_ROOT}/raw_zim"
  while IFS= read -r zim_path; do
    rsync -ah --info=stats2,progress2 "${zim_path}" "${DEST_ROOT}/raw_zim/"
  done < <(find data -maxdepth 2 -type f -name "*.zim" | sort)
else
  echo "skip raw .zim sync (no .zim files found under data/)"
fi

sync_dir "data/extracted" "${DEST_ROOT}/extracted" "--exclude=*.zim"
sync_dir "data/shards" "${DEST_ROOT}/shards"
sync_dir "artifacts/tokenizer" "${DEST_ROOT}/tokenizer"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${DEST_ROOT}/logs/last_sync_utc.txt"
echo "sync complete"

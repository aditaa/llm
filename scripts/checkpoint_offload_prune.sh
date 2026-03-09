#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/checkpoint_offload_prune.sh [options]

Sync local checkpoint runs to warm storage and prune older local runs.

Options:
  --local-checkpoints-dir DIR   Local checkpoints root (default: artifacts/checkpoints)
  --warm-checkpoints-dir DIR    Warm checkpoints root (default: /mnt/ceph/llm/data/checkpoints)
  --keep-local-runs N           Keep newest N non-active local runs (default: 1)
  --sync-only                   Only sync to warm storage; do not prune local runs
  --dry-run                     Print planned actions without rsync/delete
  -h, --help                    Show help

Notes:
  - Active training output dirs are always kept locally.
  - Prune is only allowed after warm/local byte counts match.
USAGE
}

LOCAL_CHECKPOINTS_DIR="artifacts/checkpoints"
WARM_CHECKPOINTS_DIR="/mnt/ceph/llm/data/checkpoints"
KEEP_LOCAL_RUNS=1
SYNC_ONLY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-checkpoints-dir)
      LOCAL_CHECKPOINTS_DIR="${2:-}"
      shift 2
      ;;
    --warm-checkpoints-dir)
      WARM_CHECKPOINTS_DIR="${2:-}"
      shift 2
      ;;
    --keep-local-runs)
      KEEP_LOCAL_RUNS="${2:-}"
      shift 2
      ;;
    --sync-only)
      SYNC_ONLY=1
      shift
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
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$KEEP_LOCAL_RUNS" =~ ^[0-9]+$ ]]; then
  echo "error: --keep-local-runs must be a non-negative integer" >&2
  exit 2
fi

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$1"
}

to_abs() {
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$1"
  else
    readlink -f "$1" 2>/dev/null || echo "$1"
  fi
}

LOCAL_CHECKPOINTS_DIR="$(to_abs "$LOCAL_CHECKPOINTS_DIR")"
WARM_CHECKPOINTS_DIR="$(to_abs "$WARM_CHECKPOINTS_DIR")"

if [[ ! -d "$LOCAL_CHECKPOINTS_DIR" ]]; then
  log "skip local_checkpoints_missing path=$LOCAL_CHECKPOINTS_DIR"
  exit 0
fi

if [[ "$DRY_RUN" -ne 1 ]]; then
  mkdir -p "$WARM_CHECKPOINTS_DIR"
else
  log "dry_run mkdir -p $WARM_CHECKPOINTS_DIR"
fi

declare -a run_names=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  run_names+=("$line")
done < <(find "$LOCAL_CHECKPOINTS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null | sort -nr | awk '{print $2}')

if [[ "${#run_names[@]}" -eq 0 ]]; then
  log "skip no_local_runs"
  exit 0
fi

declare -A active_keep=()
while IFS= read -r outdir; do
  [[ -z "$outdir" ]] && continue
  out_abs="$(to_abs "$outdir")"
  if [[ "$out_abs" == "$LOCAL_CHECKPOINTS_DIR/"* ]]; then
    active_keep["$(basename "$out_abs")"]=1
  else
    candidate="$LOCAL_CHECKPOINTS_DIR/$(basename "$outdir")"
    if [[ -d "$candidate" ]]; then
      active_keep["$(basename "$candidate")"]=1
    fi
  fi
done < <(ps -eo args= | awk '
  /llm\.cli train/ && /--output-dir/ {
    for (i = 1; i <= NF; i++) {
      if ($i == "--output-dir" && i < NF) {
        print $(i + 1)
      }
    }
  }
')

declare -A keep_names=()
for name in "${!active_keep[@]}"; do
  keep_names["$name"]=1
done

kept_recent=0
for name in "${run_names[@]}"; do
  if [[ -n "${keep_names[$name]+x}" ]]; then
    continue
  fi
  if [[ "$kept_recent" -lt "$KEEP_LOCAL_RUNS" ]]; then
    keep_names["$name"]=1
    kept_recent=$((kept_recent + 1))
  fi
done

log "plan runs=${#run_names[@]} keep_local_runs=$KEEP_LOCAL_RUNS active_kept=${#active_keep[@]} sync_only=$SYNC_ONLY dry_run=$DRY_RUN"

sync_one() {
  local name="$1"
  local src="$LOCAL_CHECKPOINTS_DIR/$name"
  local dst="$WARM_CHECKPOINTS_DIR/$name"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry_run rsync -a $src/ $dst/"
    return
  fi
  mkdir -p "$dst"
  rsync -a "$src/" "$dst/"
}

dir_bytes() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo 0
    return
  fi
  du -sb "$path" 2>/dev/null | awk '{print $1}'
}

prune_one() {
  local name="$1"
  local src="$LOCAL_CHECKPOINTS_DIR/$name"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry_run prune_local path=$src"
    return
  fi
  find "$src" -type f -delete
  find "$src" -depth -type d -empty -delete
  log "pruned_local path=$src"
}

for name in "${run_names[@]}"; do
  sync_one "$name"
done

if [[ "$SYNC_ONLY" -eq 1 ]]; then
  log "done mode=sync_only"
  exit 0
fi

for name in "${run_names[@]}"; do
  if [[ -n "${keep_names[$name]+x}" ]]; then
    log "keep_local name=$name"
    continue
  fi

  local_dir="$LOCAL_CHECKPOINTS_DIR/$name"
  warm_dir="$WARM_CHECKPOINTS_DIR/$name"
  local_bytes="$(dir_bytes "$local_dir")"
  warm_bytes="$(dir_bytes "$warm_dir")"
  if [[ "$local_bytes" -eq 0 ]]; then
    log "skip_prune name=$name reason=local_missing_or_empty"
    continue
  fi
  if [[ "$local_bytes" -ne "$warm_bytes" ]]; then
    log "skip_prune name=$name reason=size_mismatch local_bytes=$local_bytes warm_bytes=$warm_bytes"
    continue
  fi
  prune_one "$name"
done

log "done"

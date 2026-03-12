#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/checkpoint_step_offload.sh [options]

Offload older checkpoint step files (ckpt_step_*.pt) to warm storage while
keeping the newest step files local for fast resume.

Options:
  --local-checkpoints-dir DIR   Local checkpoints root (default: artifacts/checkpoints)
  --warm-checkpoints-dir DIR    Warm checkpoints root (default: /mnt/ceph/llm/data/checkpoints)
  --keep-last-steps N           Keep newest N step checkpoints per run locally (default: 6)
  --max-files N                 Maximum candidate files to process this run (default: 24, 0 = unlimited)
  --run-name NAME               Restrict to one run directory name (repeatable)
  --dry-run                     Print planned actions without copy/delete
  -h, --help                    Show help

Notes:
  - Only ckpt_step_*.pt files are offloaded.
  - last.pt, best.pt, rollback checkpoints, and safetensors are untouched.
  - A file is deleted locally only after warm copy size matches.
USAGE
}

LOCAL_CHECKPOINTS_DIR="artifacts/checkpoints"
WARM_CHECKPOINTS_DIR="/mnt/ceph/llm/data/checkpoints"
KEEP_LAST_STEPS=6
MAX_FILES=24
DRY_RUN=0
RUN_NAMES=()

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
    --keep-last-steps)
      KEEP_LAST_STEPS="${2:-}"
      shift 2
      ;;
    --max-files)
      MAX_FILES="${2:-}"
      shift 2
      ;;
    --run-name)
      RUN_NAMES+=("${2:-}")
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
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$KEEP_LAST_STEPS" =~ ^[0-9]+$ ]]; then
  echo "error: --keep-last-steps must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$MAX_FILES" =~ ^[0-9]+$ ]]; then
  echo "error: --max-files must be a non-negative integer" >&2
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

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

candidates_tsv="$tmp_dir/candidates.tsv"
: > "$candidates_tsv"

declare -a run_dirs=()
if [[ "${#RUN_NAMES[@]}" -gt 0 ]]; then
  for run_name in "${RUN_NAMES[@]}"; do
    run_dir="$LOCAL_CHECKPOINTS_DIR/$run_name"
    if [[ -d "$run_dir" ]]; then
      run_dirs+=("$run_dir")
    else
      log "skip run_missing name=$run_name path=$run_dir"
    fi
  done
else
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    run_dirs+=("$line")
  done < <(find "$LOCAL_CHECKPOINTS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ "${#run_dirs[@]}" -eq 0 ]]; then
  log "skip no_run_dirs"
  exit 0
fi

extract_step_num() {
  local fname="$1"
  local stem="${fname#ckpt_step_}"
  stem="${stem%.pt}"
  echo $((10#$stem))
}

for run_dir in "${run_dirs[@]}"; do
  run_name="$(basename "$run_dir")"
  mapfile -t entries < <(
    find "$run_dir" -maxdepth 1 -type f -name 'ckpt_step_*.pt' -printf '%f\t%p\t%s\n' | sort -t$'\t' -k1,1
  )
  count="${#entries[@]}"
  if [[ "$count" -le "$KEEP_LAST_STEPS" ]]; then
    log "skip run=$run_name reason=insufficient_step_files count=$count keep_last_steps=$KEEP_LAST_STEPS"
    continue
  fi

  candidate_count=$((count - KEEP_LAST_STEPS))
  for ((idx=0; idx<candidate_count; idx++)); do
    IFS=$'\t' read -r fname fpath fsize <<<"${entries[$idx]}"
    step_num="$(extract_step_num "$fname")"
    printf '%d\t%s\t%s\t%s\n' "$step_num" "$fpath" "$run_name" "$fsize" >> "$candidates_tsv"
  done
done

if [[ ! -s "$candidates_tsv" ]]; then
  log "skip no_step_candidates"
  exit 0
fi

sort -n -t$'\t' -k1,1 "$candidates_tsv" > "$tmp_dir/candidates.sorted.tsv"

processed=0
copied=0
removed=0
skipped_existing=0
failed=0

while IFS=$'\t' read -r step_num src run_name src_size; do
  [[ -z "$src" ]] && continue
  if [[ "$MAX_FILES" -gt 0 && "$processed" -ge "$MAX_FILES" ]]; then
    break
  fi
  processed=$((processed + 1))

  if [[ ! -f "$src" ]]; then
    log "skip source_missing run=$run_name step=$step_num path=$src"
    failed=$((failed + 1))
    continue
  fi

  dst_dir="$WARM_CHECKPOINTS_DIR/$run_name"
  dst="$dst_dir/$(basename "$src")"
  dst_tmp="${dst}.tmp_offload"
  action="copy"

  if [[ -f "$dst" ]]; then
    dst_size="$(stat -c%s "$dst" 2>/dev/null || echo -1)"
    if [[ "$dst_size" -eq "$src_size" ]]; then
      action="reuse"
    fi
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry_run step_offload run=$run_name step=$step_num action=$action src=$src dst=$dst"
    continue
  fi

  mkdir -p "$dst_dir"
  if [[ "$action" == "copy" ]]; then
    rm -f "$dst_tmp"
    if ! rsync -a "$src" "$dst_tmp"; then
      log "copy_fail run=$run_name step=$step_num src=$src dst_tmp=$dst_tmp"
      failed=$((failed + 1))
      rm -f "$dst_tmp"
      continue
    fi
    mv -f "$dst_tmp" "$dst"
    copied=$((copied + 1))
  else
    skipped_existing=$((skipped_existing + 1))
  fi

  dst_size_after="$(stat -c%s "$dst" 2>/dev/null || echo -1)"
  if [[ "$dst_size_after" -ne "$src_size" ]]; then
    log "verify_fail run=$run_name step=$step_num src_size=$src_size dst_size=$dst_size_after src=$src dst=$dst"
    failed=$((failed + 1))
    continue
  fi

  rm -f "$src"
  removed=$((removed + 1))
  log "offload_done run=$run_name step=$step_num src_removed=1 dst=$dst"
done < "$tmp_dir/candidates.sorted.tsv"

log "done processed=$processed copied=$copied removed=$removed reused_existing=$skipped_existing failed=$failed dry_run=$DRY_RUN keep_last_steps=$KEEP_LAST_STEPS max_files=$MAX_FILES"

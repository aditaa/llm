#!/usr/bin/env bash
set -euo pipefail

# Safe shard offload cycle:
# 1) Reconcile offloaded manifests (restore untrained / under-coverage)
# 2) Run gated shard offload
# 3) Reconcile again to self-heal mismatches

OFFLOAD_ARGS_DEFAULT="--shards-root data/shards_global/fineweb-global-bpe-v1 --warm-shards-root /mnt/ceph/llm/data/shards_global/fineweb-global-bpe-v1 --keep-local-batches 24 --target-free-gib 180 --max-batches 24 --disable-offloaded-manifests --require-trained-batches-file artifacts/reports/train_supervisor_phase1_talk/trained_batch_names.txt,artifacts/reports/train_supervisor_350bt/trained_batch_names.txt --skip-if-trained-file-missing --min-active-manifests 48 --min-active-train-tokens 40000000000"
RECONCILE_ARGS_DEFAULT="--shards-root data/shards_global/fineweb-global-bpe-v1 --trained-batches-file artifacts/reports/train_supervisor_phase1_talk/trained_batch_names.txt,artifacts/reports/train_supervisor_350bt/trained_batch_names.txt --skip-if-trained-file-missing --rehydrate-active-symlink-bins"

OFFLOAD_ARGS_RAW="${LLM_SHARD_OFFLOAD_ARGS:-$OFFLOAD_ARGS_DEFAULT}"
RECONCILE_ARGS_RAW="${LLM_SHARD_OFFLOAD_RECONCILE_ARGS:-$RECONCILE_ARGS_DEFAULT}"

expand_args() {
  local raw="$1"
  local -n out_ref="$2"
  out_ref=()
  eval "set -- $raw"
  while [[ $# -gt 0 ]]; do
    out_ref+=("$1")
    shift
  done
}

run_cycle() {
  local phase="$1"
  local -a reconcile_args=()
  expand_args "$RECONCILE_ARGS_RAW" reconcile_args
  local report="artifacts/reports/offload_reconcile_${phase}_$(date +%Y%m%d_%H%M%S).json"
  echo "[$(date -Iseconds)] offload_cycle_reconcile phase=$phase args=${RECONCILE_ARGS_RAW} report=$report"
  PYTHONPATH=src .venv/bin/python scripts/reconcile_offloaded_manifests.py \
    "${reconcile_args[@]}" \
    --report-output "$report"
}

run_cycle "pre"

declare -a offload_args=()
expand_args "$OFFLOAD_ARGS_RAW" offload_args

shards_root_from_args() {
  local default_root="data/shards_global/fineweb-global-bpe-v1"
  local idx
  for ((idx = 0; idx < ${#offload_args[@]}; idx++)); do
    if [[ "${offload_args[$idx]}" == "--shards-root" ]] && ((idx + 1 < ${#offload_args[@]})); then
      echo "${offload_args[$((idx + 1))]}"
      return 0
    fi
  done
  echo "$default_root"
}

echo "[$(date -Iseconds)] offload_cycle_offload args=${OFFLOAD_ARGS_RAW}"
PYTHONPATH=src .venv/bin/python scripts/offload_shard_bins_to_warm.py "${offload_args[@]}"

run_cycle "post"

SHARDS_ROOT="$(shards_root_from_args)"
echo "[$(date -Iseconds)] offload_cycle_enforce_hot_only shards_root=$SHARDS_ROOT"
PYTHONPATH=src .venv/bin/python scripts/enforce_hot_only_manifests.py --shards-root "$SHARDS_ROOT"

echo "[$(date -Iseconds)] offload_cycle_done"

#!/usr/bin/env bash
set -uo pipefail

cd /root/llm

STATE_DIR="/media/llm/data/reports/fineweb_edu_shard_loop"
LOG_FILE="$STATE_DIR/shard_guard.log"
mkdir -p "$STATE_DIR"

while true; do
  echo "[$(date -Iseconds)] shard_guard_start" >> "$LOG_FILE"
  set +e
  bash scripts/fineweb_edu_shard_loop.sh \
    --source-root /media/llm/data/fineweb/fineweb-edu-full/data \
    --shards-root /media/llm/data/shards_global/fineweb-global-bpe-v1 \
    --tokenizer-path /media/llm/data/tokenizer/fineweb-global-bpe-v1.json \
    --state-dir /media/llm/data/reports/fineweb_edu_shard_loop \
    --python-bin /root/llm/.venv/bin/python \
    --job-prefix fwedu \
    --process-max-files 32 \
    --shard-jobs 5 \
    --tokenizer-threads 6 \
    --encode-batch-size 2048 \
    --shard-size-tokens 20000000 \
    --sleep-seconds 45 >> "$LOG_FILE" 2>&1
  rc=$?
  set -e
  echo "[$(date -Iseconds)] shard_guard_exit rc=$rc" >> "$LOG_FILE"
  if [[ "$rc" -eq 3 ]]; then
    sleep 120
  else
    sleep 20
  fi
done

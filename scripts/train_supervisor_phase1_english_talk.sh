#!/usr/bin/env bash
set -euo pipefail

# Phase-1 profile: optimize for English conversational quality before code specialization.
# You can override any default by appending flags when invoking this script.
exec bash scripts/train_supervisor_rtx5070_350bt.sh \
  --step-chunk 2000 \
  --poll-seconds 60 \
  --batch-size 12 \
  --target-effective-batch 24 \
  --min-train-tokens 30000000000 \
  --min-batch-size 6 \
  --max-batch-size 20 \
  --batch-step 2 \
  --checkpoint-keep-last 6 \
  --checkpoint-keep-every 10000 \
  --ema-decay 0.999 \
  --dedupe-report-keep 240 \
  --eval-suite configs/eval/english_talk_suite_v1.json \
  --eval-promotion-policy configs/eval/promotion_policy_talk_v1.json \
  --generation-suite configs/eval/generation_talk_smoke_v1.json \
  --generation-every-chunks 1 \
  --no-train-fail-on-eval-regression \
  "$@"

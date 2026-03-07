#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-}"
DEST_DIR="${2:-}"
TOKEN="${HF_TOKEN:-}"

if [[ -z "$REPO_ID" || -z "$DEST_DIR" ]]; then
  echo "Usage: bash scripts/hf_download_model.sh <repo-id> <dest-dir> [hf-token-env]" >&2
  echo "Example: bash scripts/hf_download_model.sh aditaa/llm-from-scratch-v1 /srv/models/llm-v1" >&2
  exit 1
fi

if [[ ! -x ".venv/bin/hf" ]]; then
  echo "error: .venv/bin/hf not found. Install train tooling first." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
echo "Downloading model snapshot: ${REPO_ID} -> ${DEST_DIR}"

if [[ -n "$TOKEN" ]]; then
  HF_HUB_DISABLE_XET=1 .venv/bin/hf download "$REPO_ID" \
    --repo-type model \
    --local-dir "$DEST_DIR" \
    --token "$TOKEN"
else
  HF_HUB_DISABLE_XET=1 .venv/bin/hf download "$REPO_ID" \
    --repo-type model \
    --local-dir "$DEST_DIR"
fi

echo "download_complete=$DEST_DIR"

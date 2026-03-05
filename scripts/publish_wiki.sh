#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WIKI_SRC="${REPO_ROOT}/wiki"
WIKI_URL="${1:-git@github.com:aditaa/llm.wiki.git}"
TMP_DIR="$(mktemp -d)"
WIKI_DIR="${TMP_DIR}/llm.wiki"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if [[ ! -d "${WIKI_SRC}" ]]; then
  echo "error: wiki source directory not found: ${WIKI_SRC}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync is required but not installed" >&2
  exit 1
fi

echo "cloning wiki repo: ${WIKI_URL}"
git clone "${WIKI_URL}" "${WIKI_DIR}"

echo "syncing wiki pages from ${WIKI_SRC}"
rsync -ah --delete --include='*.md' --exclude='*' "${WIKI_SRC}/" "${WIKI_DIR}/"

cd "${WIKI_DIR}"
if [[ -z "$(git status --porcelain)" ]]; then
  echo "no wiki changes to publish"
  exit 0
fi

git add .
git commit -m "Update wiki from repo docs ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
git push
echo "wiki publish complete"

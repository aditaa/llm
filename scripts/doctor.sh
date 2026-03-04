#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ROOT_DIR}/.venv"

echo "== Binary checks =="
for bin in git python3 rg pdftotext; do
  if command -v "${bin}" >/dev/null 2>&1; then
    echo "ok: ${bin}"
  else
    echo "missing: ${bin}"
  fi
done

if [[ -x "${VENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${VENV_PATH}/bin/python"
  echo "Using virtualenv Python: ${PYTHON_BIN}"
else
  PYTHON_BIN="python3"
  echo ".venv not found, using system Python: ${PYTHON_BIN}"
fi

echo "== Python package checks =="
"${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["black", "mypy", "ruff", "numpy", "torch", "tiktoken", "matplotlib", "pandas", "tqdm"]
for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"ok: {mod}")
    except Exception:
        print(f"missing: {mod}")
PY

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ROOT_DIR}/.venv"

if ! python3 -m pip --version >/dev/null 2>&1; then
  echo "pip not found; installing user-level pip with get-pip.py."
  wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
  python3 /tmp/get-pip.py --user --break-system-packages
fi

if ! python3 -m venv "${VENV_PATH}"; then
  echo "python3 -m venv failed; trying virtualenv fallback."
  python3 -m pip install --user --break-system-packages virtualenv
  python3 -m virtualenv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements/server-python-dev.txt"

git -C "${ROOT_DIR}" submodule update --init --recursive

echo "Bootstrap complete."
echo "Activate with: source .venv/bin/activate"

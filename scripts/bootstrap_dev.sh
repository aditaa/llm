#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ROOT_DIR}/.venv"

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements/server-python-dev.txt"

git -C "${ROOT_DIR}" submodule update --init --recursive

echo "Bootstrap complete."
echo "Activate with: source .venv/bin/activate"

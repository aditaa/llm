#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo ".venv not found. Run scripts/bootstrap_dev.sh first."
  exit 1
fi

source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements/server-python-train.txt"

echo "Training extras installed."

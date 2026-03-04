#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_LIST="${ROOT_DIR}/requirements/server-system-ubuntu.txt"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found. This script currently supports Ubuntu/Debian."
  exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "Run as root or install sudo."
    exit 1
  fi
else
  SUDO=""
fi

${SUDO} apt-get update
${SUDO} xargs -a "${PACKAGE_LIST}" apt-get install -y

echo "System packages installed from ${PACKAGE_LIST}"

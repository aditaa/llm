#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sudo bash scripts/set_swappiness.sh [options]

Set Linux vm.swappiness for LLM training hosts.

Options:
  --value N             Swappiness value 0-100 (default: 10)
  --persist             Also write /etc/sysctl.d/99-llm-swappiness.conf
  --sysctl-file FILE    Override persistent file path
  -h, --help            Show help
USAGE
}

VALUE=10
PERSIST=0
SYSCTL_FILE="/etc/sysctl.d/99-llm-swappiness.conf"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --value)
      VALUE="${2:-}"
      shift 2
      ;;
    --persist)
      PERSIST=1
      shift
      ;;
    --sysctl-file)
      SYSCTL_FILE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$VALUE" =~ ^[0-9]+$ ]] || [[ "$VALUE" -lt 0 || "$VALUE" -gt 100 ]]; then
  echo "error: --value must be an integer in [0,100]" >&2
  exit 2
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "error: root required (run with sudo)" >&2
  exit 1
fi

sysctl -w "vm.swappiness=$VALUE"

if [[ "$PERSIST" -eq 1 ]]; then
  mkdir -p "$(dirname "$SYSCTL_FILE")"
  printf 'vm.swappiness = %s\n' "$VALUE" > "$SYSCTL_FILE"
  sysctl --system >/dev/null
  echo "persisted vm.swappiness=$VALUE file=$SYSCTL_FILE"
fi

echo "done vm.swappiness=$VALUE"

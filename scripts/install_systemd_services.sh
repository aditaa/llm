#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="$(id -un)"
SYSTEMD_DIR="/etc/systemd/system"
ENV_TARGET="/etc/llm/llm.env"
ENABLE=1
START=1
INSTALL_WATCHDOG=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/install_systemd_services.sh [options]

Install and optionally enable/start systemd service units for long-running
LLM pipeline workers.

Options:
  --repo-dir DIR            Repository directory baked into unit files
  --user NAME               Service user (default: current user)
  --systemd-dir DIR         Unit install directory (default: /etc/systemd/system)
  --env-target FILE         Environment file path (default: /etc/llm/llm.env)
  --install-watchdog        Also install/enable HF watchdog service unit
  --no-enable               Do not run systemctl enable
  --no-start                Do not run systemctl restart/start
  -h, --help                Show help

Examples:
  bash scripts/install_systemd_services.sh
  bash scripts/install_systemd_services.sh --install-watchdog
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --user)
      RUN_USER="$2"
      shift 2
      ;;
    --systemd-dir)
      SYSTEMD_DIR="$2"
      shift 2
      ;;
    --env-target)
      ENV_TARGET="$2"
      shift 2
      ;;
    --install-watchdog)
      INSTALL_WATCHDOG=1
      shift
      ;;
    --no-enable)
      ENABLE=0
      shift
      ;;
    --no-start)
      START=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "$REPO_DIR" ]]; then
  echo "error: repo dir not found: $REPO_DIR" >&2
  exit 1
fi

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "error: need root privileges (run as root or install sudo)" >&2
    exit 1
  fi
fi

$SUDO mkdir -p "$SYSTEMD_DIR"
$SUDO mkdir -p "$(dirname "$ENV_TARGET")"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

install_unit() {
  local src_name="$1"
  local dst_name="$2"
  sed -e "s#__REPO_DIR__#$REPO_DIR#g" -e "s#__USER__#$RUN_USER#g" \
    "$REPO_DIR/deploy/systemd/$src_name" > "$tmp_dir/$dst_name"
  $SUDO install -m 0644 "$tmp_dir/$dst_name" "$SYSTEMD_DIR/$dst_name"
  echo "installed_unit=$SYSTEMD_DIR/$dst_name"
}

install_unit "llm-train-supervisor.service" "llm-train-supervisor.service"
install_unit "llm-fineweb-prefetch.service" "llm-fineweb-prefetch.service"
if [[ "$INSTALL_WATCHDOG" -eq 1 ]]; then
  install_unit "llm-hf-download-watchdog.service" "llm-hf-download-watchdog.service"
fi

if [[ ! -f "$ENV_TARGET" ]]; then
  $SUDO install -m 0644 "$REPO_DIR/deploy/systemd/llm.env.example" "$ENV_TARGET"
  echo "installed_env_template=$ENV_TARGET"
else
  echo "env_exists=$ENV_TARGET"
fi

$SUDO systemctl daemon-reload

units=(llm-train-supervisor.service llm-fineweb-prefetch.service)
if [[ "$INSTALL_WATCHDOG" -eq 1 ]]; then
  units+=(llm-hf-download-watchdog.service)
fi

if [[ "$ENABLE" -eq 1 ]]; then
  $SUDO systemctl enable "${units[@]}"
  echo "enabled_units=${units[*]}"
fi

if [[ "$START" -eq 1 ]]; then
  for unit in "${units[@]}"; do
    $SUDO systemctl restart "$unit"
  done
  echo "restarted_units=${units[*]}"
fi

echo "done=1"

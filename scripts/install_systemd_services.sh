#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="$(id -un)"
SYSTEMD_DIR="/etc/systemd/system"
ENV_TARGET="/etc/llm/llm.env"
ENABLE=1
START=1
INSTALL_WATCHDOG=0
INSTALL_PREFETCH=0
INSTALL_MAINTENANCE=1

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/install_systemd_services.sh [options]

Install and optionally enable/start systemd service units for long-running
LLM pipeline workers (supervisor + stage/shard watchdog, optional prefetch/HF watchdog).
Also installs maintenance units (checkpoint offload/prune timer + bad-parquet
revalidate timer + shard-offload timer + VM swappiness tune service).

Options:
  --repo-dir DIR            Repository directory baked into unit files
  --user NAME               Service user (default: current user)
  --systemd-dir DIR         Unit install directory (default: /etc/systemd/system)
  --env-target FILE         Environment file path (default: /etc/llm/llm.env)
  --install-prefetch        Also install/enable prefetch service unit
  --install-watchdog        Also install/enable HF watchdog service unit
  --no-maintenance          Skip maintenance units (offload/revalidate/shard-offload timers + VM tuning)
  --no-enable               Do not run systemctl enable
  --no-start                Do not run systemctl restart/start
  -h, --help                Show help

Examples:
  bash scripts/install_systemd_services.sh
  bash scripts/install_systemd_services.sh --install-watchdog --install-prefetch
  bash scripts/install_systemd_services.sh --no-maintenance
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
    --install-prefetch)
      INSTALL_PREFETCH=1
      shift
      ;;
    --no-maintenance)
      INSTALL_MAINTENANCE=0
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
install_unit "llm-fineweb-stage-shard-loop.service" "llm-fineweb-stage-shard-loop.service"
install_unit "llm-fineweb-stage-shard-watchdog.service" "llm-fineweb-stage-shard-watchdog.service"
if [[ "$INSTALL_PREFETCH" -eq 1 ]]; then
  install_unit "llm-fineweb-prefetch.service" "llm-fineweb-prefetch.service"
fi
if [[ "$INSTALL_WATCHDOG" -eq 1 ]]; then
  install_unit "llm-hf-download-watchdog.service" "llm-hf-download-watchdog.service"
fi
if [[ "$INSTALL_MAINTENANCE" -eq 1 ]]; then
  install_unit "llm-checkpoint-offload-prune.service" "llm-checkpoint-offload-prune.service"
  install_unit "llm-checkpoint-offload-prune.timer" "llm-checkpoint-offload-prune.timer"
  install_unit "llm-bad-parquet-revalidate.service" "llm-bad-parquet-revalidate.service"
  install_unit "llm-bad-parquet-revalidate.timer" "llm-bad-parquet-revalidate.timer"
  install_unit "llm-shard-offload.service" "llm-shard-offload.service"
  install_unit "llm-shard-offload.timer" "llm-shard-offload.timer"
  install_unit "llm-vm-swappiness.service" "llm-vm-swappiness.service"
fi

if [[ ! -f "$ENV_TARGET" ]]; then
  $SUDO install -m 0644 "$REPO_DIR/deploy/systemd/llm.env.example" "$ENV_TARGET"
  echo "installed_env_template=$ENV_TARGET"
else
  echo "env_exists=$ENV_TARGET"
fi

$SUDO systemctl daemon-reload

units=(
  llm-train-supervisor.service
  llm-fineweb-stage-shard-watchdog.service
)
timer_units=()
if [[ "$INSTALL_PREFETCH" -eq 1 ]]; then
  units+=(llm-fineweb-prefetch.service)
fi
if [[ "$INSTALL_WATCHDOG" -eq 1 ]]; then
  units+=(llm-hf-download-watchdog.service)
fi
if [[ "$INSTALL_MAINTENANCE" -eq 1 ]]; then
  units+=(llm-vm-swappiness.service)
  timer_units+=(llm-checkpoint-offload-prune.timer llm-bad-parquet-revalidate.timer llm-shard-offload.timer)
fi

if [[ "$ENABLE" -eq 1 ]]; then
  $SUDO systemctl enable "${units[@]}"
  if [[ "${#timer_units[@]}" -gt 0 ]]; then
    $SUDO systemctl enable "${timer_units[@]}"
  fi
  echo "enabled_units=${units[*]}"
  if [[ "${#timer_units[@]}" -gt 0 ]]; then
    echo "enabled_timers=${timer_units[*]}"
  fi
fi

if [[ "$START" -eq 1 ]]; then
  for unit in "${units[@]}"; do
    $SUDO systemctl restart "$unit"
  done
  for unit in "${timer_units[@]}"; do
    $SUDO systemctl restart "$unit"
  done
  echo "restarted_units=${units[*]}"
  if [[ "${#timer_units[@]}" -gt 0 ]]; then
    echo "restarted_timers=${timer_units[*]}"
  fi
fi

echo "done=1"

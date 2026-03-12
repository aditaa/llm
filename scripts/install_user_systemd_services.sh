#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
USER_SYSTEMD_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
ENV_TARGET="${XDG_CONFIG_HOME:-$HOME/.config}/llm/llm.env"
ENABLE=1
START=1
INSTALL_WATCHDOG=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/install_user_systemd_services.sh [options]

Install user-level systemd units for persistent LLM pipeline workers.
This path does not require root and is suitable when sudo/system units are unavailable.

Options:
  --repo-dir DIR            Repository directory baked into unit files
  --user-systemd-dir DIR    Unit install directory (default: ~/.config/systemd/user)
  --env-target FILE         Environment file path (default: ~/.config/llm/llm.env)
  --install-watchdog        Also install/enable HF download watchdog unit
  --no-enable               Do not run systemctl --user enable
  --no-start                Do not run systemctl --user restart/start
  -h, --help                Show help

Installed by default:
  - llm-train-supervisor.service
  - llm-fineweb-stage-shard-watchdog.service
  - llm-shard-offload.service + llm-shard-offload.timer
  - llm-checkpoint-offload-prune.service + llm-checkpoint-offload-prune.timer
  - llm-checkpoint-step-offload.service + llm-checkpoint-step-offload.timer
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --user-systemd-dir)
      USER_SYSTEMD_DIR="$2"
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

if ! command -v systemctl >/dev/null 2>&1; then
  echo "error: systemctl not found" >&2
  exit 1
fi

mkdir -p "$USER_SYSTEMD_DIR"
mkdir -p "$(dirname "$ENV_TARGET")"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

install_unit() {
  local src_name="$1"
  local dst_name="$2"
  sed -e "s#__REPO_DIR__#$REPO_DIR#g" \
    "$REPO_DIR/deploy/systemd/user/$src_name" > "$tmp_dir/$dst_name"
  install -m 0644 "$tmp_dir/$dst_name" "$USER_SYSTEMD_DIR/$dst_name"
  echo "installed_unit=$USER_SYSTEMD_DIR/$dst_name"
}

run_user_systemctl() {
  local action="$1"
  shift
  local out
  if out="$(systemctl --user "$action" "$@" 2>&1)"; then
    return 0
  fi
  if grep -qi "transient or generated" <<<"$out"; then
    echo "warn=skip_${action}_transient_unit unit=$*"
    return 0
  fi
  echo "$out" >&2
  return 1
}

install_unit "llm-train-supervisor.service" "llm-train-supervisor.service"
install_unit "llm-fineweb-stage-shard-watchdog.service" "llm-fineweb-stage-shard-watchdog.service"
install_unit "llm-shard-offload.service" "llm-shard-offload.service"
install_unit "llm-shard-offload.timer" "llm-shard-offload.timer"
install_unit "llm-checkpoint-offload-prune.service" "llm-checkpoint-offload-prune.service"
install_unit "llm-checkpoint-offload-prune.timer" "llm-checkpoint-offload-prune.timer"
install_unit "llm-checkpoint-step-offload.service" "llm-checkpoint-step-offload.service"
install_unit "llm-checkpoint-step-offload.timer" "llm-checkpoint-step-offload.timer"
if [[ "$INSTALL_WATCHDOG" -eq 1 ]]; then
  install_unit "llm-hf-download-watchdog.service" "llm-hf-download-watchdog.service"
fi

if [[ ! -f "$ENV_TARGET" ]]; then
  install -m 0644 "$REPO_DIR/deploy/systemd/llm.env.example" "$ENV_TARGET"
  echo "installed_env_template=$ENV_TARGET"
else
  echo "env_exists=$ENV_TARGET"
fi

systemctl --user daemon-reload

units=(
  llm-train-supervisor.service
  llm-fineweb-stage-shard-watchdog.service
)
timer_units=(
  llm-shard-offload.timer
  llm-checkpoint-offload-prune.timer
  llm-checkpoint-step-offload.timer
)
if [[ "$INSTALL_WATCHDOG" -eq 1 ]]; then
  units+=(llm-hf-download-watchdog.service)
fi

if [[ "$ENABLE" -eq 1 ]]; then
  for unit in "${units[@]}"; do
    run_user_systemctl enable "$unit"
  done
  for unit in "${timer_units[@]}"; do
    run_user_systemctl enable "$unit"
  done
  echo "enabled_units=${units[*]}"
  echo "enabled_timers=${timer_units[*]}"
fi

if [[ "$START" -eq 1 ]]; then
  for unit in "${units[@]}"; do
    run_user_systemctl restart "$unit"
  done
  for unit in "${timer_units[@]}"; do
    run_user_systemctl restart "$unit"
  done
  echo "restarted_units=${units[*]}"
  echo "restarted_timers=${timer_units[*]}"
fi

echo "hint=For reboot persistence without active login, ask root to run: loginctl enable-linger $(id -un)"
echo "done=1"

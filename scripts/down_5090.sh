#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONTAINER="${CONTAINER:-fish-speech}"
RUN_DIR="$REPO_ROOT/run"

PROXY_PID_FILE="$RUN_DIR/proxy.pid"
SESSION_PID_FILE="$RUN_DIR/session_mode.pid"

STOP_MODEL="${STOP_MODEL:-1}"
STOP_PROXY="${STOP_PROXY:-1}"
STOP_SESSION="${STOP_SESSION:-1}"

docker_cmd() {
  if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
    sudo docker "$@"
  else
    docker "$@"
  fi
}

wait_pid_exit() {
  local pid="$1"
  local label="$2"

  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    sleep 0.25
  done

  echo "Force killing ${label} pid=${pid}"
  kill -9 "$pid" 2>/dev/null || true
}

stop_from_pidfile() {
  local pid_file="$1"
  local label="$2"
  local pattern="$3"

  echo "Stopping ${label}"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait_pid_exit "$pid" "$label"
    fi
    rm -f "$pid_file"
  fi

  pkill -f "$pattern" 2>/dev/null || true
}

if [[ "$STOP_PROXY" == "1" ]]; then
  stop_from_pidfile "$PROXY_PID_FILE" "proxy" 'tools.proxy.fish_proxy_pcm:app'
else
  echo "Skipping proxy stop"
fi

if [[ "$STOP_SESSION" == "1" ]]; then
  stop_from_pidfile "$SESSION_PID_FILE" "session_mode" 'session_mode.app:app'
else
  echo "Skipping session_mode stop"
fi

if [[ "$STOP_MODEL" == "1" ]]; then
  echo "Stopping model container"
  docker_cmd rm -f "$CONTAINER" 2>/dev/null || true
else
  echo "Skipping model container stop"
fi

echo "Stop sequence completed"
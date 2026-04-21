#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONTAINER="${CONTAINER:-fish-speech}"
RUN_DIR="$REPO_ROOT/run"
PID_FILE="$RUN_DIR/proxy.pid"

docker_cmd() {
  if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
    sudo docker "$@"
  else
    docker "$@"
  fi
}

echo "[1/2] Stopping proxy"
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
      kill -9 "$PID" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
fi
pkill -f 'uvicorn.*tools.proxy.fish_proxy_pcm:app' 2>/dev/null || true

echo "[2/2] Stopping model container"
docker_cmd rm -f "$CONTAINER" 2>/dev/null || true

echo "All stopped"
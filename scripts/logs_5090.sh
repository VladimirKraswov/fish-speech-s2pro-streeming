#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONTAINER="${CONTAINER:-fish-speech}"
PROXY_LOG_FILE="${PROXY_LOG_FILE:-$REPO_ROOT/logs/proxy.log}"
SESSION_LOG_FILE="${SESSION_LOG_FILE:-$REPO_ROOT/logs/session_mode.log}"

mkdir -p "$REPO_ROOT/logs"
touch "$PROXY_LOG_FILE"

docker_cmd() {
  if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
    sudo docker "$@"
  else
    docker "$@"
  fi
}

cleanup() {
  [[ -n "${MODEL_PID:-}" ]] && kill "$MODEL_PID" 2>/dev/null || true
  [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null || true
  [[ -n "${SESSION_PID:-}" ]] && kill "$SESSION_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Live logs ==="
echo "model container: $CONTAINER"
echo "proxy log: $PROXY_LOG_FILE"
echo "session log: $SESSION_LOG_FILE"
echo

(
  docker_cmd logs -f "$CONTAINER" 2>&1 | sed 's/^/[model] /'
) &
MODEL_PID=$!

(
  tail -n 50 -F "$PROXY_LOG_FILE" 2>/dev/null | sed 's/^/[proxy] /'
) &
PROXY_PID=$!

(
  tail -n 50 -F "$SESSION_LOG_FILE" 2>/dev/null | sed 's/^/[session] /'
) &
SESSION_PID=$!

wait
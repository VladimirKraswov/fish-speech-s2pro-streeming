#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

CONTAINER="${CONTAINER:-fish-speech-5090}"
SERVER_PORT="${SERVER_PORT:-$(runtime_get 'network.server.port')}"
PROXY_PORT="${PROXY_PORT:-$(runtime_get 'network.proxy.port')}"
PID_FILE="$(proxy_pid_file)"
WEBUI_PID_FILE="$(webui_pid_file)"

echo "=== Fish Speech status ==="
echo "runtime:   $RUNTIME_CONFIG"
echo "container: $CONTAINER"
echo

echo "[docker]"
docker_cmd ps -a --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
echo

echo "[server health]"
curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/health" || echo "server: unavailable"
echo
echo

echo "[proxy health]"
curl -sf "http://127.0.0.1:${PROXY_PORT}/health" || echo "proxy: unavailable"
echo
echo

echo "[web ui health]"
curl -sf "http://127.0.0.1:9001/health" || echo "web-ui: unavailable"
echo
echo

echo "[proxy pid]"
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "proxy pid: $PID (running)"
  else
    echo "proxy pid file exists, but process is not running"
  fi
else
  echo "proxy pid file not found"
fi
echo

echo "[web ui pid]"
if [[ -f "$WEBUI_PID_FILE" ]]; then
  PID="$(cat "$WEBUI_PID_FILE" 2>/dev/null || true)"
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "web ui pid: $PID (running)"
  else
    echo "web ui pid file exists, but process is not running"
  fi
else
  echo "web ui pid file not found"
fi
echo

echo "[gpu memory]"
curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/debug/memory" || echo "memory endpoint unavailable"
echo
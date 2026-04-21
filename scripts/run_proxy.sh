#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PROXY_PORT="${PROXY_PORT:-9000}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
PROXY_LOG_LEVEL="${PROXY_LOG_LEVEL:-info}"

LOG_DIR="$REPO_ROOT/logs"
RUN_DIR="$REPO_ROOT/run"
LOG_FILE="$LOG_DIR/proxy.log"
PID_FILE="$RUN_DIR/proxy.pid"

mkdir -p "$LOG_DIR" "$RUN_DIR"

if [[ -d "$REPO_ROOT/.venv-proxy" ]]; then
  VENV_PATH="$REPO_ROOT/.venv-proxy"
elif [[ -d "$REPO_ROOT/.venv" ]]; then
  VENV_PATH="$REPO_ROOT/.venv"
else
  echo "ERROR: virtualenv not found (.venv-proxy or .venv)" >&2
  exit 1
fi

pkill -f 'uvicorn.*tools.proxy.fish_proxy_pcm:app' 2>/dev/null || true
rm -f "$PID_FILE"

set +u
source "$VENV_PATH/bin/activate"
set -u

nohup uvicorn tools.proxy.fish_proxy_pcm:app \
  --host "$PROXY_HOST" \
  --port "$PROXY_PORT" \
  --log-level "$PROXY_LOG_LEVEL" \
  >> "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

sleep 1

if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "ERROR: proxy failed to start. Check $LOG_FILE" >&2
  exit 1
fi

echo "Proxy started"
echo "  pid:  $(cat "$PID_FILE")"
echo "  log:  $LOG_FILE"
echo "  url:  http://127.0.0.1:${PROXY_PORT}/health"
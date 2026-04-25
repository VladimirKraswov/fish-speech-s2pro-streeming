#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

WEBUI_HOST="${WEBUI_HOST:-0.0.0.0}"
WEBUI_PORT="${WEBUI_PORT:-9001}"
WEBUI_LOG_LEVEL="${WEBUI_LOG_LEVEL:-info}"
WEBUI_START_TIMEOUT="${WEBUI_START_TIMEOUT:-60}"

LOG_DIR="$REPO_ROOT/logs"
RUN_DIR="$REPO_ROOT/run"
LOG_FILE="$(webui_log_file)"
PID_FILE="$(webui_pid_file)"

mkdir -p "$LOG_DIR" "$RUN_DIR"
touch "$LOG_FILE"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$OLD_PID" 2>/dev/null; then
      kill -9 "$OLD_PID" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

nohup "$PYTHON_BIN" -m fish_speech_web_ui.server \
  --host "$WEBUI_HOST" \
  --port "$WEBUI_PORT" \
  >> "$LOG_FILE" 2>&1 &

WEBUI_PID=$!
echo "$WEBUI_PID" > "$PID_FILE"

echo "Waiting for Web UI to become healthy..."
if ! wait_http_ok "http://127.0.0.1:${WEBUI_PORT}/health" "$WEBUI_START_TIMEOUT" 1; then
  echo "ERROR: Web UI failed to become healthy" >&2
  tail -n 100 "$LOG_FILE" >&2 || true
  rm -f "$PID_FILE"
  exit 1
fi

echo "Web UI started"
echo "  pid: $WEBUI_PID"
echo "  url: http://127.0.0.1:${WEBUI_PORT}"
echo "  log: $LOG_FILE"

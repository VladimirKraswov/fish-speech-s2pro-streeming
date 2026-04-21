#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SESSION_PORT="${SESSION_PORT:-8765}"
SESSION_HOST="${SESSION_HOST:-0.0.0.0}"
SESSION_LOG_LEVEL="${SESSION_LOG_LEVEL:-info}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="$REPO_ROOT/logs"
RUN_DIR="$REPO_ROOT/run"
LOG_FILE="$LOG_DIR/session_mode.log"
PID_FILE="$RUN_DIR/session_mode.pid"
VENV_PATH="$REPO_ROOT/.venv-session"

mkdir -p "$LOG_DIR" "$RUN_DIR"
touch "$LOG_FILE"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Creating session_mode virtualenv at $VENV_PATH"
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

set +u
source "$VENV_PATH/bin/activate"
set -u

echo "Installing/checking session_mode dependencies in $VENV_PATH"
python -m pip install -U pip setuptools wheel >/dev/null
python -m pip install fastapi "uvicorn[standard]" httpx loguru pydantic pydantic-settings >/dev/null

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$PID_FILE"
fi

pkill -f 'uvicorn.*session_mode.app:app' 2>/dev/null || true

echo "Starting session_mode manager..."
nohup python -m uvicorn session_mode.app:app \
  --host "$SESSION_HOST" \
  --port "$SESSION_PORT" \
  --log-level "$SESSION_LOG_LEVEL" \
  >> "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

sleep 1

if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "ERROR: session_mode failed to start. Check $LOG_FILE" >&2
  exit 1
fi

echo "Session Mode Manager started"
echo "  pid: $(cat "$PID_FILE")"
echo "  log: $LOG_FILE"
echo "  url: http://127.0.0.1:${SESSION_PORT}/health"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PROXY_PORT="${PROXY_PORT:-9000}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
PROXY_LOG_LEVEL="${PROXY_LOG_LEVEL:-info}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="$REPO_ROOT/logs"
RUN_DIR="$REPO_ROOT/run"
LOG_FILE="$LOG_DIR/proxy.log"
PID_FILE="$RUN_DIR/proxy.pid"
VENV_PATH="$REPO_ROOT/.venv-proxy"

mkdir -p "$LOG_DIR" "$RUN_DIR"
touch "$LOG_FILE"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Creating proxy virtualenv at $VENV_PATH"
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

set +u
source "$VENV_PATH/bin/activate"
set -u

if ! python - <<'PY' >/dev/null 2>&1
import fastapi, uvicorn, httpx
print("ok")
PY
then
  echo "Installing proxy dependencies into $VENV_PATH"
  python -m pip install -U pip setuptools wheel
  python -m pip install fastapi "uvicorn[standard]" httpx
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$PID_FILE"
fi

pkill -f 'uvicorn.*tools.proxy.fish_proxy_pcm:app' 2>/dev/null || true

nohup python -m uvicorn tools.proxy.fish_proxy_pcm:app \
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
echo "  pid: $(cat "$PID_FILE")"
echo "  log: $LOG_FILE"
echo "  url: http://127.0.0.1:${PROXY_PORT}/health"
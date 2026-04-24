#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PROXY_PORT="${PROXY_PORT:-9000}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
PROXY_LOG_LEVEL="${PROXY_LOG_LEVEL:-info}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROXY_START_TIMEOUT="${PROXY_START_TIMEOUT:-60}"
DEFAULT_REFERENCE_ID="${DEFAULT_REFERENCE_ID:-voice}"
UPSTREAM_TTS_URL="${UPSTREAM_TTS_URL:-http://127.0.0.1:8080/v1/tts}"
SESSION_TTL_SEC="${SESSION_TTL_SEC:-1800}"
SESSION_MAX_COUNT="${SESSION_MAX_COUNT:-128}"

LOG_DIR="$REPO_ROOT/logs"
RUN_DIR="$REPO_ROOT/run"
LOG_FILE="$LOG_DIR/proxy.log"
PID_FILE="$RUN_DIR/proxy.pid"
VENV_PATH="$REPO_ROOT/.venv-proxy"
VENV_PYTHON="$VENV_PATH/bin/python"

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

if ! "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
import fastapi, uvicorn, httpx, pydantic
print("ok")
PY
then
  echo "Installing proxy dependencies into $VENV_PATH"
  "$VENV_PYTHON" -m pip install -U pip setuptools wheel
  "$VENV_PYTHON" -m pip install fastapi "uvicorn[standard]" httpx pydantic
fi

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

pkill -f 'uvicorn.*tools.proxy.fish_proxy_pcm:app' 2>/dev/null || true

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export DEFAULT_REFERENCE_ID
export UPSTREAM_TTS_URL
export SESSION_TTL_SEC
export SESSION_MAX_COUNT

nohup "$VENV_PYTHON" -m uvicorn tools.proxy.fish_proxy_pcm:app \
  --host "$PROXY_HOST" \
  --port "$PROXY_PORT" \
  --log-level "$PROXY_LOG_LEVEL" \
  >> "$LOG_FILE" 2>&1 &

PROXY_PID=$!
echo "$PROXY_PID" > "$PID_FILE"

echo "Waiting for proxy to become healthy..."
START_TS="$(date +%s)"

while true; do
  if curl -sf "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
    break
  fi

  if ! kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "ERROR: proxy failed to start. Last log lines:" >&2
    tail -n 100 "$LOG_FILE" >&2 || true
    rm -f "$PID_FILE"
    exit 1
  fi

  NOW_TS="$(date +%s)"
  ELAPSED="$((NOW_TS - START_TS))"

  if [[ "$ELAPSED" -ge "$PROXY_START_TIMEOUT" ]]; then
    echo "ERROR: proxy did not become healthy within ${PROXY_START_TIMEOUT}s" >&2
    tail -n 100 "$LOG_FILE" >&2 || true
    exit 1
  fi

  sleep 1
done

echo "Proxy started"
echo "  pid: $PROXY_PID"
echo "  log: $LOG_FILE"
echo "  health: http://127.0.0.1:${PROXY_PORT}/health"
echo "  open session: curl -X POST http://127.0.0.1:${PROXY_PORT}/session/open -H 'Content-Type: application/json' -d '{\"config_text\":\"{\\\"tts\\\":{\\\"reference_id\\\":\\\"voice\\\"}}\"}'"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

require_cmd "$PYTHON_BIN"
require_cmd curl

PROXY_HOST="${PROXY_HOST:-$(runtime_get 'network.proxy.host')}"
PROXY_PORT="${PROXY_PORT:-$(runtime_get 'network.proxy.port')}"
PROXY_LOG_LEVEL="${PROXY_LOG_LEVEL:-info}"
PROXY_START_TIMEOUT="${PROXY_START_TIMEOUT:-120}"
PROXY_PYTHON="${PROXY_PYTHON:-}"

if [[ -z "$PROXY_PYTHON" ]]; then
  if [[ -x "$REPO_ROOT/.venv-proxy/bin/python" ]]; then
    PROXY_PYTHON="$REPO_ROOT/.venv-proxy/bin/python"
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PROXY_PYTHON="$REPO_ROOT/.venv/bin/python"
  fi
fi

if [[ -z "$PROXY_PYTHON" || ! -x "$PROXY_PYTHON" ]]; then
  echo "ERROR: proxy python not found." >&2
  echo "Expected one of:" >&2
  echo "  $REPO_ROOT/.venv-proxy/bin/python" >&2
  echo "  $REPO_ROOT/.venv/bin/python" >&2
  echo "Create/update it with:" >&2
  echo "  python3.12 -m venv .venv-proxy && .venv-proxy/bin/python -m pip install -U pip && .venv-proxy/bin/python -m pip install -r tools/proxy/requirements.txt" >&2
  exit 1
fi

if ! "$PROXY_PYTHON" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "ERROR: proxy python must be Python >= 3.10" >&2
  echo "Selected python: $PROXY_PYTHON" >&2
  "$PROXY_PYTHON" --version >&2 || true
  echo "Create/update it with:" >&2
  echo "  python3.12 -m venv .venv-proxy && .venv-proxy/bin/python -m pip install -U pip && .venv-proxy/bin/python -m pip install -r tools/proxy/requirements.txt" >&2
  exit 1
fi

if ! "$PROXY_PYTHON" - <<'PY'
import fastapi
import httpx
import uvicorn
PY
then
  echo "ERROR: proxy python is missing required packages: fastapi, uvicorn, httpx" >&2
  echo "Selected python: $PROXY_PYTHON" >&2
  echo "Create/update it with:" >&2
  echo "  $PROXY_PYTHON -m pip install -U pip && $PROXY_PYTHON -m pip install -r tools/proxy/requirements.txt" >&2
  exit 1
fi

LOG_DIR="$REPO_ROOT/logs"
RUN_DIR="$REPO_ROOT/run"
LOG_FILE="$(proxy_log_file)"
PID_FILE="$(proxy_pid_file)"

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

pkill -f 'uvicorn.*tools.tts_server.proxy.pcm:app' 2>/dev/null || true
pkill -f 'uvicorn.*tools.proxy.fish_proxy_pcm:app' 2>/dev/null || true

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

nohup "$PROXY_PYTHON" -m uvicorn tools.tts_server.proxy.pcm:app \
  --host "$PROXY_HOST" \
  --port "$PROXY_PORT" \
  --log-level "$PROXY_LOG_LEVEL" \
  >> "$LOG_FILE" 2>&1 &

PROXY_PID=$!
echo "$PROXY_PID" > "$PID_FILE"

echo "Waiting for proxy to become healthy..."
if ! wait_http_ok "http://127.0.0.1:${PROXY_PORT}/health" "$PROXY_START_TIMEOUT" 1; then
  echo "ERROR: proxy failed to become healthy" >&2
  tail -n 200 "$LOG_FILE" >&2 || true
  rm -f "$PID_FILE"
  exit 1
fi

echo "Proxy started"
echo "  pid: $PROXY_PID"
echo "  python: $PROXY_PYTHON"
echo "  log: $LOG_FILE"
echo "  health: http://127.0.0.1:${PROXY_PORT}/health"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PROXY_PORT="${PROXY_PORT:-9000}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
PROXY_LOG_LEVEL="${PROXY_LOG_LEVEL:-info}"
PROXY_APP="${PROXY_APP:-tools.proxy.fish_proxy_pcm:app}"

MODEL_PORT="${MODEL_PORT:-8080}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://127.0.0.1:${MODEL_PORT}}"
WAIT_FOR_MODEL="${WAIT_FOR_MODEL:-1}"

PROXY_START_TIMEOUT="${PROXY_START_TIMEOUT:-120}"
MODEL_HEALTH_TIMEOUT="${MODEL_HEALTH_TIMEOUT:-1800}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="$REPO_ROOT/logs"
RUN_DIR="$REPO_ROOT/run"
LOG_FILE="${PROXY_LOG_FILE:-$LOG_DIR/proxy.log}"
PID_FILE="$RUN_DIR/proxy.pid"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv-proxy}"

mkdir -p "$LOG_DIR" "$RUN_DIR"
touch "$LOG_FILE"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: python not found: $PYTHON_BIN" >&2
  exit 1
fi

wait_for_url() {
  local url="$1"
  local label="$2"
  local timeout="$3"

  local started
  started="$(date +%s)"

  while true; do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi

    local now elapsed
    now="$(date +%s)"
    elapsed="$((now - started))"

    if (( elapsed >= timeout )); then
      echo "ERROR: ${label} is not ready within ${timeout}s: ${url}" >&2
      return 1
    fi

    sleep 2
  done
}

if [[ "$WAIT_FOR_MODEL" == "1" ]]; then
  echo "Waiting for model backend: ${MODEL_BASE_URL}/v1/health"
  wait_for_url "${MODEL_BASE_URL}/v1/health" "model backend" "$MODEL_HEALTH_TIMEOUT"
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

pkill -f "$PROXY_APP" 2>/dev/null || true

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

nohup python -m uvicorn "$PROXY_APP" \
  --host "$PROXY_HOST" \
  --port "$PROXY_PORT" \
  --log-level "$PROXY_LOG_LEVEL" \
  >> "$LOG_FILE" 2>&1 &

NEW_PID="$!"
echo "$NEW_PID" > "$PID_FILE"

if ! wait_for_url "http://127.0.0.1:${PROXY_PORT}/health" "proxy" "$PROXY_START_TIMEOUT"; then
  echo "ERROR: proxy failed to start. Last log lines:" >&2
  tail -n 100 "$LOG_FILE" 2>/dev/null || true
  exit 1
fi

echo "Proxy started"
echo "  pid: $NEW_PID"
echo "  log: $LOG_FILE"
echo "  health: http://127.0.0.1:${PROXY_PORT}/health"
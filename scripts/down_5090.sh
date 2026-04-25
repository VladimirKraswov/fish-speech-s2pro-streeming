#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

CONTAINER="${CONTAINER:-fish-speech-5090}"
PID_FILE="$(proxy_pid_file)"
WEBUI_PID_FILE="$(webui_pid_file)"

echo "Stopping Web UI"
if [[ -f "$WEBUI_PID_FILE" ]]; then
  PID="$(cat "$WEBUI_PID_FILE" 2>/dev/null || true)"
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
      kill -9 "$PID" 2>/dev/null || true
    fi
  fi
  rm -f "$WEBUI_PID_FILE"
fi

pkill -f 'python3 -m fish_speech_web_ui.server' 2>/dev/null || true

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

pkill -f 'uvicorn.*fish_speech_server.proxy.pcm:app' 2>/dev/null || true

echo "[2/2] Stopping model container"
docker_cmd rm -f "$CONTAINER" 2>/dev/null || true

echo "All stopped"

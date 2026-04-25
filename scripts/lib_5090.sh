#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$REPO_ROOT/config/runtime.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

docker_cmd() {
  if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
    sudo docker "$@"
  else
    docker "$@"
  fi
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

require_file() {
  [[ -f "$1" ]] || {
    echo "ERROR: file not found: $1" >&2
    exit 1
  }
}

require_dir() {
  [[ -d "$1" ]] || {
    echo "ERROR: directory not found: $1" >&2
    exit 1
  }
}

runtime_get() {
  local dotted_path="$1"
  "$PYTHON_BIN" - "$RUNTIME_CONFIG" "$dotted_path" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
dotted = sys.argv[2]

with open(cfg_path, "r", encoding="utf-8") as f:
    obj = json.load(f)

cur = obj
for part in dotted.split("."):
    cur = cur[part]

if isinstance(cur, bool):
    print("true" if cur else "false")
elif cur is None:
    print("")
else:
    print(cur)
PY
}

runtime_path() {
  local dotted_path="$1"
  local value
  value="$(runtime_get "$dotted_path")"
  if [[ -z "$value" ]]; then
    echo ""
    return 0
  fi
  if [[ "$value" = /* ]]; then
    echo "$value"
  else
    echo "$REPO_ROOT/$value"
  fi
}

proxy_pid_file() {
  echo "$REPO_ROOT/run/proxy.pid"
}

proxy_log_file() {
  echo "${PROXY_LOG_FILE:-$REPO_ROOT/logs/proxy.log}"
}

webui_pid_file() {
  echo "$REPO_ROOT/run/webui.pid"
}

webui_log_file() {
  echo "${WEBUI_LOG_FILE:-$REPO_ROOT/logs/webui.log}"
}

wait_http_ok() {
  local url="$1"
  local timeout="${2:-300}"
  local step="${3:-2}"

  local started
  started="$(date +%s)"

  while true; do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi

    local now elapsed
    now="$(date +%s)"
    elapsed="$((now - started))"

    if [[ "$elapsed" -ge "$timeout" ]]; then
      echo "ERROR: timeout waiting for $url (${timeout}s)" >&2
      return 1
    fi

    sleep "$step"
  done
}
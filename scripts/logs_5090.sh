#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

CONTAINER="${CONTAINER:-fish-speech-5090}"
PROXY_LOG_FILE="$(proxy_log_file)"

mkdir -p "$REPO_ROOT/logs"
touch "$PROXY_LOG_FILE"

cleanup() {
  [[ -n "${MODEL_PID:-}" ]] && kill "$MODEL_PID" 2>/dev/null || true
  [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Live logs ==="
echo "model container: $CONTAINER"
echo "proxy log: $PROXY_LOG_FILE"
echo

(
  docker_cmd logs -f "$CONTAINER" 2>&1 | sed 's/^/[model] /'
) &
MODEL_PID=$!

(
  tail -n 50 -F "$PROXY_LOG_FILE" 2>/dev/null | sed 's/^/[proxy] /'
) &
PROXY_PID=$!

wait
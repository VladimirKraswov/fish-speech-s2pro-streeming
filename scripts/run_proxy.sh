#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# По умолчанию запускаем PCM/NDJSON proxy для минимального TTFA в браузере.
PROXY_APP="${PROXY_APP:-tools.proxy.fish_proxy_pcm:app}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install uv or run inside the project environment." >&2
  exit 1
fi

echo "=== Fish Speech proxy ==="
echo "  app=$PROXY_APP"
echo "  host=$HOST"
echo "  port=$PORT"
echo "  log_level=$LOG_LEVEL"
echo

exec uv run uvicorn "$PROXY_APP" \
  --host "$HOST" \
  --port "$PORT" \
  --log-level "$LOG_LEVEL"
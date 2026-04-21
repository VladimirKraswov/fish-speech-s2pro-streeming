#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8080}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
WARMUP_TIMEOUT="${WARMUP_TIMEOUT:-1800}"
WARMUP_TEXT="${WARMUP_TEXT:-Привет. Это дополнительный прогрев стримингового режима для Fish Speech.}"
OUT_FILE="${OUT_FILE:-$REPO_ROOT/logs/warmup_stream.wav}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$REPO_ROOT/logs"

wait_for_health() {
  local started
  started="$(date +%s)"

  while true; do
    if curl -sf "$BASE_URL/v1/health" >/dev/null 2>&1; then
      return 0
    fi

    local now elapsed
    now="$(date +%s)"
    elapsed="$((now - started))"

    echo "Still warming up... ${elapsed}s"

    if (( elapsed >= WARMUP_TIMEOUT )); then
      echo "ERROR: model did not become healthy within ${WARMUP_TIMEOUT}s" >&2
      return 1
    fi

    sleep 10
  done
}

echo "Waiting for model health: $BASE_URL/v1/health"
wait_for_health
echo "Model is healthy"
echo "Sending one extra warmup streaming request..."

REQUEST_JSON="$(
WARMUP_TEXT="$WARMUP_TEXT" "$PYTHON_BIN" - <<'PY'
import json
import os

print(json.dumps({
    "text": os.environ["WARMUP_TEXT"],
    "streaming": True,
}, ensure_ascii=False))
PY
)"

curl -sf -X POST "$BASE_URL/v1/tts" \
  -H 'Content-Type: application/json' \
  -d "$REQUEST_JSON" \
  --output "$OUT_FILE" \
  --max-time 180 >/dev/null

if [[ ! -s "$OUT_FILE" ]]; then
  echo "ERROR: warmup output is empty: $OUT_FILE" >&2
  exit 1
fi

echo "Warmup completed"
echo "Output: $OUT_FILE"
echo "Current memory:"
curl -sf "$BASE_URL/v1/debug/memory" || true
echo
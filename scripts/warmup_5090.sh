# scripts/warmup_5090.sh
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8080}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
WARMUP_TIMEOUT="${WARMUP_TIMEOUT:-1800}"
WARMUP_TEXT="${WARMUP_TEXT:-Привет. Это дополнительный прогрев стримингового режима для Fish Speech.}"
OUT_FILE="${OUT_FILE:-$REPO_ROOT/logs/warmup_stream.wav}"

mkdir -p "$REPO_ROOT/logs"

echo "Waiting for model health: $BASE_URL/v1/health"

START_TS="$(date +%s)"
while true; do
  if curl -sf "$BASE_URL/v1/health" >/dev/null 2>&1; then
    break
  fi

  NOW_TS="$(date +%s)"
  ELAPSED="$((NOW_TS - START_TS))"
  echo "Still warming up... ${ELAPSED}s"

  if [[ "$ELAPSED" -ge "$WARMUP_TIMEOUT" ]]; then
    echo "ERROR: model did not become healthy within ${WARMUP_TIMEOUT}s" >&2
    exit 1
  fi

  sleep 10
done

echo "Model is healthy after $(( $(date +%s) - START_TS ))s"
echo "Sending one extra warmup streaming request..."

curl -sf -X POST "$BASE_URL/v1/tts" \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"${WARMUP_TEXT}\",\"streaming\":true}" \
  --output "$OUT_FILE" \
  --max-time 180 >/dev/null

echo "Warmup completed"
echo "Current memory:"
curl -sf "$BASE_URL/v1/debug/memory" || true
echo
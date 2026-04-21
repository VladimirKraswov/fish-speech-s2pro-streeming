 
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:9000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
OUT_DIR="${OUT_DIR:-${TMPDIR:-/tmp}/fish-speech-warmup}"

TEXT_ONESHOT="${TEXT_ONESHOT:-Привет. Это короткий прогревочный запрос.}"
TEXT_STREAM="${TEXT_STREAM:-Сегодня утром я решил пройтись по улице и проверить, как быстро начинается потоковое воспроизведение.}"
REFERENCE_ID="${REFERENCE_ID:-}"
WARM_PROXY="${WARM_PROXY:-1}"

mkdir -p "$OUT_DIR"

wait_http() {
  local url="$1"
  local timeout="$2"
  local waited=0

  until curl -sf --connect-timeout 2 "$url" >/dev/null 2>&1; do
    sleep 2
    waited=$((waited + 2))
    if [[ "$waited" -ge "$timeout" ]]; then
      echo "ERROR: timeout waiting for $url" >&2
      exit 1
    fi
  done
}

echo "=== Warmup 5090 ==="
echo "  BASE_URL=$BASE_URL"
echo "  PROXY_URL=$PROXY_URL"
echo "  OUT_DIR=$OUT_DIR"
echo "  REFERENCE_ID=${REFERENCE_ID:-<empty>}"
echo

echo "[1/4] Waiting for API health..."
wait_http "$BASE_URL/v1/health" "$HEALTH_TIMEOUT"
echo "API is ready."

if [[ "$WARM_PROXY" == "1" ]]; then
  echo "[2/4] Waiting for proxy health..."
  wait_http "$PROXY_URL/health" 120
  echo "Proxy is ready."
else
  echo "[2/4] Proxy warmup disabled."
fi

build_json() {
  local text="$1"
  local streaming="$2"

  if [[ -n "$REFERENCE_ID" ]]; then
    printf '{"text":"%s","streaming":%s,"reference_id":"%s"}' \
      "$(printf '%s' "$text" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read())[1:-1])')" \
      "$streaming" \
      "$REFERENCE_ID"
  else
    printf '{"text":"%s","streaming":%s}' \
      "$(printf '%s' "$text" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read())[1:-1])')" \
      "$streaming"
  fi
}

echo "[3/4] Warmup oneshot TTS..."
curl -sf -X POST "$BASE_URL/v1/tts" \
  -H "Content-Type: application/json" \
  -d "$(build_json "$TEXT_ONESHOT" false)" \
  --output "$OUT_DIR/warmup_oneshot.wav" \
  --max-time 120

echo "Saved: $OUT_DIR/warmup_oneshot.wav"

echo "[4/4] Warmup streaming TTS..."
curl -sf -X POST "$BASE_URL/v1/tts" \
  -H "Content-Type: application/json" \
  -d "$(build_json "$TEXT_STREAM" true)" \
  --output "$OUT_DIR/warmup_stream.wav" \
  --max-time 120

echo "Saved: $OUT_DIR/warmup_stream.wav"

if [[ "$WARM_PROXY" == "1" ]]; then
  echo
  echo "Extra warmup: proxy PCM stream..."
  if [[ -n "$REFERENCE_ID" ]]; then
    curl -sfG "$PROXY_URL/pcm-stream" \
      --data-urlencode "text=$TEXT_STREAM" \
      --data-urlencode "reference_id=$REFERENCE_ID" \
      --output "$OUT_DIR/warmup_proxy.ndjson" \
      --max-time 120
  else
    curl -sfG "$PROXY_URL/pcm-stream" \
      --data-urlencode "text=$TEXT_STREAM" \
      --output "$OUT_DIR/warmup_proxy.ndjson" \
      --max-time 120
  fi
  echo "Saved: $OUT_DIR/warmup_proxy.ndjson"
fi

echo
echo "Warmup done."
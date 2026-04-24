#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

require_cmd "$PYTHON_BIN"
require_cmd curl

PORT="${PORT:-$(runtime_get 'network.server.port')}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
WARMUP_TIMEOUT="${WARMUP_TIMEOUT:-1800}"
OUT_FILE="${OUT_FILE:-$REPO_ROOT/logs/warmup_stream.wav}"

WARMUP_TEXT="${WARMUP_TEXT:-$(runtime_get 'warmup.text')}"
WARMUP_REFERENCE_ID="${WARMUP_REFERENCE_ID:-$(runtime_get 'warmup.reference_id')}"
WARMUP_MAX_NEW_TOKENS="${WARMUP_MAX_NEW_TOKENS:-$(runtime_get 'warmup.max_new_tokens')}"
WARMUP_CHUNK_LENGTH="${WARMUP_CHUNK_LENGTH:-$(runtime_get 'warmup.chunk_length')}"
WARMUP_INITIAL_STREAM_CHUNK_SIZE="${WARMUP_INITIAL_STREAM_CHUNK_SIZE:-$(runtime_get 'warmup.initial_stream_chunk_size')}"
WARMUP_STREAM_CHUNK_SIZE="${WARMUP_STREAM_CHUNK_SIZE:-$(runtime_get 'warmup.stream_chunk_size')}"

TOP_P="${TOP_P:-$(runtime_get 'proxy.tts.top_p')}"
REPETITION_PENALTY="${REPETITION_PENALTY:-$(runtime_get 'proxy.tts.repetition_penalty')}"
TEMPERATURE="${TEMPERATURE:-$(runtime_get 'proxy.tts.temperature')}"
USE_MEMORY_CACHE="${USE_MEMORY_CACHE:-$(runtime_get 'proxy.tts.use_memory_cache')}"

if [[ -z "$WARMUP_REFERENCE_ID" ]]; then
  WARMUP_REFERENCE_ID="$(runtime_get 'proxy.default_reference_id')"
fi

mkdir -p "$REPO_ROOT/logs"

echo "Waiting for model health: $BASE_URL/v1/health"
if ! wait_http_ok "$BASE_URL/v1/health" "$WARMUP_TIMEOUT" 5; then
  echo "ERROR: model did not become healthy within ${WARMUP_TIMEOUT}s" >&2
  exit 1
fi

BODY="$(
  WARMUP_TEXT="$WARMUP_TEXT" \
  WARMUP_REFERENCE_ID="$WARMUP_REFERENCE_ID" \
  WARMUP_MAX_NEW_TOKENS="$WARMUP_MAX_NEW_TOKENS" \
  WARMUP_CHUNK_LENGTH="$WARMUP_CHUNK_LENGTH" \
  WARMUP_INITIAL_STREAM_CHUNK_SIZE="$WARMUP_INITIAL_STREAM_CHUNK_SIZE" \
  WARMUP_STREAM_CHUNK_SIZE="$WARMUP_STREAM_CHUNK_SIZE" \
  TOP_P="$TOP_P" \
  REPETITION_PENALTY="$REPETITION_PENALTY" \
  TEMPERATURE="$TEMPERATURE" \
  USE_MEMORY_CACHE="$USE_MEMORY_CACHE" \
  "$PYTHON_BIN" - <<'PY'
import json
import os

payload = {
    "text": os.environ["WARMUP_TEXT"],
    "streaming": True,
    "stream_tokens": True,
    "reference_id": os.environ["WARMUP_REFERENCE_ID"],
    "format": "wav",
    "normalize": True,
    "use_memory_cache": os.environ["USE_MEMORY_CACHE"],
    "max_new_tokens": int(os.environ["WARMUP_MAX_NEW_TOKENS"]),
    "chunk_length": int(os.environ["WARMUP_CHUNK_LENGTH"]),
    "top_p": float(os.environ["TOP_P"]),
    "repetition_penalty": float(os.environ["REPETITION_PENALTY"]),
    "temperature": float(os.environ["TEMPERATURE"]),
    "initial_stream_chunk_size": int(os.environ["WARMUP_INITIAL_STREAM_CHUNK_SIZE"]),
    "stream_chunk_size": int(os.environ["WARMUP_STREAM_CHUNK_SIZE"]),
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"

echo "Sending extra warmup streaming request..."
echo "Using warmup reference: $WARMUP_REFERENCE_ID"

curl -sf -X POST "$BASE_URL/v1/tts" \
  -H 'Content-Type: application/json' \
  -d "$BODY" \
  --output "$OUT_FILE" \
  --max-time 240 >/dev/null

echo "Warmup completed"
echo "Current memory:"
curl -sf "$BASE_URL/v1/debug/memory" || true
echo
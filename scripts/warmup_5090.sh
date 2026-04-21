#!/usr/bin/env bash
set -euo pipefail

MODEL_PORT="${MODEL_PORT:-8080}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${MODEL_PORT}}"
TIMEOUT="${WARMUP_TIMEOUT:-1800}"
INTERVAL="${WARMUP_INTERVAL:-5}"

elapsed=0

echo "Waiting for model warmup on ${BASE_URL} ..."
until curl -sf "${BASE_URL}/v1/health" >/dev/null 2>&1; do
  if (( elapsed % 30 == 0 )); then
    echo "Still warming up... ${elapsed}s"
    sudo docker logs --tail 30 fish-speech 2>&1 || true
    echo
  fi

  if (( elapsed >= TIMEOUT )); then
    echo "ERROR: warmup timeout after ${TIMEOUT}s" >&2
    exit 1
  fi

  sleep "$INTERVAL"
  elapsed=$((elapsed + INTERVAL))
done

echo "Model is healthy after ${elapsed}s"

echo "Sending one extra warmup streaming request..."
curl -sf -X POST "${BASE_URL}/v1/tts" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Warmup request after startup.","streaming":true}' \
  --output /dev/null || true

echo "Warmup completed"

echo "Current memory:"
curl -sf "${BASE_URL}/v1/debug/memory" || true
echo
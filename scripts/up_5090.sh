#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

require_cmd "$PYTHON_BIN"
require_cmd curl
require_cmd docker

IMAGE="${IMAGE:-fish-speech-server:cu129}"
CONTAINER="${CONTAINER:-fish-speech-5090}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
START_PROXY="${START_PROXY:-1}"
EXTRA_WARMUP="${EXTRA_WARMUP:-1}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"

CUDA_VER="${CUDA_VER:-12.9.0}"
UV_EXTRA="${UV_EXTRA:-cu129}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SERVER_HOST="${SERVER_HOST:-$(runtime_get 'network.server.host')}"
SERVER_PORT="${SERVER_PORT:-$(runtime_get 'network.server.port')}"
PROXY_PORT="${PROXY_PORT:-$(runtime_get 'network.proxy.port')}"

LLAMA_CHECKPOINT_PATH="${LLAMA_CHECKPOINT_PATH:-$(runtime_path 'paths.llama_checkpoint_path')}"
DECODER_CHECKPOINT_PATH="${DECODER_CHECKPOINT_PATH:-$(runtime_path 'paths.decoder_checkpoint_path')}"
DEFAULT_REFERENCE_ID="${DEFAULT_REFERENCE_ID:-$(runtime_get 'proxy.default_reference_id')}"
WARMUP_REFERENCE_ID="${WARMUP_REFERENCE_ID:-$(runtime_get 'warmup.reference_id')}"

if [[ -z "$WARMUP_REFERENCE_ID" ]]; then
  WARMUP_REFERENCE_ID="$DEFAULT_REFERENCE_ID"
fi

require_file "$RUNTIME_CONFIG"
require_dir "$LLAMA_CHECKPOINT_PATH"
require_file "$DECODER_CHECKPOINT_PATH"

echo "=== Fish Speech startup (RTX 5090) ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "RUNTIME_CONFIG=$RUNTIME_CONFIG"
echo "IMAGE=$IMAGE"
echo "CONTAINER=$CONTAINER"
echo "SERVER=${SERVER_HOST}:${SERVER_PORT}"
echo "PROXY_PORT=$PROXY_PORT"
echo "LLAMA_CHECKPOINT_PATH=$LLAMA_CHECKPOINT_PATH"
echo "DECODER_CHECKPOINT_PATH=$DECODER_CHECKPOINT_PATH"
echo "DEFAULT_REFERENCE_ID=$DEFAULT_REFERENCE_ID"
echo "WARMUP_REFERENCE_ID=$WARMUP_REFERENCE_ID"
echo

if [[ "$BUILD_IMAGE" == "1" ]] || ! docker_cmd image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[1/5] Building Docker image..."
  docker_cmd build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER="$CUDA_VER" \
    --build-arg UV_EXTRA="$UV_EXTRA" \
    --target server \
    -t "$IMAGE" .
else
  echo "[1/5] Using existing image: $IMAGE"
fi

echo "[2/5] Stopping previous processes..."
CONTAINER="$CONTAINER" DOCKER_USE_SUDO="${DOCKER_USE_SUDO:-0}" bash "$REPO_ROOT/scripts/down_5090.sh"

echo "[3/5] Starting model container..."
CID="$(
  docker_cmd run -d --rm \
    --name "$CONTAINER" \
    -p "${SERVER_PORT}:${SERVER_PORT}" \
    --gpus all \
    -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    -e PYTHONPATH=/workspace \
    -v "$REPO_ROOT":/workspace \
    -w /workspace \
    --entrypoint /app/.venv/bin/python \
    "$IMAGE" \
    -m fish_speech_server.app \
    --listen "0.0.0.0:${SERVER_PORT}"
)"

echo "Container started: $CID"
echo

echo "[4/5] Waiting for model health..."
if ! wait_http_ok "http://127.0.0.1:${SERVER_PORT}/v1/health" "$HEALTH_TIMEOUT" 5; then
  echo "ERROR: model did not become healthy" >&2
  docker_cmd logs --tail 200 "$CONTAINER" || true
  exit 1
fi

echo "Model is healthy"
echo
echo "Current model memory:"
curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/debug/memory" || true
echo
echo

if [[ "$EXTRA_WARMUP" == "1" ]]; then
  echo "Running extra warmup request..."
  PORT="$SERVER_PORT" \
  BASE_URL="http://127.0.0.1:${SERVER_PORT}" \
  WARMUP_REFERENCE_ID="$WARMUP_REFERENCE_ID" \
  bash "$REPO_ROOT/scripts/warmup_5090.sh"
  echo
fi

if [[ "$START_PROXY" == "1" ]]; then
  echo "[5/5] Starting proxy..."
  PROXY_PORT="$PROXY_PORT" \
  bash "$REPO_ROOT/scripts/run_proxy.sh"
else
  echo "[5/5] Proxy start skipped"
fi

echo
echo "=== READY ==="
echo "Model health:       http://127.0.0.1:${SERVER_PORT}/v1/health"
echo "Model memory:       http://127.0.0.1:${SERVER_PORT}/v1/debug/memory"
echo "Proxy health:       http://127.0.0.1:${PROXY_PORT}/health"
echo "Proxy open session: http://127.0.0.1:${PROXY_PORT}/session/open"
echo "Logs:               bash scripts/logs_5090.sh"
echo "Status:             bash scripts/status_5090.sh"

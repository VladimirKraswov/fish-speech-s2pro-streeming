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
BUILD_IMAGE="${BUILD_IMAGE:-0}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"

CUDA_VER="${CUDA_VER:-12.9.0}"
UV_EXTRA="${UV_EXTRA:-cu129}"

SERVER_PORT="${SERVER_PORT:-$(runtime_get 'network.server.port')}"
PROXY_PORT="${PROXY_PORT:-$(runtime_get 'network.proxy.port')}"
WEBUI_PORT=9001

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
echo "SERVER_PORT=$SERVER_PORT"
echo "PROXY_PORT=$PROXY_PORT"
echo "WEBUI_PORT=$WEBUI_PORT"
echo

if [[ "$BUILD_IMAGE" == "1" ]]; then
  echo "[1/4] Building full-stack Docker images..."
  docker_compose_cmd build
else
  echo "[1/4] Using existing images (skip build)"
fi

echo "[2/4] Stopping previous processes..."
bash "$REPO_ROOT/scripts/down_5090.sh"

echo "[3/4] Starting services (full-stack)..."
docker_compose_cmd --profile full-stack up -d

echo "[4/4] Waiting for services health..."
if ! wait_http_ok "http://127.0.0.1:${SERVER_PORT}/v1/health" "$HEALTH_TIMEOUT" 5; then
  echo "ERROR: model did not become healthy" >&2
  docker_compose_cmd logs server --tail 200
  exit 1
fi

if ! wait_http_ok "http://127.0.0.1:${PROXY_PORT}/health" 60 2; then
  echo "ERROR: proxy did not become healthy" >&2
  docker_compose_cmd logs proxy --tail 200
  exit 1
fi

if ! wait_http_ok "http://127.0.0.1:${WEBUI_PORT}/health" 30 1; then
  echo "ERROR: web-ui did not become healthy" >&2
  docker_compose_cmd logs web-ui --tail 200
  exit 1
fi

echo "All services are healthy"
echo

echo "Current model memory:"
curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/debug/memory" || true
echo
echo

if [[ "${EXTRA_WARMUP:-1}" == "1" ]]; then
  echo "Running extra warmup request..."
  PORT="$SERVER_PORT" \
  BASE_URL="http://127.0.0.1:${SERVER_PORT}" \
  WARMUP_REFERENCE_ID="$WARMUP_REFERENCE_ID" \
  bash "$REPO_ROOT/scripts/warmup_5090.sh"
  echo
fi

echo
echo "=== READY ==="
echo "Model API:   http://127.0.0.1:${SERVER_PORT}/v1/health"
echo "Proxy API:   http://127.0.0.1:${PROXY_PORT}/health"
echo "Web UI:      http://127.0.0.1:${WEBUI_PORT}"
echo "Logs:        docker compose logs -f"
echo "Status:      bash scripts/status_5090.sh"

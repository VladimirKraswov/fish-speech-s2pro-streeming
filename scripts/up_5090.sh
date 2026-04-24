#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-fish-speech-webui:cu129}"
CONTAINER="${CONTAINER:-fish-speech}"
PORT="${PORT:-8080}"
PROXY_PORT="${PROXY_PORT:-9000}"

COMPILE="${COMPILE:-1}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
START_PROXY="${START_PROXY:-1}"
EXTRA_WARMUP="${EXTRA_WARMUP:-1}"

CUDA_VER="${CUDA_VER:-12.9.0}"
UV_EXTRA="${UV_EXTRA:-cu129}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/s2-pro}"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
FISH_CACHE_MAX_SEQ_LEN="${FISH_CACHE_MAX_SEQ_LEN:-512}"
FISH_MAX_NEW_TOKENS_CAP="${FISH_MAX_NEW_TOKENS_CAP:-64}"
DEFAULT_REFERENCE_ID="${DEFAULT_REFERENCE_ID:-voice}"
SESSION_TTL_SEC="${SESSION_TTL_SEC:-1800}"
SESSION_MAX_COUNT="${SESSION_MAX_COUNT:-128}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

docker_cmd() {
  if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
    sudo docker "$@"
  else
    docker "$@"
  fi
}

echo "=== Fish Speech full startup (RTX 5090) ==="
echo "  REPO_ROOT=$REPO_ROOT"
echo "  IMAGE=$IMAGE"
echo "  CONTAINER=$CONTAINER"
echo "  PORT=$PORT"
echo "  PROXY_PORT=$PROXY_PORT"
echo "  DEFAULT_REFERENCE_ID=$DEFAULT_REFERENCE_ID"
echo "  SESSION_TTL_SEC=$SESSION_TTL_SEC"
echo

if [[ ! -d "$CHECKPOINTS_DIR" ]] || [[ -z "$(ls -A "$CHECKPOINTS_DIR" 2>/dev/null)" ]]; then
  echo "ERROR: checkpoints not found in $CHECKPOINTS_DIR" >&2
  exit 1
fi

if [[ "$BUILD_IMAGE" == "1" ]] || ! docker_cmd image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[1/5] Building Docker image..."
  docker_cmd build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER="$CUDA_VER" \
    --build-arg UV_EXTRA="$UV_EXTRA" \
    --target webui \
    -t "$IMAGE" .
else
  echo "[1/5] Using existing image: $IMAGE"
fi

echo "[2/5] Stopping previous processes..."
DOCKER_USE_SUDO="${DOCKER_USE_SUDO:-0}" bash "$REPO_ROOT/scripts/down_5090.sh"

echo "[3/5] Starting model container..."
CID="$({
  docker_cmd run -d --rm \
    --name "$CONTAINER" \
    -p "$PORT:8080" \
    --gpus all \
    -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    -e FISH_CACHE_MAX_SEQ_LEN="$FISH_CACHE_MAX_SEQ_LEN" \
    -e FISH_MAX_NEW_TOKENS_CAP="$FISH_MAX_NEW_TOKENS_CAP" \
    -e PYTHONPATH=/workspace \
    -v "$REPO_ROOT":/workspace \
    -w /workspace \
    --entrypoint /app/.venv/bin/python \
    "$IMAGE" \
    /workspace/tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --device cuda \
    --llama-checkpoint-path "/workspace/$CHECKPOINTS_DIR" \
    --decoder-checkpoint-path "/workspace/$CHECKPOINTS_DIR/codec.pth" \
    $([[ "$COMPILE" == "1" ]] && echo --compile || true)
)}"

echo "Container started: $CID"
echo

echo "[4/5] Waiting for model health..."
START_TS="$(date +%s)"
while true; do
  if curl -sf "http://127.0.0.1:${PORT}/v1/health" >/dev/null 2>&1; then
    break
  fi

  if ! docker_cmd ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: container exited during startup" >&2
    docker_cmd logs --tail 200 "$CONTAINER" 2>&1 || true
    exit 1
  fi

  NOW_TS="$(date +%s)"
  ELAPSED="$((NOW_TS - START_TS))"
  if (( ELAPSED % 30 == 0 )); then
    echo "Still warming up... ${ELAPSED}s"
  fi

  if [[ "$ELAPSED" -ge "$HEALTH_TIMEOUT" ]]; then
    echo "ERROR: model did not become healthy within ${HEALTH_TIMEOUT}s" >&2
    docker_cmd logs --tail 200 "$CONTAINER" 2>&1 || true
    exit 1
  fi

  sleep 5
done

echo "Model is healthy after $(( $(date +%s) - START_TS ))s"

if [[ "$EXTRA_WARMUP" == "1" ]]; then
  echo "Sending one extra warmup streaming request..."
  BASE_URL="http://127.0.0.1:${PORT}" \
  WARMUP_REFERENCE_ID="$DEFAULT_REFERENCE_ID" \
  bash "$REPO_ROOT/scripts/warmup_5090.sh"
fi

echo "Current model memory:"
curl -sf "http://127.0.0.1:${PORT}/v1/debug/memory" || true
echo
echo

if [[ "$START_PROXY" == "1" ]]; then
  echo "[5/5] Starting proxy..."
  PROXY_PORT="$PROXY_PORT" \
  DEFAULT_REFERENCE_ID="$DEFAULT_REFERENCE_ID" \
  SESSION_TTL_SEC="$SESSION_TTL_SEC" \
  SESSION_MAX_COUNT="$SESSION_MAX_COUNT" \
  bash "$REPO_ROOT/scripts/run_proxy.sh"
fi

echo
echo "=== READY ==="
echo "Model health:      http://127.0.0.1:${PORT}/v1/health"
echo "Model memory:      http://127.0.0.1:${PORT}/v1/debug/memory"
echo "Proxy health:      http://127.0.0.1:${PROXY_PORT}/health"
echo "Proxy open session:http://127.0.0.1:${PROXY_PORT}/session/open"
echo "Proxy stream:      http://127.0.0.1:${PROXY_PORT}/pcm-stream?session_id=...&text=Привет"

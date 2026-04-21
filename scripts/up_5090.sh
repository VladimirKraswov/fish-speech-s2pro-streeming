#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE="${IMAGE:-fish-speech-webui:cu129}"
CONTAINER="${CONTAINER:-fish-speech}"

PORT="${PORT:-8080}"
PROXY_PORT="${PROXY_PORT:-9000}"
SESSION_PORT="${SESSION_PORT:-8765}"

PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
SESSION_HOST="${SESSION_HOST:-0.0.0.0}"

MODE="${MODE:-full}"                 # full | streaming | session | model
RESTART_MODEL="${RESTART_MODEL:-0}" # 1 -> force restart/recompile model
BUILD_IMAGE="${BUILD_IMAGE:-0}"
COMPILE="${COMPILE:-1}"
EXTRA_WARMUP="${EXTRA_WARMUP:-1}"

CUDA_VER="${CUDA_VER:-12.9.0}"
UV_EXTRA="${UV_EXTRA:-cu129}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/s2-pro}"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
FISH_CACHE_MAX_SEQ_LEN="${FISH_CACHE_MAX_SEQ_LEN:-320}"
FISH_MAX_NEW_TOKENS_CAP="${FISH_MAX_NEW_TOKENS_CAP:-64}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"

MODEL_BASE_URL="http://127.0.0.1:${PORT}"
PROXY_BASE_URL="http://127.0.0.1:${PROXY_PORT}"
SESSION_BASE_URL="http://127.0.0.1:${SESSION_PORT}"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

docker_cmd() {
  if [[ "${DOCKER_USE_SUDO:-0}" == "1" ]]; then
    sudo docker "$@"
  else
    docker "$@"
  fi
}

is_true() {
  [[ "${1:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]
}

wait_for_url() {
  local url="$1"
  local label="$2"
  local timeout="$3"

  local started
  started="$(date +%s)"

  while true; do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi

    local now elapsed
    now="$(date +%s)"
    elapsed="$((now - started))"

    if (( elapsed > 0 && elapsed % 30 == 0 )); then
      echo "${label}: still waiting... ${elapsed}s"
    fi

    if (( elapsed >= timeout )); then
      echo "ERROR: ${label} not ready within ${timeout}s: ${url}" >&2
      return 1
    fi

    sleep 5
  done
}

container_running() {
  docker_cmd ps --format '{{.Names}}' | grep -qx "$CONTAINER"
}

select_mode() {
  case "$MODE" in
    full)
      DEFAULT_START_PROXY=1
      DEFAULT_START_SESSION=1
      ;;
    streaming|stream)
      DEFAULT_START_PROXY=1
      DEFAULT_START_SESSION=0
      ;;
    session)
      DEFAULT_START_PROXY=0
      DEFAULT_START_SESSION=1
      ;;
    model)
      DEFAULT_START_PROXY=0
      DEFAULT_START_SESSION=0
      ;;
    *)
      echo "ERROR: unsupported MODE=$MODE (use: full | streaming | session | model)" >&2
      exit 1
      ;;
  esac

  START_PROXY="${START_PROXY:-$DEFAULT_START_PROXY}"
  START_SESSION="${START_SESSION:-$DEFAULT_START_SESSION}"
}

build_image_if_needed() {
  if is_true "$BUILD_IMAGE" || ! docker_cmd image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building Docker image: $IMAGE"
    docker_cmd build \
      --platform linux/amd64 \
      -f docker/Dockerfile \
      --build-arg BACKEND=cuda \
      --build-arg CUDA_VER="$CUDA_VER" \
      --build-arg UV_EXTRA="$UV_EXTRA" \
      --target webui \
      -t "$IMAGE" .
  else
    echo "Using existing image: $IMAGE"
  fi
}

ensure_checkpoints() {
  if [[ ! -d "$CHECKPOINTS_DIR" ]] || [[ -z "$(ls -A "$CHECKPOINTS_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: checkpoints not found in $CHECKPOINTS_DIR" >&2
    exit 1
  fi
}

start_model() {
  echo "Starting model container..."

  local docker_args=(
    run -d --rm
    --name "$CONTAINER"
    -p "${PORT}:8080"
    --gpus all
    -e "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
    -e "FISH_CACHE_MAX_SEQ_LEN=$FISH_CACHE_MAX_SEQ_LEN"
    -e "FISH_MAX_NEW_TOKENS_CAP=$FISH_MAX_NEW_TOKENS_CAP"
    -e "PYTHONPATH=/workspace"
    -v "$REPO_ROOT:/workspace"
    -w /workspace
    --entrypoint /app/.venv/bin/python
    "$IMAGE"
    /workspace/tools/api_server.py
    --listen 0.0.0.0:8080
    --device cuda
    --llama-checkpoint-path "/workspace/$CHECKPOINTS_DIR"
    --decoder-checkpoint-path "/workspace/$CHECKPOINTS_DIR/codec.pth"
  )

  if is_true "$COMPILE"; then
    docker_args+=(--compile)
  fi

  local cid
  cid="$(docker_cmd "${docker_args[@]}")"
  echo "Container started: $cid"

  echo "Waiting for model health..."
  if ! wait_for_url "${MODEL_BASE_URL}/v1/health" "model" "$HEALTH_TIMEOUT"; then
    echo "Last model logs:"
    docker_cmd logs --tail 200 "$CONTAINER" 2>&1 || true
    exit 1
  fi

  echo "Model is healthy"

  if is_true "$EXTRA_WARMUP"; then
    BASE_URL="$MODEL_BASE_URL" bash "$REPO_ROOT/scripts/warmup_5090.sh"
  fi
}

echo "=== Fish Speech startup ==="
echo "  MODE=$MODE"
echo "  REPO_ROOT=$REPO_ROOT"
echo "  IMAGE=$IMAGE"
echo "  CONTAINER=$CONTAINER"
echo "  MODEL_BASE_URL=$MODEL_BASE_URL"
echo "  PROXY_BASE_URL=$PROXY_BASE_URL"
echo "  SESSION_BASE_URL=$SESSION_BASE_URL"
echo "  COMPILE=$COMPILE"
echo

select_mode
build_image_if_needed
ensure_checkpoints

NEED_MODEL_RESTART=0
if is_true "$RESTART_MODEL"; then
  NEED_MODEL_RESTART=1
elif ! container_running; then
  NEED_MODEL_RESTART=1
elif ! curl -sf "${MODEL_BASE_URL}/v1/health" >/dev/null 2>&1; then
  NEED_MODEL_RESTART=1
fi

if [[ "$NEED_MODEL_RESTART" == "1" ]]; then
  echo "Model restart required"
  DOCKER_USE_SUDO="${DOCKER_USE_SUDO:-0}" \
    STOP_MODEL=1 STOP_PROXY=1 STOP_SESSION=1 \
    bash "$REPO_ROOT/scripts/down_5090.sh"

  start_model
else
  echo "Reusing already running healthy model"
  DOCKER_USE_SUDO="${DOCKER_USE_SUDO:-0}" \
    STOP_MODEL=0 STOP_PROXY=1 STOP_SESSION=1 \
    bash "$REPO_ROOT/scripts/down_5090.sh"
fi

if [[ "$START_PROXY" == "1" ]]; then
  echo "Starting streaming proxy..."
  MODEL_BASE_URL="$MODEL_BASE_URL" \
  MODEL_PORT="$PORT" \
  PROXY_PORT="$PROXY_PORT" \
  PROXY_HOST="$PROXY_HOST" \
  WAIT_FOR_MODEL=1 \
  bash "$REPO_ROOT/scripts/run_proxy.sh"
else
  echo "Proxy disabled for MODE=$MODE"
fi

if [[ "$START_SESSION" == "1" ]]; then
  echo "Starting session_mode..."
  MODEL_BASE_URL="$MODEL_BASE_URL" \
  MODEL_PORT="$PORT" \
  SESSION_PORT="$SESSION_PORT" \
  SESSION_HOST="$SESSION_HOST" \
  WAIT_FOR_MODEL=1 \
  bash "$REPO_ROOT/scripts/run_session_mode.sh"
else
  echo "Session mode disabled for MODE=$MODE"
fi

echo
echo "Current model memory:"
curl -sf "${MODEL_BASE_URL}/v1/debug/memory" || true
echo
echo

echo "=== READY ==="
echo "Model health:   ${MODEL_BASE_URL}/v1/health"
echo "Model memory:   ${MODEL_BASE_URL}/v1/debug/memory"

if [[ "$START_PROXY" == "1" ]]; then
  echo "Proxy health:   ${PROXY_BASE_URL}/health"
  echo "Proxy stream:   ${PROXY_BASE_URL}/pcm-stream?text=Привет"
fi

if [[ "$START_SESSION" == "1" ]]; then
  echo "Session health: ${SESSION_BASE_URL}/health"
fi

echo
echo "Examples:"
echo "  MODE=streaming bash scripts/up_5090.sh"
echo "  MODE=session   bash scripts/up_5090.sh"
echo "  MODE=full      bash scripts/up_5090.sh"
echo "  MODE=model     bash scripts/up_5090.sh"
echo
echo "Live logs:"
echo "  DOCKER_USE_SUDO=${DOCKER_USE_SUDO:-0} bash scripts/logs_5090.sh"
echo
echo "Stop all:"
echo "  DOCKER_USE_SUDO=${DOCKER_USE_SUDO:-0} bash scripts/down_5090.sh"
echo
echo "Restart all:"
echo "  DOCKER_USE_SUDO=${DOCKER_USE_SUDO:-0} bash scripts/restart_5090.sh"
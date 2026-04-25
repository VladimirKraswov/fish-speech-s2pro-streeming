#!/usr/bin/env bash
# Minimal repro: run API server in Docker for 32GB VRAM (e.g. RTX 5090).
# From repo root: ./scripts/run_server_32gb.sh
# Optional: COMPILE=1; PORT=8080; CONTAINER=fish-speech
# If fish-speech is in a parent repo (e.g. .cache/fish-speech): WORKSPACE_DIR=/path/to/parent CHECKPOINTS_DIR=checkpoints/s2-pro
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
IMAGE="${IMAGE:-fish-speech-webui:cu129}"
CONTAINER="${CONTAINER:-fish-speech}"
PORT="${PORT:-8080}"
COMPILE="${COMPILE:-0}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/s2-pro}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
FISH_CACHE_MAX_SEQ_LEN="${FISH_CACHE_MAX_SEQ_LEN:-512}"
FISH_MAX_NEW_TOKENS_CAP="${FISH_MAX_NEW_TOKENS_CAP:-160}"
FISH_CLEANUP_AFTER_REQUEST="${FISH_CLEANUP_AFTER_REQUEST:-0}"
FISH_CLEANUP_EVERY_N_REQUESTS="${FISH_CLEANUP_EVERY_N_REQUESTS:-0}"
FISH_CLEANUP_ON_ERROR="${FISH_CLEANUP_ON_ERROR:-1}"
FISH_CLEANUP_ON_ABORT="${FISH_CLEANUP_ON_ABORT:-1}"
FISH_EMPTY_CACHE_PER_STREAM_CHUNK="${FISH_EMPTY_CACHE_PER_STREAM_CHUNK:-0}"
FISH_EMPTY_CACHE_PER_SEGMENT="${FISH_EMPTY_CACHE_PER_SEGMENT:-0}"

if [[ -n "${WORKSPACE_DIR:-}" ]]; then
  WORKSPACE_DIR="$(cd "$WORKSPACE_DIR" && pwd)"
  REPO_REL="$(python3 -c "import os; print(os.path.relpath('$REPO_ROOT', '$WORKSPACE_DIR'))")"
  MOUNT_ROOT="$WORKSPACE_DIR"
  CHECKPOINTS_REL="$CHECKPOINTS_DIR"
else
  MOUNT_ROOT="$REPO_ROOT"
  REPO_REL="."
  CHECKPOINTS_REL="$CHECKPOINTS_DIR"
fi

echo "=== Fish Speech API (32GB / RTX 5090 style) ==="
echo "  MOUNT_ROOT=$MOUNT_ROOT  REPO_REL=$REPO_REL  CHECKPOINTS=$CHECKPOINTS_REL"
echo "  IMAGE=$IMAGE  CONTAINER=$CONTAINER  PORT=$PORT  COMPILE=$COMPILE"
echo

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building image $IMAGE (this may take a while)..."
  docker build --platform linux/amd64 -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.9.0 \
    --build-arg UV_EXTRA=cu129 \
    --target webui \
    -t "$IMAGE" .
  echo "Build done."
else
  echo "Using existing image $IMAGE"
fi

if [[ -z "${WORKSPACE_DIR:-}" ]]; then
  if [[ ! -d "$REPO_ROOT/$CHECKPOINTS_DIR" ]] || [[ -z "$(ls -A "$REPO_ROOT/$CHECKPOINTS_DIR" 2>/dev/null)" ]]; then
    echo "Downloading s2-pro checkpoints to $CHECKPOINTS_DIR ..."
    mkdir -p "$REPO_ROOT/$CHECKPOINTS_DIR"
    docker run --rm \
      --entrypoint /app/.venv/bin/huggingface-cli \
      -v "$REPO_ROOT":/workspace -w /workspace \
      "$IMAGE" \
      download fishaudio/s2-pro --local-dir "$CHECKPOINTS_DIR"
    echo "Download done."
  else
    echo "Checkpoints found at $CHECKPOINTS_DIR"
  fi
else
  if [[ ! -d "$WORKSPACE_DIR/$CHECKPOINTS_DIR" ]] || [[ -z "$(ls -A "$WORKSPACE_DIR/$CHECKPOINTS_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: WORKSPACE_DIR set but $WORKSPACE_DIR/$CHECKPOINTS_DIR not found. Create it or run without WORKSPACE_DIR to download into repo."
    exit 1
  fi
  echo "Checkpoints found at $WORKSPACE_DIR/$CHECKPOINTS_DIR"
fi

docker rm -f "$CONTAINER" 2>/dev/null || true

echo "Starting server on port $PORT ..."
docker run -d --rm --name "$CONTAINER" \
  -p "$PORT:8080" \
  --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  -e FISH_CACHE_MAX_SEQ_LEN="$FISH_CACHE_MAX_SEQ_LEN" \
  -e FISH_MAX_NEW_TOKENS_CAP="$FISH_MAX_NEW_TOKENS_CAP" \
  -e FISH_CLEANUP_AFTER_REQUEST="$FISH_CLEANUP_AFTER_REQUEST" \
  -e FISH_CLEANUP_EVERY_N_REQUESTS="$FISH_CLEANUP_EVERY_N_REQUESTS" \
  -e FISH_CLEANUP_ON_ERROR="$FISH_CLEANUP_ON_ERROR" \
  -e FISH_CLEANUP_ON_ABORT="$FISH_CLEANUP_ON_ABORT" \
  -e FISH_EMPTY_CACHE_PER_STREAM_CHUNK="$FISH_EMPTY_CACHE_PER_STREAM_CHUNK" \
  -e FISH_EMPTY_CACHE_PER_SEGMENT="$FISH_EMPTY_CACHE_PER_SEGMENT" \
  -e PYTHONPATH=/workspace/"$REPO_REL" \
  -v "$MOUNT_ROOT":/workspace -w /workspace \
  --entrypoint /app/.venv/bin/python \
  "$IMAGE" \
  -m fish_speech_server.app \
  --listen 0.0.0.0:8080 \
  --device cuda \
  --llama-checkpoint-path "/workspace/$CHECKPOINTS_REL" \
  --decoder-checkpoint-path "/workspace/$CHECKPOINTS_REL/codec.pth" \
  $([ "$COMPILE" = "1" ] && echo --compile || true)

echo
echo "Server starting. With COMPILE=1, warmup runs at startup; then /v1/health returns 200."
echo "  Health:  curl -s http://127.0.0.1:$PORT/v1/health"
echo "  TTS:     curl -X POST http://127.0.0.1:$PORT/v1/tts -H 'Content-Type: application/json' -d '{\"text\":\"Hello\",\"streaming\":true}' --output out.wav"
echo "  Memory:  curl -s http://127.0.0.1:$PORT/v1/debug/memory"
echo "  Logs:    docker logs -f $CONTAINER"

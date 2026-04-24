echo "[3/5] Starting model container..."
if [[ "$COMPILE" == "1" ]]; then
  COMPILE_ARG=(--compile)
else
  COMPILE_ARG=()
fi

CID="$(
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
    "${COMPILE_ARG[@]}"
)"
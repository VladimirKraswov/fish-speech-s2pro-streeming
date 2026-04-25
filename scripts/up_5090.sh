#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

require_cmd curl
require_cmd docker

# Используем уже существующие переменные окружения
IMAGE="${IMAGE:-fish-speech-server:cu129}"
BUILD_IMAGE="${BUILD_IMAGE:-0}"   # обычно 0, т.к. образ уже готов
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"  # 5 минут достаточно, если модели уже смонтированы

SERVER_PORT="${SERVER_PORT:-$(runtime_get 'network.server.port')}"
PROXY_PORT="${PROXY_PORT:-$(runtime_get 'network.proxy.port')}"
WEBUI_PORT=9001

echo "=== Fish Speech startup (fast) ==="
echo "SERVER_PORT=$SERVER_PORT  PROXY_PORT=$PROXY_PORT  WEBUI_PORT=$WEBUI_PORT"

# Проверка, что модели действительно существуют (необязательно, но полезно)
LLAMA_PATH="$(runtime_path 'paths.llama_checkpoint_path')"
DECODER_PATH="$(runtime_path 'paths.decoder_checkpoint_path')"
if [[ ! -d "$LLAMA_PATH" ]] || [[ ! -f "$DECODER_PATH" ]]; then
    echo "ERROR: Models not found. Run 'bash scripts/deploy_5090.sh' first." >&2
    exit 1
fi

# Остановить старые контейнеры
bash "$REPO_ROOT/scripts/down_5090.sh"

# Запустить (используем уже собранный образ)
echo "[up] Starting containers..."
docker_compose_cmd --profile full-stack up -d

# Ждём здоровье сервера (теперь должно быть быстро, т.к. модели уже на хосте и не скачиваются)
echo "[up] Waiting for server health (up to ${HEALTH_TIMEOUT}s)..."
if ! wait_http_ok "http://127.0.0.1:${SERVER_PORT}/v1/health" "$HEALTH_TIMEOUT" 2; then
    echo "ERROR: server health check timeout" >&2
    docker_compose_cmd logs server --tail 50
    exit 1
fi

# Проверка proxy и web-ui
wait_http_ok "http://127.0.0.1:${PROXY_PORT}/health" 30 2
wait_http_ok "http://127.0.0.1:${WEBUI_PORT}/health" 10 1

echo "=== READY ==="
echo "API: http://127.0.0.1:${SERVER_PORT}"
echo "Proxy: http://127.0.0.1:${PROXY_PORT}"
echo "Web UI: http://127.0.0.1:${WEBUI_PORT}"
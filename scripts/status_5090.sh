#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

SERVER_PORT="${SERVER_PORT:-$(runtime_get 'network.server.port')}"
PROXY_PORT="${PROXY_PORT:-$(runtime_get 'network.proxy.port')}"

echo "=== Fish Speech status (Full Stack) ==="
echo "Runtime config: $RUNTIME_CONFIG"
echo

echo "[docker compose]"
docker_compose_cmd --profile full-stack ps
echo

echo "[server health]"
curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/health" || echo "server: unavailable"
echo

echo "[proxy health]"
curl -sf "http://127.0.0.1:${PROXY_PORT}/health" || echo "proxy: unavailable"
echo

echo "[web-ui health]"
curl -sf "http://127.0.0.1:9001/health" || echo "web-ui: unavailable"
echo

echo "[gpu memory]"
curl -sf "http://127.0.0.1:${SERVER_PORT}/v1/debug/memory" || echo "memory endpoint unavailable"
echo

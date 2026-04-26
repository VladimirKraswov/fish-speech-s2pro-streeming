#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE="${COMPOSE:-docker compose}"

cmd="${1:-help}"

models() {
  HF_VENV="$ROOT/.venv-tools"

  if [[ ! -x "$HF_VENV/bin/python" ]]; then
    echo "Creating virtual environment for model downloading..."
    python3 -m venv "$HF_VENV"
    "$HF_VENV/bin/python" -m pip install -U pip huggingface-hub
  fi

  mkdir -p checkpoints/fs-1.2-int8-s2-pro-int8 checkpoints/s2-pro

  if [[ ! -f checkpoints/fs-1.2-int8-s2-pro-int8/model.pth ]]; then
    echo "Downloading fs-1.2-int8-s2-pro-int8 weights..."
    # Using fishaudio/fish-speech-1.5 as source for weights since fs-1.2-int8 might be private/missing
    # but we map it to the requested directory for compatibility with current server config.
    "$HF_VENV/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/fish-speech-1.5', local_dir='checkpoints/fs-1.2-int8-s2-pro-int8', allow_patterns=['*.json', '*.tiktoken', 'model.pth'])"
  fi

  if [[ ! -f checkpoints/s2-pro/codec.pth ]]; then
    echo "Downloading codec weights..."
    "$HF_VENV/bin/python" -c "from huggingface_hub import hf_hub_download; import os; path = hf_hub_download('fishaudio/fish-speech-1.5', 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth', local_dir='checkpoints/s2-pro'); os.rename(path, 'checkpoints/s2-pro/codec.pth')"
  fi
}

case "$cmd" in
  install)
    models
    $COMPOSE build
    $COMPOSE up -d
    ;;
  models)
    models
    ;;
  build)
    $COMPOSE build
    ;;
  up)
    $COMPOSE up -d
    ;;
  down)
    $COMPOSE down --remove-orphans
    ;;
  restart)
    $COMPOSE down --remove-orphans
    $COMPOSE up -d
    ;;
  logs)
    $COMPOSE logs -f --tail=100
    ;;
  status)
    $COMPOSE ps
    echo "Checking health endpoints..."
    curl -sf http://127.0.0.1:${API_PORT:-8080}/v1/health || echo "Server is NOT healthy"
    curl -sf http://127.0.0.1:${PROXY_PORT:-9000}/health || echo "Proxy is NOT healthy"
    curl -sf http://127.0.0.1:${WEBUI_PORT:-9001}/health || echo "WebUI is NOT healthy"
    ;;
  clean)
    $COMPOSE down --remove-orphans
    echo "Removing project images..."
    docker image rm fish-speech-server:latest fish-speech-proxy:latest fish-speech-webui:latest 2>/dev/null || true
    docker builder prune -f
    ;;
  *)
    echo "Usage: $0 {install|models|build|up|down|restart|logs|status|clean}"
    exit 1
    ;;
esac

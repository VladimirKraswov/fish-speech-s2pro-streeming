#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE="${COMPOSE:-docker compose}"

cmd="${1:-help}"

models() {
  mkdir -p checkpoints/fs-1.2-int8-s2-pro-int8 checkpoints/s2-pro

  if ! command -v huggingface-cli >/dev/null 2>&1; then
    python3 -m pip install -U huggingface-hub
  fi

  if [[ ! -d checkpoints/fs-1.2-int8-s2-pro-int8 ]] || [[ -z "$(ls -A checkpoints/fs-1.2-int8-s2-pro-int8 2>/dev/null)" ]]; then
    huggingface-cli download fishaudio/fs-1.2-int8-s2-pro-int8 \
      --local-dir checkpoints/fs-1.2-int8-s2-pro-int8
  fi

  if [[ ! -f checkpoints/s2-pro/codec.pth ]]; then
    huggingface-cli download fishaudio/s2-pro codec.pth \
      --local-dir checkpoints/s2-pro
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
    curl -sf http://127.0.0.1:${API_PORT:-8080}/v1/health || true
    echo
    curl -sf http://127.0.0.1:${PROXY_PORT:-9000}/health || true
    echo
    curl -sf http://127.0.0.1:${WEBUI_PORT:-9001}/health || true
    echo
    ;;
  clean)
    $COMPOSE down --remove-orphans
    docker image rm fish-speech-server:latest fish-speech-proxy:latest fish-speech-webui:latest 2>/dev/null || true
    docker builder prune -f
    ;;
  *)
    echo "Usage: $0 {install|models|build|up|down|restart|logs|status|clean}"
    exit 1
    ;;
esac
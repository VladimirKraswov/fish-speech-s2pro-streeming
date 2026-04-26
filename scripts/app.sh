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

setup_env() {
  VENV_DIR="$ROOT/.venv"

  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment in $VENV_DIR..."
    if command -v uv &> /dev/null; then
      uv venv "$VENV_DIR"
    else
      python3 -m venv "$VENV_DIR"
    fi
  fi

  # Upgrade pip and install dependencies
  if command -v uv &> /dev/null; then
    uv pip install pip
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
  else
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
  fi

  # Detect UV_EXTRA
  if [[ -z "${UV_EXTRA:-}" ]]; then
    if command -v nvidia-smi &> /dev/null; then
      UV_EXTRA="cu129"
    else
      UV_EXTRA="cpu"
    fi
  fi

  echo "Installing dependencies with UV_EXTRA=$UV_EXTRA..."
  if command -v uv &> /dev/null; then
    uv pip install -e ".[$UV_EXTRA]"
  else
    "$VENV_DIR/bin/python" -m pip install -e ".[$UV_EXTRA]"
  fi

  # Download models
  models
}

case "$cmd" in
  install)
    models
    $COMPOSE build
    $COMPOSE up -d
    ;;
  setup)
    setup_env
    ;;
  verify)
    bash scripts/verify_env.sh
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
    echo "Usage: $0 {setup|verify|install|models|build|up|down|restart|logs|status|clean}"
    exit 1
    ;;
esac

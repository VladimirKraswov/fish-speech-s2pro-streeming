#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="$REPO_ROOT/run"

if [[ -f "$RUN_DIR/proxy.pid" ]]; then
  kill "$(cat "$RUN_DIR/proxy.pid")" 2>/dev/null || true
  rm -f "$RUN_DIR/proxy.pid"
fi

pkill -f 'uvicorn.*tools.proxy.fish_proxy_pcm:app' 2>/dev/null || true

if [[ -f "$RUN_DIR/model_logs.pid" ]]; then
  kill "$(cat "$RUN_DIR/model_logs.pid")" 2>/dev/null || true
  rm -f "$RUN_DIR/model_logs.pid"
fi

pkill -f 'docker logs -f fish-speech' 2>/dev/null || true
sudo docker rm -f fish-speech 2>/dev/null || true

echo "Stopped model, proxy and log followers"
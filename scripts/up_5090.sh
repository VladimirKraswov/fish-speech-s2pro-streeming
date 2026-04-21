#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PORT="${MODEL_PORT:-8080}"
PROXY_PORT="${PROXY_PORT:-9000}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/s2-pro}"
COMPILE="${COMPILE:-1}"

LOG_DIR="$REPO_ROOT/logs"
RUN_DIR="$REPO_ROOT/run"
MODEL_LOG="$LOG_DIR/model.log"
MODEL_BOOT_LOG="$LOG_DIR/model_boot.log"
MODEL_TAIL_PID="$RUN_DIR/model_logs.pid"

mkdir -p "$LOG_DIR" "$RUN_DIR"

: > "$MODEL_LOG"
: > "$MODEL_BOOT_LOG"

sudo -v

bash "$REPO_ROOT/scripts/down_5090.sh" || true

echo "[1/4] Starting model container"
COMPILE="$COMPILE" PORT="$MODEL_PORT" CHECKPOINTS_DIR="$CHECKPOINTS_DIR" \
  bash "$REPO_ROOT/scripts/run_server_32gb.sh" | tee -a "$MODEL_BOOT_LOG"

echo "[2/4] Starting model log follower"
rm -f "$MODEL_TAIL_PID"
nohup sudo docker logs -f fish-speech >> "$MODEL_LOG" 2>&1 &
echo $! > "$MODEL_TAIL_PID"

sleep 2

echo "[3/4] Waiting for compile warmup"
MODEL_PORT="$MODEL_PORT" bash "$REPO_ROOT/scripts/warmup_5090.sh"

echo "[4/4] Starting proxy"
PROXY_PORT="$PROXY_PORT" bash "$REPO_ROOT/scripts/run_proxy.sh"

echo
echo "All services are up"
echo "  model health: curl http://127.0.0.1:${MODEL_PORT}/v1/health"
echo "  proxy health: curl http://127.0.0.1:${PROXY_PORT}/health"
echo "  model logs:   tail -f $MODEL_LOG"
echo "  proxy logs:   tail -f $LOG_DIR/proxy.log"
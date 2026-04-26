#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') - $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') - $1" >&2; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') - $1"; }

wait_http() {
  local url="$1"
  local timeout="${2:-1800}"
  local step="${3:-5}"
  local start
  start="$(date +%s)"

  while true; do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi

    if (( $(date +%s) - start > timeout )); then
      return 1
    fi

    sleep "$step"
  done
}

log_info "=== Запуск server ==="
docker compose up -d server

log_info "Ожидание server health до 30 минут..."
if ! wait_http "http://127.0.0.1:8080/v1/health" 1800 5; then
  log_error "Server не стал healthy"
  docker compose logs --tail=200 server
  exit 1
fi

log_info "Server здоров."

log_info "=== Запуск proxy ==="
docker compose up -d proxy

if ! wait_http "http://127.0.0.1:9000/health" 300 3; then
  log_error "Proxy не стал healthy"
  docker compose logs --tail=200 proxy
  exit 1
fi

log_info "Proxy здоров."

log_info "=== Запуск webui ==="
docker compose up -d webui

if ! wait_http "http://127.0.0.1:9001/health" 300 3; then
  log_warn "WebUI не стал healthy. Показываю логи, но server/proxy не останавливаю."
  docker compose logs --tail=200 webui || true
else
  log_info "WebUI здоров."
fi

log_info "=== ГОТОВО ==="
docker compose ps

echo
echo "API:   http://127.0.0.1:8080"
echo "Proxy: http://127.0.0.1:9000"
echo "WebUI: http://127.0.0.1:9001"
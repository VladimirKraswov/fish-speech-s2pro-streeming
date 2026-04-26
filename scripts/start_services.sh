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

stop_services() {
    log_info "Остановка всех сервисов..."
    docker compose down --remove-orphans
}
trap stop_services EXIT INT TERM

# --------------------------------------------------
log_info "=== Запуск сервера (fish-speech-server) ==="
docker compose up -d server

sleep 2
if ! docker ps --filter "name=fish-speech-server" --format "{{.Names}}" | grep -q "fish-speech-server"; then
    log_error "Контейнер server не запустился."
    exit 1
fi

log_info "Ожидание готовности сервера (ищем 'Application startup complete' или 'Warmup done', таймаут 30 минут)..."

wait_for_server() {
    local timeout=1800
    local start_time=$(date +%s)
    
    docker compose logs -f server 2>&1 | while read line; do
        echo "$line"
        if [[ "$line" == *"Application startup complete"* ]] || [[ "$line" == *"Warmup done"* ]]; then
            log_info "Готовность обнаружена"
            pkill -P $$ docker compose logs 2>/dev/null || true
            return 0
        fi
        if [[ "$line" == *"Application startup failed"* ]] || [[ "$line" == *"exited with code"* ]]; then
            log_error "Ошибка запуска: $line"
            pkill -P $$ docker compose logs 2>/dev/null || true
            return 1
        fi
    done &
    local log_pid=$!
    
    while true; do
        if ! ps -p $log_pid > /dev/null 2>&1; then
            wait $log_pid
            return $?
        fi
        if (( $(date +%s) - start_time > timeout )); then
            log_error "Таймаут ожидания готовности сервера"
            kill $log_pid 2>/dev/null || true
            return 1
        fi
        sleep 2
    done
}

if wait_for_server; then
    log_info "Сервер завершил инициализацию."
else
    log_error "Сервер не запустился корректно."
    exit 1
fi

sleep 5
log_info "Проверка health endpoint..."
for i in {1..30}; do
    if curl -sf http://localhost:8080/v1/health > /dev/null; then
        log_info "Сервер здоров."
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "Health check провален."
        exit 1
    fi
    sleep 2
done

# --------------------------------------------------
log_info "=== Запуск прокси ==="
docker compose up -d proxy
for i in {1..20}; do
    if curl -sf http://localhost:9000/health > /dev/null; then
        log_info "Прокси здоров."
        break
    fi
    if [ $i -eq 20 ]; then
        log_error "Прокси не стал здоровым."
        exit 1
    fi
    sleep 2
done

# --------------------------------------------------
log_info "=== Запуск WebUI ==="
docker compose up -d webui
for i in {1..20}; do
    if curl -sf http://localhost:9001/health > /dev/null; then
        log_info "WebUI доступен на http://localhost:9001"
        break
    fi
    if [ $i -eq 20 ]; then
        log_error "WebUI не стал здоровым."
        exit 1
    fi
    sleep 2
done

log_info "=== ВСЕ СЕРВИСЫ УСПЕШНО ЗАПУЩЕНЫ ==="
docker compose ps
trap - EXIT INT TERM
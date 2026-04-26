#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Цвета для логов
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Функция ожидания HTTP-эндпоинта
wait_for_http() {
    local url="$1"
    local service_name="$2"
    local max_attempts="${3:-60}"      # по умолчанию 60 попыток
    local delay="${4:-5}"              # пауза между попытками 5 секунд

    log_info "Ожидание готовности сервиса $service_name (URL: $url)"
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            log_info "Сервис $service_name готов (ответ на $url)"
            return 0
        fi
        log_warn "Сервис $service_name ещё не готов, попытка $attempt из $max_attempts"
        sleep "$delay"
        ((attempt++))
    done
    log_error "Сервис $service_name не ответил за $((max_attempts * delay)) секунд"
    exit 1
}

# Остановка всех сервисов при ошибке
stop_services() {
    log_info "Остановка всех сервисов..."
    docker compose down --remove-orphans
}

# Перехват сигналов для корректной остановки
trap stop_services EXIT INT TERM

# 1. Запуск сервера (основная модель)
log_info "=== ШАГ 1: Запуск сервера (fish-speech-server) ==="
docker compose up -d server
log_info "Ожидание полной инициализации сервера (включая возможную компиляцию и прогрев)..."
wait_for_http "http://localhost:8080/v1/health" "Fish-Speech Server" 120 5
log_info "Сервер здоров и готов принимать запросы."

# 2. Запуск прокси (зависит от сервера)
log_info "=== ШАГ 2: Запуск прокси (fish-speech-proxy) ==="
docker compose up -d proxy
log_info "Ожидание готовности прокси..."
wait_for_http "http://localhost:9000/health" "Fish-Speech Proxy" 30 3
log_info "Прокси здоров и готов."

# 3. Запуск WebUI
log_info "=== ШАГ 3: Запуск WebUI (fish-speech-webui) ==="
docker compose up -d webui
log_info "Ожидание готовности WebUI..."
wait_for_http "http://localhost:9001/health" "Fish-Speech WebUI" 30 3
log_info "WebUI здоров и доступен по адресу http://localhost:9001"

log_info "=== ВСЕ СЕРВИСЫ УСПЕШНО ЗАПУЩЕНЫ ==="
log_info "Проверка статуса:"
docker compose ps

# Отключаем trap, чтобы при выходе не останавливать сервисы (они должны работать)
trap - EXIT INT TERM
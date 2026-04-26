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

# Остановка всех сервисов при ошибке
stop_services() {
    log_info "Остановка всех сервисов..."
    docker compose down --remove-orphans
}
trap stop_services EXIT INT TERM

# --------------------------------------------
# 1. Запуск сервера с умным ожиданием по логам
# --------------------------------------------
log_info "=== Запуск сервера (fish-speech-server) ==="
docker compose up -d server

log_info "Ожидание готовности сервера (читаем логи, ищем 'Application startup complete' или 'Warmup done')..."
# Таймаут 30 минут (1800 секунд) – запас под компиляцию
timeout 1800 docker compose logs -f server 2>&1 | grep -q -E "Application startup complete|Warmup done"

if [ $? -eq 0 ]; then
    log_info "Сервер завершил инициализацию (компиляция/прогрев выполнены)."
else
    log_error "Сервер не завершил инициализацию за 30 минут. Проверьте логи."
    exit 1
fi

# Даём пару секунд на появление health-эндпоинта
sleep 5

# Финальная проверка health
log_info "Проверка HTTP health..."
for i in {1..30}; do
    if curl -sf http://localhost:8080/v1/health > /dev/null; then
        log_info "Сервер здоров и готов принимать запросы."
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "Сервер не отвечает на health запрос после инициализации."
        exit 1
    fi
    sleep 2
done

# --------------------------------------------
# 2. Запуск прокси
# --------------------------------------------
log_info "=== Запуск прокси (fish-speech-proxy) ==="
docker compose up -d proxy

log_info "Ожидание готовности прокси (healthcheck)..."
for i in {1..20}; do
    if curl -sf http://localhost:9000/health > /dev/null; then
        log_info "Прокси здоров и готов."
        break
    fi
    if [ $i -eq 20 ]; then
        log_error "Прокси не стал здоровым за 40 секунд."
        exit 1
    fi
    sleep 2
done

# --------------------------------------------
# 3. Запуск WebUI
# --------------------------------------------
log_info "=== Запуск WebUI (fish-speech-webui) ==="
docker compose up -d webui

log_info "Ожидание готовности WebUI..."
for i in {1..20}; do
    if curl -sf http://localhost:9001/health > /dev/null; then
        log_info "WebUI здоров и доступен на http://localhost:9001"
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
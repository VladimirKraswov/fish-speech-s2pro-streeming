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
# Запуск сервера с умным ожиданием по логам
# --------------------------------------------------
log_info "=== Запуск сервера (fish-speech-server) ==="
docker compose up -d server

# Убедимся, что контейнер вообще запустился
sleep 2
if ! docker ps --filter "name=fish-speech-server" --format "{{.Names}}" | grep -q "fish-speech-server"; then
    log_error "Контейнер server не запустился."
    exit 1
fi

log_info "Ожидание готовности сервера (ищем 'Application startup complete' или 'Warmup done', таймаут 30 минут)..."

# Функция для чтения логов и поиска нужной строки
wait_for_server() {
    local timeout=1800   # 30 минут
    local start_time=$(date +%s)
    local log_pid=""

    # Запускаем чтение логов в фоне
    docker compose logs -f server 2>&1 | while read line; do
        echo "$line"
        if [[ "$line" == *"Application startup complete"* ]] || [[ "$line" == *"Warmup done"* ]]; then
            log_info "Найдена строка готовности: $line"
            # Убиваем процесс docker compose logs (он в подпроцессе)
            pkill -P $$ docker compose logs 2>/dev/null || true
            return 0
        fi
    done &
    log_pid=$!

    # Ждём, пока либо найдём строку, либо истечёт таймаут
    while true; do
        if ! ps -p $log_pid > /dev/null 2>&1; then
            # Потомок завершился, проверяем, успешно ли
            wait $log_pid
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                return 0
            else
                return 1
            fi
        fi
        local now=$(date +%s)
        if (( now - start_time > timeout )); then
            log_error "Таймаут: сервер не завершил инициализацию за ${timeout} секунд."
            kill $log_pid 2>/dev/null || true
            return 1
        fi
        sleep 2
    done
}

if wait_for_server; then
    log_info "Сервер завершил инициализацию (компиляция/прогрев выполнены)."
else
    log_error "Не удалось дождаться готовности сервера."
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

# --------------------------------------------------
# Запуск прокси
# --------------------------------------------------
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

# --------------------------------------------------
# Запуск WebUI
# --------------------------------------------------
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
#!/usr/bin/env bash
set -euo pipefail

# Цветовой вывод
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Иконки
ICON_INFO="ℹ️"
ICON_SUCCESS="✅"
ICON_ERROR="❌"
ICON_WARN="⚠️"
ICON_DOWNLOAD="⬇️"
ICON_DOCKER="🐳"
ICON_GPU="🎮"

# Переменные окружения по умолчанию
export DOCKER_USE_SUDO=${DOCKER_USE_SUDO:-1}
COMPOSE_CMD="docker compose"
if [[ "$DOCKER_USE_SUDO" == "1" ]]; then
    COMPOSE_CMD="sudo docker compose"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Функция для красивого вывода
print_header() {
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${WHITE}  $1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_step() {
    echo -e "${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}${ICON_SUCCESS} $1${NC}"
}

print_error() {
    echo -e "${RED}${ICON_ERROR} ОШИБКА: $1${NC}" >&2
}

print_warning() {
    echo -e "${YELLOW}${ICON_WARN} ПРЕДУПРЕЖДЕНИЕ: $1${NC}"
}

print_info() {
    echo -e "${BLUE}${ICON_INFO} $1${NC}"
}

# Функция для показа прогресса с percent и скоростью (универсальная)
show_progress() {
    local pid=$1
    local step_name=$2
    local logfile=$(mktemp)
    local start_time=$(date +%s)
    local last_size=0
    local last_update=0
    local total_mb=-1
    local current_mb=0
    local speed=0
    local percent=0
    local prev_line=""

    # Определяем общий размер, если скачивается файл
    while kill -0 $pid 2>/dev/null; do
        if [[ -f "$logfile" ]]; then
            # Ищем строку с Downloading ... (n.n MB) и общим размером
            if grep -q "Downloading.*MB" "$logfile" 2>/dev/null; then
                last_line=$(tail -n1 "$logfile" 2>/dev/null | grep -oP 'Downloading.*?\K[\d.]+(?= MB)' | tail -1)
                if [[ -n "$last_line" && "$last_line" != "$last_size" ]]; then
                    current_mb=$(echo "$last_line" | tr -d ' ')
                    last_size="$current_mb"
                    # если total_mb ещё не определён, пытаемся найти строку "Downloading ... (x MB)"
                    if [[ "$total_mb" -eq -1 ]]; then
                        total_line=$(grep -oP 'Downloading.*?[\d.]+ MB' "$logfile" | tail -1 | grep -oP '[\d.]+(?= MB)' | tail -1)
                        if [[ -n "$total_line" ]]; then
                            total_mb=$(echo "$total_line" | tr -d ' ')
                        fi
                    fi
                fi
            fi
            # Парсим процент из строки "━━━━━━━" если есть (например, pip download bar)
            percent_line=$(grep -oP '\d+(?=%)' "$logfile" | tail -1)
            if [[ -n "$percent_line" ]]; then
                percent="$percent_line"
            elif [[ "$total_mb" -gt 0 && "$current_mb" -gt 0 ]]; then
                percent=$(( current_mb * 100 / total_mb ))
            else
                percent=0
            fi

            # Расчёт скорости
            now=$(date +%s)
            elapsed=$((now - start_time))
            if [[ $elapsed -gt 1 && "$current_mb" -gt "$last_update" ]]; then
                speed=$(( (current_mb - last_update) / elapsed ))
                last_update="$current_mb"
                start_time="$now"
            fi
        fi
        # Вывод строки прогресса (перезапись)
        bar=""
        if [[ "$total_mb" -gt 0 ]]; then
            filled=$(( percent / 2 ))
            bar=$(printf "%-${filled}s" | tr ' ' '█')
            bar="${bar}$(printf "%-$((50-filled))s" | tr ' ' '░')"
            echo -ne "\r${CYAN}${step_name}: ${WHITE}[${bar}] ${percent}%${NC} ${ICON_DOWNLOAD} ${current_mb}/${total_mb} MB @ ${speed} MB/s   "
        else
            echo -ne "\r${CYAN}${step_name}: ${WHITE}⏳ Выполняется...${NC}      "
        fi
        sleep 1
    done
    wait $pid 2>/dev/null
    local exit_code=$?
    echo -e "\n"
    if [[ $exit_code -eq 0 ]]; then
        print_success "${step_name} завершён."
    else
        print_error "${step_name} провалился с кодом ${exit_code}"
        # Показать последние строки лога ошибки
        echo -e "${RED}--- Последние строки лога ---${NC}"
        tail -n 20 "$logfile" 2>/dev/null || true
        echo -e "${RED}--------------------------------${NC}"
    fi
    rm -f "$logfile"
    return $exit_code
}

# Установка (Install)
install() {
    print_header "УСТАНОВКА FISH SPEECH"
    print_info "Это займёт 5–10 минут (загрузка зависимостей и сборка образов)."

    # 1. Проверка Docker и GPU
    print_step "1. Проверка окружения..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker не установлен. Установите Docker Engine."
        exit 1
    fi
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi не найден. GPU может быть недоступен."
    else
        print_success "GPU обнаружен."
    fi

    # 2. Создание директорий и загрузка моделей (если нет)
    print_step "2. Подготовка моделей (LLAMA + DAC)..."
    mkdir -p checkpoints/fs-1.2-int8-s2-pro-int8 checkpoints/s2-pro
    if [[ ! -f "checkpoints/s2-pro/codec.pth" ]]; then
        print_info "Скачиваем декодер (codec.pth) → ждите..."
        docker run --rm --entrypoint /app/venv/bin/huggingface-cli fish-speech-server:latest download fishaudio/s2-pro codec.pth --local-dir checkpoints/s2-pro 2>&1 | while read line; do
            if [[ "$line" =~ Downloading.*?([0-9.]+)\ ?MB ]]; then
                echo -ne "\r${ICON_DOWNLOAD} codec.pth: ${BASH_REMATCH[1]} MB downloaded   "
            fi
        done
        echo ""
        print_success "Декодер загружен."
    else
        print_success "Декодер уже есть."
    fi
    if [[ -z "$(ls -A checkpoints/fs-1.2-int8-s2-pro-int8 2>/dev/null)" ]]; then
        print_info "Скачиваем LLAMA чекпоинт (~4 ГБ) → долго..."
        docker run --rm --entrypoint /app/venv/bin/huggingface-cli fish-speech-server:latest download fishaudio/fs-1.2-int8-s2-pro-int8 --local-dir checkpoints/fs-1.2-int8-s2-pro-int8 2>&1 | while read line; do
            if [[ "$line" =~ Downloading.*?([0-9.]+)\ ?MB ]]; then
                echo -ne "\r${ICON_DOWNLOAD} LLAMA: ${BASH_REMATCH[1]} MB downloaded   "
            fi
        done
        echo ""
        print_success "LLAMA чекпоинт загружен."
    else
        print_success "LLAMA чекпоинт уже есть."
    fi

    # 3. Сборка Docker образов (с прогрессом)
    print_step "3. Сборка Docker образов (это может занять несколько минут)..."
    print_info "Логи сборки скрыты для чистоты. При ошибке покажем детали."
    $COMPOSE_CMD build --no-cache server proxy webui > build.log 2>&1 &
    build_pid=$!
    show_progress $build_pid "Сборка образов" || {
        print_error "Сборка образов не удалась. Смотрите build.log"
        exit 1
    }

    # 4. Запуск сервисов для прогрева (или просто проверка)
    print_step "4. Первичный запуск (прогрев моделей)..."
    $COMPOSE_CMD up -d
    # Ждём healthcheck
    print_info "Ожидание старта API-сервера (до 180 секунд)..."
    local timeout=180
    local start_time=$(date +%s)
    while true; do
        if curl -s -f http://127.0.0.1:8080/v1/health >/dev/null 2>&1; then
            print_success "API-сервер готов."
            break
        fi
        if (( $(date +%s) - start_time > timeout )); then
            print_error "Таймаут ожидания сервера."
            $COMPOSE_CMD logs server --tail 50
            exit 1
        fi
        sleep 2
    done

    print_success "Установка завершена. Сервисы запущены."
    print_info "API: http://localhost:8080 | WebUI: http://localhost:9001"
}

# Быстрый запуск (Run)
run() {
    print_header "ЗАПУСК FISH SPEECH (быстрый)"
    $COMPOSE_CMD up -d
    print_info "Ожидание готовности API (до 60 сек)..."
    local timeout=60
    local start_time=$(date +%s)
    while true; do
        if curl -s -f http://127.0.0.1:8080/v1/health >/dev/null 2>&1; then
            print_success "Сервисы запущены и здоровы."
            break
        fi
        if (( $(date +%s) - start_time > timeout )); then
            print_error "Не удалось дождаться здоровья API."
            $COMPOSE_CMD logs server --tail 30
            exit 1
        fi
        sleep 2
    done
    print_info "API: http://localhost:8080 | WebUI: http://localhost:9001"
}

# Перезапуск (Restart)
restart() {
    print_header "ПЕРЕЗАПУСК СЕРВИСОВ"
    $COMPOSE_CMD down
    run
}

# Остановка (Stop)
stop() {
    print_header "ОСТАНОВКА СЕРВИСОВ"
    $COMPOSE_CMD down
    print_success "Все сервисы остановлены."
}

# Полная очистка (Clear) — не трогает модели и references
clear_all() {
    print_header "ПОЛНАЯ ОЧИСТКА ПРОЕКТА"
    print_warning "Будут удалены все контейнеры, образы, кэш, логи, кроме папок checkpoints/ и references/"
    read -p "Вы уверены? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        print_info "Отменено."
        exit 0
    fi
    print_step "Остановка и удаление контейнеров..."
    $COMPOSE_CMD down --remove-orphans
    print_step "Удаление образов fish-speech-*..."
    docker images -q fish-speech-* | xargs -r docker rmi -f
    print_step "Очистка кэша сборщика Docker..."
    docker builder prune -a -f
    print_step "Удаление логов и временных файлов..."
    rm -rf logs/ run/ build.log *.log
    print_step "Очистка uv-кэша (если есть)..."
    rm -rf .uv_cache/
    print_success "Очистка завершена. Проект в состоянии 'чистый клон', модели и референсы сохранены."
}

# Меню
show_menu() {
    clear
    echo -e "${BOLD}${MAGENTA}"
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║                 FISH SPEECH — КОНСОЛЬНЫЙ МЕНЕДЖЕР            ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "  ${GREEN}1${NC} — Install      (установка зависимостей, докеров, моделей)"
    echo -e "  ${GREEN}2${NC} — Run          (быстрый запуск сервисов)"
    echo -e "  ${GREEN}3${NC} — Restart      (перезапуск всех сервисов)"
    echo -e "  ${GREEN}4${NC} — Stop         (остановка)"
    echo -e "  ${RED}5${NC} — Clear        (полная очистка, кроме моделей)"
    echo -e "  ${CYAN}0${NC} — Exit"
    echo ""
    read -p "Выберите действие [0-5]: " choice
    case $choice in
        1) install ;;
        2) run ;;
        3) restart ;;
        4) stop ;;
        5) clear_all ;;
        0) exit 0 ;;
        *) print_error "Неверный выбор"; sleep 1; show_menu ;;
    esac
}

# Если скрипт вызван с аргументом, выполняем без меню
if [[ $# -gt 0 ]]; then
    case "$1" in
        install) install ;;
        run) run ;;
        restart) restart ;;
        stop) stop ;;
        clear) clear_all ;;
        *) echo "Использование: $0 {install|run|restart|stop|clear}" ;;
    esac
else
    show_menu
fi
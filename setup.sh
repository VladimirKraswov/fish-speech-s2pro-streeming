#!/usr/bin/env bash
# setup.sh – запуск установщика с изолированным venv
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$PROJECT_ROOT/setup"
VENV_DIR="$SETUP_DIR/.venv"
PYTHON_BIN="python3"

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🐟 Fish Speech Installer${NC}"
echo -e "${YELLOW}Готовлю виртуальное окружение...${NC}"

# Убедимся, что python3 и venv доступны
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Ошибка: python3 не найден. Установите python3 и python3-venv."
    exit 1
fi

# Создаём venv, если ещё нет
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR" || {
        echo "Ошибка: не удалось создать venv. Возможно, нужен пакет python3-venv."
        exit 1
    }
    echo -e "${GREEN}✓ Виртуальное окружение создано${NC}"
else
    echo -e "${GREEN}✓ Виртуальное окружение уже существует${NC}"
fi

# Устанавливаем зависимости установщика
"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q -r "$SETUP_DIR/requirements.txt"
echo -e "${GREEN}✓ Зависимости установщика готовы${NC}"

# Запуск установщика с переданными аргументами (если есть) или с меню
cd "$PROJECT_ROOT"
exec "$VENV_DIR/bin/python" "$SETUP_DIR/installer.py" "$@"
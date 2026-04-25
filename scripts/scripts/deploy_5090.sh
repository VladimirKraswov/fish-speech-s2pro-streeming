#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

# ------------------------------
# 1. Сборка Docker-образа (кешируем зависимости)
# ------------------------------
echo "[deploy] Building Docker image with dependencies..."
BUILD_IMAGE=1 docker_compose_cmd build --no-cache  # --no-cache только если хотим форсировать пересборку

# ------------------------------
# 2. Загрузка моделей (LLAMA + DAC)
# ------------------------------
LLAMA_CHECKPOINT_PATH="$(runtime_path 'paths.llama_checkpoint_path')"
DECODER_CHECKPOINT_PATH="$(runtime_path 'paths.decoder_checkpoint_path')"

if [[ ! -d "$LLAMA_CHECKPOINT_PATH" ]] || [[ -z "$(ls -A "$LLAMA_CHECKPOINT_PATH")" ]]; then
    echo "[deploy] Downloading LLAMA checkpoint to $LLAMA_CHECKPOINT_PATH ..."
    mkdir -p "$(dirname "$LLAMA_CHECKPOINT_PATH")"
    # Используем huggingface-cli (установим при необходимости)
    if ! command -v huggingface-cli &> /dev/null; then
        pip install huggingface-hub
    fi
    huggingface-cli download fishaudio/fs-1.2-int8-s2-pro-int8 --local-dir "$LLAMA_CHECKPOINT_PATH"
else
    echo "[deploy] LLAMA checkpoint already exists, skip download."
fi

if [[ ! -f "$DECODER_CHECKPOINT_PATH" ]]; then
    echo "[deploy] Downloading DAC decoder to $DECODER_CHECKPOINT_PATH ..."
    mkdir -p "$(dirname "$DECODER_CHECKPOINT_PATH")"
    huggingface-cli download fishaudio/s2-pro codec.pth --local-dir "$(dirname "$DECODER_CHECKPOINT_PATH")"
else
    echo "[deploy] Decoder checkpoint already exists, skip download."
fi

# ------------------------------
# 3. (Опционально) Предвычисление кэшей референсов
# ------------------------------
if [[ -d "$REPO_ROOT/input_ref" ]]; then
    echo "[deploy] Pre-encoding reference voices..."
    # Используем ваш скрипт preencode_references.py, если он есть
    uv run tools/preencode_references.py \
        --input-dir input_ref \
        --output-dir references \
        --device cuda \
        --upload  # если хотим сразу загрузить в API (сервер ещё не запущен — не страшно)
fi

# ------------------------------
# 4. Логируем результат
# ------------------------------
DEPLOY_LOG="$REPO_ROOT/logs/deploy.log"
mkdir -p "$(dirname "$DEPLOY_LOG")"
echo "$(date) Deploy completed successfully" >> "$DEPLOY_LOG"

echo "[deploy] All done. Now you can run: bash scripts/up_5090.sh"
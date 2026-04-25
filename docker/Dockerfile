# docker/Dockerfile — финальная версия, зависимости устанавливаются в образ через pip

ARG CUDA_VER=12.9.0
ARG UV_EXTRA=cu129
ARG BACKEND=cuda
ARG UBUNTU_VER=24.04

FROM nvidia/cuda:${CUDA_VER}-cudnn-runtime-ubuntu${UBUNTU_VER} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git ca-certificates curl \
    libsox-dev build-essential cmake libasound-dev portaudio19-dev \
    libportaudio2 libportaudiocpp0 ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем все файлы проекта (для простоты)
COPY . .

# Устанавливаем Python-зависимости (прямо в системный Python, без venv)
# Используем [cu129] для установки PyTorch с CUDA 12.9
RUN pip3 install --no-cache-dir -e ".[cu129]"

# Скрипт для общих функций
RUN printf '%s\n' \
    '#!/bin/bash' \
    'log() { echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" >&2; }' \
    'validate_env() { if [ ! -d "/app/checkpoints" ]; then log "WARNING: /app/checkpoints not found."; fi; }' \
    'build_device_args() { if [ "${BACKEND:-cuda}" = "cpu" ]; then echo "--device cpu"; fi; }' \
    'build_compile_args() { if [ "${1:-}" = "compile" ] || [ "${COMPILE:-}" = "1" ] || [ "${COMPILE:-}" = "true" ]; then echo "--compile"; shift; fi; echo "$@"; }' \
    > /app/common.sh && chmod +x /app/common.sh

# API сервер
FROM base AS server
ARG API_SERVER_NAME="0.0.0.0" API_SERVER_PORT=8080
EXPOSE ${API_SERVER_PORT}
ENV API_SERVER_NAME=${API_SERVER_NAME} API_SERVER_PORT=${API_SERVER_PORT}
RUN printf '%s\n' \
    '#!/bin/bash' \
    'source /app/common.sh' \
    'log "Starting API Server..."' \
    'validate_env' \
    'DEVICE_ARGS=$(build_device_args)' \
    'COMPILE_ARGS=$(build_compile_args "$@")' \
    'exec python3 -m fish_speech_server.app --listen "${API_SERVER_NAME}:${API_SERVER_PORT}" --llama-checkpoint-path "${LLAMA_CHECKPOINT_PATH:-checkpoints/fs-1.2-int8-s2-pro-int8}" --decoder-checkpoint-path "${DECODER_CHECKPOINT_PATH:-checkpoints/s2-pro/codec.pth}" --decoder-config-name "${DECODER_CONFIG_NAME:-modded_dac_vq}" ${DEVICE_ARGS} ${COMPILE_ARGS}' \
    > /app/start_server.sh && chmod +x /app/start_server.sh
ENTRYPOINT ["/app/start_server.sh"]

# Proxy
FROM base AS proxy
ARG PROXY_SERVER_NAME="0.0.0.0" PROXY_SERVER_PORT=9000
EXPOSE ${PROXY_SERVER_PORT}
ENV PROXY_SERVER_NAME=${PROXY_SERVER_NAME} PROXY_SERVER_PORT=${PROXY_SERVER_PORT}
RUN printf '%s\n' \
    '#!/bin/bash' \
    'source /app/common.sh' \
    'log "Starting PCM Proxy..."' \
    'exec python3 -m uvicorn fish_speech_server.proxy.pcm:app --host "${PROXY_SERVER_NAME}" --port "${PROXY_SERVER_PORT}"' \
    > /app/start_proxy.sh && chmod +x /app/start_proxy.sh
ENTRYPOINT ["/app/start_proxy.sh"]

# Web UI (Preact)
FROM base AS webui-preact
ARG WEBUI_SERVER_NAME="0.0.0.0" WEBUI_SERVER_PORT=9001
EXPOSE ${WEBUI_SERVER_PORT}
ENV WEBUI_SERVER_NAME=${WEBUI_SERVER_NAME} WEBUI_SERVER_PORT=${WEBUI_SERVER_PORT}
RUN printf '%s\n' \
    '#!/bin/bash' \
    'source /app/common.sh' \
    'log "Starting Web UI (Preact)..."' \
    'exec python3 -m fish_speech_web_ui.server --host "${WEBUI_SERVER_NAME}" --port "${WEBUI_SERVER_PORT}"' \
    > /app/start_webui_preact.sh && chmod +x /app/start_webui_preact.sh
ENTRYPOINT ["/app/start_webui_preact.sh"]
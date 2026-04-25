# docker/Dockerfile

# IMPORTANT: The docker images do not contain the checkpoints. You need to mount the checkpoints to the container.

# Build the image:
#   docker build \
#       --platform linux/amd64 \
#       -f docker/Dockerfile \
#       --build-arg BACKEND=[cuda, cpu] \
#       --target [webui, server, proxy, webui-preact] \
#       -t fish-speech-[webui, server, proxy, webui-preact]:[cuda, cpu] .

# Select the specific cuda version (see https://hub.docker.com/r/nvidia/cuda/)
ARG CUDA_VER=12.6.0
# Adapt the uv extra to fit the cuda version (one of [cu126, cu128, cu129])
ARG UV_EXTRA=cu126
ARG BACKEND=cuda

ARG UBUNTU_VER=24.04
ARG PY_VER=3.12
ARG UV_VERSION=0.8.15

# Create non-root user early for security
ARG USERNAME=fish
ARG USER_UID=1000
ARG USER_GID=1000

##############################################################
# Base stage per backend
##############################################################

# --- CUDA (x86_64) ---
FROM nvidia/cuda:${CUDA_VER}-cudnn-runtime-ubuntu${UBUNTU_VER} AS base-cuda
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies in a single layer with cleanup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        git \
        ca-certificates \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- CPU-only (portable x86_64) ---
FROM python:${PY_VER}-slim AS base-cpu
ENV UV_EXTRA=cpu

# Install system dependencies in a single layer with cleanup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


##############################################################
# UV stage
##############################################################

ARG UV_VERSION
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

##############################################################
# Shared app base stage
##############################################################

FROM base-${BACKEND} AS app-base

ARG PY_VER
ARG BACKEND
ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG UV_VERSION
ARG UV_EXTRA

ENV BACKEND=${BACKEND} \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for audio processing
ARG DEPENDENCIES=" \
    libsox-dev \
    build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg"

# Install system dependencies with caching and cleanup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get install -y --no-install-recommends ${DEPENDENCIES} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install specific uv version
COPY --from=uv-bin /uv /uvx /bin/

# Create non-root user (or use existing user)
RUN set -eux; \
    if getent group ${USER_GID} >/dev/null 2>&1; then \
        echo "Group ${USER_GID} already exists"; \
    else \
        groupadd -g ${USER_GID} ${USERNAME}; \
    fi; \
    if id -u ${USER_UID} >/dev/null 2>&1; then \
        echo "User ${USER_UID} already exists, using existing user"; \
        EXISTING_USER=$(id -un ${USER_UID}); \
        mkdir -p /app /home/${EXISTING_USER}/.cache; \
        chown -R ${USER_UID}:${USER_GID} /app /home/${EXISTING_USER}/.cache; \
    else \
        useradd -m -u ${USER_UID} -g ${USER_GID} ${USERNAME}; \
        mkdir -p /app /home/${USERNAME}/.cache; \
        chown -R ${USERNAME}:${USERNAME} /app /home/${USERNAME}/.cache; \
    fi

# Create references directory with proper permissions for the non-root user
RUN mkdir -p /app/references \
    && chown -R ${USER_UID}:${USER_GID} /app/references \
    && chmod 755 /app/references

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY --chown=${USER_UID}:${USER_GID} pyproject.toml uv.lock README.md ./

# Switch to non-root user for package installation
USER ${USER_UID}:${USER_GID}

# Install Python dependencies (cacheable by lockfiles)
RUN --mount=type=cache,target=/tmp/uv-cache,uid=${USER_UID},gid=${USER_GID} \
    uv python pin ${PY_VER} \
    && uv sync --extra ${UV_EXTRA} --frozen --no-install-project

# Copy application code
COPY --chown=${USER_UID}:${USER_GID} . .

# Install the local package after copying source code
RUN uv sync --extra ${UV_EXTRA} --frozen

# Create common entrypoint script
RUN printf '%s\n' \
    '#!/bin/bash' \
    'set -euo pipefail' \
    '' \
    '# Set user info from build args' \
    "USER_UID=${USER_UID}" \
    "USER_GID=${USER_GID}" \
    '' \
    '# Logging function' \
    'log() { echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*" >&2; }' \
    '' \
    '# Validate environment' \
    'validate_env() {' \
    '    if [ ! -d "/app/checkpoints" ]; then' \
    '        log "WARNING: /app/checkpoints directory not found. Please mount your checkpoints."' \
    '    fi' \
    '    if [ ! -d "/app/references" ]; then' \
    '        log "WARNING: /app/references directory not found. Please mount your references."' \
    '    else' \
    '        if [ ! -w "/app/references" ]; then' \
    '            log "ERROR: Cannot write to /app/references directory. Please ensure UID ${USER_UID} has permissions."' \
    '            exit 1' \
    '        fi' \
    '    fi' \
    '}' \
    '' \
    '# Build device arguments' \
    'build_device_args() {' \
    '    if [ "${BACKEND:-}" = "cpu" ]; then' \
    '        echo "--device cpu"' \
    '    fi' \
    '}' \
    '' \
    '# Build compile arguments' \
    'build_compile_args() {' \
    '    if [ "${1:-}" = "compile" ] || [ "${COMPILE:-}" = "1" ] || [ "${COMPILE:-}" = "true" ]; then' \
    '        echo "--compile"' \
    '        shift' \
    '    fi' \
    '    echo "$@"' \
    '}' \
    > /app/common.sh && chmod +x /app/common.sh

##############################################################
# App stages
##############################################################

# Gradio WebUI
FROM app-base AS webui
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ARG GRADIO_SERVER_NAME="0.0.0.0"
ARG GRADIO_SERVER_PORT=7860
EXPOSE ${GRADIO_SERVER_PORT}
ENV GRADIO_SERVER_NAME=${GRADIO_SERVER_NAME}
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

RUN printf '%s\n' \
    '#!/bin/bash' \
    'source /app/common.sh' \
    'log "Starting Gradio WebUI..."' \
    'validate_env' \
    'DEVICE_ARGS=$(build_device_args)' \
    'COMPILE_ARGS=$(build_compile_args "$@")' \
    'exec uv run tools/run_webui.py --llama-checkpoint-path "${LLAMA_CHECKPOINT_PATH:-checkpoints/fs-1.2-int8-s2-pro-int8}" --decoder-checkpoint-path "${DECODER_CHECKPOINT_PATH:-checkpoints/s2-pro/codec.pth}" --decoder-config-name "${DECODER_CONFIG_NAME:-modded_dac_vq}" ${DEVICE_ARGS} ${COMPILE_ARGS}' \
    > /app/start_webui.sh && chmod +x /app/start_webui.sh

ENTRYPOINT ["/app/start_webui.sh"]

# API Server
FROM app-base AS server
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ARG API_SERVER_NAME="0.0.0.0"
ARG API_SERVER_PORT=8080
EXPOSE ${API_SERVER_PORT}
ENV API_SERVER_NAME=${API_SERVER_NAME}
ENV API_SERVER_PORT=${API_SERVER_PORT}

RUN printf '%s\n' \
    '#!/bin/bash' \
    'source /app/common.sh' \
    'log "Starting API Server..."' \
    'validate_env' \
    'DEVICE_ARGS=$(build_device_args)' \
    'COMPILE_ARGS=$(build_compile_args "$@")' \
    'exec uv run python -m fish_speech_server.app --listen "${API_SERVER_NAME}:${API_SERVER_PORT}" --llama-checkpoint-path "${LLAMA_CHECKPOINT_PATH:-checkpoints/fs-1.2-int8-s2-pro-int8}" --decoder-checkpoint-path "${DECODER_CHECKPOINT_PATH:-checkpoints/s2-pro/codec.pth}" --decoder-config-name "${DECODER_CONFIG_NAME:-modded_dac_vq}" ${DEVICE_ARGS} ${COMPILE_ARGS}' \
    > /app/start_server.sh && chmod +x /app/start_server.sh

ENTRYPOINT ["/app/start_server.sh"]

# Proxy Server
FROM app-base AS proxy
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ARG PROXY_SERVER_NAME="0.0.0.0"
ARG PROXY_SERVER_PORT=9000
EXPOSE ${PROXY_SERVER_PORT}
ENV PROXY_SERVER_NAME=${PROXY_SERVER_NAME}
ENV PROXY_SERVER_PORT=${PROXY_SERVER_PORT}

RUN printf '%s\n' \
    '#!/bin/bash' \
    'source /app/common.sh' \
    'log "Starting PCM Proxy..."' \
    'exec uv run python -m uvicorn fish_speech_server.proxy.pcm:app --host "${PROXY_SERVER_NAME}" --port "${PROXY_SERVER_PORT}"' \
    > /app/start_proxy.sh && chmod +x /app/start_proxy.sh

ENTRYPOINT ["/app/start_proxy.sh"]

# Preact Web UI
FROM app-base AS webui-preact
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ARG WEBUI_SERVER_NAME="0.0.0.0"
ARG WEBUI_SERVER_PORT=9001
EXPOSE ${WEBUI_SERVER_PORT}
ENV WEBUI_SERVER_NAME=${WEBUI_SERVER_NAME}
ENV WEBUI_SERVER_PORT=${WEBUI_SERVER_PORT}

RUN printf '%s\n' \
    '#!/bin/bash' \
    'source /app/common.sh' \
    'log "Starting Web UI (Preact)..."' \
    'exec uv run python -m fish_speech_web_ui.server --host "${WEBUI_SERVER_NAME}" --port "${WEBUI_SERVER_PORT}"' \
    > /app/start_webui_preact.sh && chmod +x /app/start_webui_preact.sh

ENTRYPOINT ["/app/start_webui_preact.sh"]

# Development stage
FROM app-base AS dev
USER root
RUN apt-get update && apt-get install -y --no-install-recommends vim htop strace gdb && apt-get clean && rm -rf /var/lib/apt/lists/*
USER ${USER_UID}:${USER_GID}
RUN uv sync --extra ${UV_EXTRA} --dev
ENTRYPOINT ["/bin/bash"]
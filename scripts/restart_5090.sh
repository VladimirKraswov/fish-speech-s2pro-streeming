#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONTAINER="${CONTAINER:-fish-speech-5090}"

CONTAINER="$CONTAINER" DOCKER_USE_SUDO="${DOCKER_USE_SUDO:-0}" bash "$REPO_ROOT/scripts/down_5090.sh"
sleep 2
CONTAINER="$CONTAINER" DOCKER_USE_SUDO="${DOCKER_USE_SUDO:-0}" bash "$REPO_ROOT/scripts/up_5090.sh"
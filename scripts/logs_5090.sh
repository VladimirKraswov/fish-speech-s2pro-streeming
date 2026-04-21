#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs
touch logs/model.log logs/proxy.log

tail -n 200 -F logs/model.log logs/proxy.log
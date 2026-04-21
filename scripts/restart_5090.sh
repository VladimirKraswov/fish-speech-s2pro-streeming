#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

bash "$REPO_ROOT/scripts/down_5090.sh"
sleep 2
bash "$REPO_ROOT/scripts/up_5090.sh"
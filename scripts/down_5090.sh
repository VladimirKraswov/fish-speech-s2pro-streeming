#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib_5090.sh"

cd "$REPO_ROOT"

echo "Stopping all services via docker compose..."
docker_compose_cmd --profile full-stack down --remove-orphans

# Also clean up any legacy PID files
rm -f "$REPO_ROOT/run"/*.pid

echo "All services stopped and containers removed"

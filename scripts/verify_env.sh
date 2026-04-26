#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. Determine environment
DOCKER_CONTAINER="fish-speech-server"
IS_DOCKER=false

if docker ps --filter "name=$DOCKER_CONTAINER" --filter "status=running" --format "{{.Names}}" | grep -q "$DOCKER_CONTAINER"; then
    echo "--- Found running Docker container: $DOCKER_CONTAINER ---"
    IS_DOCKER=true
    EXEC_CMD="docker exec -it $DOCKER_CONTAINER"
else
    echo "--- No running Docker container found, checking local environment ---"
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo "Active virtual environment detected: $VIRTUAL_ENV"
        EXEC_CMD="python3"
    elif [[ -d ".venv" ]]; then
        echo "Local .venv directory found."
        EXEC_CMD=".venv/bin/python"
    elif [[ -d "venv" ]]; then
        echo "Local venv directory found."
        EXEC_CMD="venv/bin/python"
    else
        echo "No virtual environment found. Creating one..."
        if command -v uv &> /dev/null; then
            uv venv .venv
            EXEC_CMD=".venv/bin/python"
        else
            python3 -m venv .venv
            EXEC_CMD=".venv/bin/python"
        fi
    fi
fi

# 2. Upgrade pip
echo "Checking/Upgrading pip..."
if [ "$IS_DOCKER" = true ]; then
    $EXEC_CMD pip install --upgrade pip
else
    $EXEC_CMD -m pip install --upgrade pip 2>/dev/null || $EXEC_CMD -m uv pip install pip 2>/dev/null || true
    $EXEC_CMD -m pip install --upgrade pip
fi

# 3. Detect UV_EXTRA and install if needed
if [[ -z "${UV_EXTRA:-}" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        UV_EXTRA="cu129"
    else
        UV_EXTRA="cpu"
    fi
fi

echo "Environment: UV_EXTRA=$UV_EXTRA"

# 4. Compare and report
echo "--- Dependency Report ---"
cat > "$ROOT/scripts/report_deps.py" <<EOF
import json
import subprocess
import sys
import re
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import pip._vendor.tomli as tomllib
    except ImportError:
        # Fallback if tomllib is not available
        tomllib = None

def get_installed_packages():
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], capture_output=True, text=True)
        if result.returncode != 0:
            return {}
        packages = json.loads(result.stdout)
        return {pkg["name"].lower().replace('_', '-'): pkg["version"] for pkg in packages}
    except Exception:
        return {}

def parse_pyproject():
    path = Path("pyproject.toml")
    if not path.exists():
        return [], {}

    if tomllib is None:
        # Very crude fallback parsing if no toml parser is available
        with open(path, "r") as f:
            content = f.read()
            deps = re.findall(r'"([^"]+)"', re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL).group(1))
            return [d.strip() for d in deps if d.strip()], {}

    with open(path, "rb") as f:
        data = tomllib.load(f)

    deps = data.get("project", {}).get("dependencies", [])
    return deps, {}

def check_dependencies():
    installed = get_installed_packages()
    deps, _ = parse_pyproject()

    req_re = re.compile(r"^([a-zA-Z0-9\-_]+)([<>=!~\s].*)?$")

    missing = []

    for dep in deps:
        # Handle cases like "name @ url"
        name_part = dep.split('@')[0].strip()
        match = req_re.match(name_part)
        if not match:
            continue
        name = match.group(1).lower().replace('_', '-')

        if name not in installed:
            missing.append(dep)

    if not missing:
        print("✅ Все зависимости установлены")
    else:
        print("Отсутствующие или не полностью установленные пакеты:")
        for m in missing:
            print(f"  - {m}")
        print(f"\nКоманда для установки: pip install -e .[\$1]")

if __name__ == "__main__":
    check_dependencies()
EOF

if [ "$IS_DOCKER" = true ]; then
    docker cp "$ROOT/scripts/report_deps.py" "$DOCKER_CONTAINER:/tmp/report_deps.py"
    docker exec "$DOCKER_CONTAINER" python3 /tmp/report_deps.py "$UV_EXTRA"
else
    $EXEC_CMD "$ROOT/scripts/report_deps.py" "$UV_EXTRA"
fi

rm "$ROOT/scripts/report_deps.py"

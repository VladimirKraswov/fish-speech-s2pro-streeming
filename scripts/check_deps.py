import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import pip._vendor.tomli as tomllib

def get_installed_packages():
    result = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"], capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    packages = json.loads(result.stdout)
    return {pkg["name"].lower(): pkg["version"] for pkg in packages}

def parse_pyproject():
    path = Path("pyproject.toml")
    if not path.exists():
        return [], {}

    with open(path, "rb") as f:
        data = tomllib.load(f)

    deps = data.get("project", {}).get("dependencies", [])
    optional_deps = data.get("project", {}).get("optional-dependencies", {})

    return deps, optional_deps

def check_dependencies():
    installed = get_installed_packages()
    deps, optional = parse_pyproject()

    # Simple parser for "name>=version" or "name==version"
    import re
    req_re = re.compile(r"^([a-zA-Z0-9\-_]+)([<>=!~\s].*)?$")

    missing = []
    mismatched = []

    # Check main dependencies
    for dep in deps:
        match = req_re.match(dep)
        if not match:
            continue
        name = match.group(1).lower()
        spec = match.group(2) or ""

        if name not in installed:
            missing.append(dep)
        # Version checking is complex with specifiers, but we can do simple ones
        # For now, let's just report missing.

    # Also check loralib since we added it
    if "loralib" not in installed:
        missing.append("loralib")

    if not missing and not mismatched:
        print("✅ Все зависимости установлены")
    else:
        if missing:
            print("Отсутствующие пакеты:")
            for m in missing:
                print(f"  - {m}")
            print(f"\nКоманда для установки: pip install {' '.join(missing)}")

        if mismatched:
            print("\nНесоответствие версий:")
            for name, desired, actual in mismatched:
                print(f"  - {name}: желаемая {desired}, фактическая {actual}")
            print("\nПредложение: обновите пакеты до указанных версий.")

if __name__ == "__main__":
    check_dependencies()

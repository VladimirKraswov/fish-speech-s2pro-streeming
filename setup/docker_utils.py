# setup/docker_utils.py

"""Обёртки для docker и docker compose."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import shlex

from .config import get_docker_sudo

def run(cmd: List[str], capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Выполнить команду с префиксом sudo, если нужно."""
    if get_docker_sudo() and cmd[0] == "docker":
        cmd = ["sudo"] + cmd
    return subprocess.run(cmd, capture_output=capture, text=True, check=check)

def compose(*args: str) -> subprocess.CompletedProcess:
    """docker compose ..."""
    return run(["docker", "compose"] + list(args))

def build_images(no_cache: bool = True) -> None:
    """Сборка образов server, proxy, webui."""
    cmd = ["build"]
    if no_cache:
        cmd.append("--no-cache")
    cmd.extend(["server", "proxy", "webui"])
    compose(*cmd)

def up_detached() -> None:
    compose("up", "-d")

def down() -> None:
    compose("down", "--remove-orphans")

def logs(service: Optional[str] = None, tail: int = 50) -> str:
    """Получить логи сервисов."""
    cmd = ["logs", "--tail", str(tail)]
    if service:
        cmd.append(service)
    result = run(["docker", "compose"] + cmd, capture=True, check=False)
    return result.stdout + result.stderr

def generate_lock_file(ui_dir: Path) -> bool:
    """Запустить npm install внутри контейнера, чтобы создать package-lock.json.
    Возвращает True при успехе, False при ошибке (например, нет сети).
    """
    ui_dir = ui_dir.resolve()
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{ui_dir}:/app",
        "-w", "/app",
        "node:22-slim", "npm", "install"
    ]
    try:
        run(cmd)
        return True
    except subprocess.CalledProcessError as e:
        # Не ругаемся, просто возвращаем False
        return False
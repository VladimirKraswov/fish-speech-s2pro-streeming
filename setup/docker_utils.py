# setup/docker_utils.py

"""Обёртки для docker и docker compose."""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Union

from .config import get_docker_sudo


def run(cmd: List[str], capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Выполнить команду с sudo и нужными переменными окружения."""
    if get_docker_sudo() and cmd[0] == "docker":
        # -E сохраняет текущее окружение пользователя
        cmd = ["sudo", "-E"] + cmd

    # Переменные для сборки Docker (если не заданы в системе)
    env = os.environ.copy()
    env.setdefault("UV_EXTRA", "cu129")
    env.setdefault("CUDA_VER", "12.9.0")

    return subprocess.run(cmd, capture_output=capture, text=True, check=check, env=env)


def compose(*args: str) -> subprocess.CompletedProcess:
    """docker compose ..."""
    return run(["docker", "compose"] + list(args))


def build_images(services: Union[str, List[str]], no_cache: bool = True) -> None:
    """Сборка одного или нескольких сервисов."""
    if isinstance(services, str):
        services = [services]
    cmd = ["build"]
    if no_cache:
        cmd.append("--no-cache")
    cmd.extend(services)
    compose(*cmd)


def try_build_service(service: str, no_cache: bool = True) -> bool:
    """Попытаться собрать один сервис. Возвращает True при успехе."""
    try:
        build_images([service], no_cache=no_cache)
        return True
    except subprocess.CalledProcessError:
        return False


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
    """Запустить npm install внутри контейнера, чтобы создать package-lock.json."""
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
    except subprocess.CalledProcessError:
        return False
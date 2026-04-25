# setup/ui_utils.py

"""Вспомогательные функции для WebUI."""

import subprocess
import shutil
from pathlib import Path
from . import docker_utils
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def ensure_package_lock():
    """Если нет package-lock.json, создать его.
    Порядок действий:
      1. Попробовать через Docker (node:22-slim).
      2. Если не вышло – попробовать локальный npm (если установлен).
      3. Если и это не удалось – объяснить проблему и завершить установку.
    """
    ui_dir = PROJECT_ROOT / "fish_speech_web_ui" / "ui"
    lock_file = ui_dir / "package-lock.json"

    if lock_file.exists():
        console.print("[green]✓ package-lock.json уже существует[/]")
        return

    console.print("[cyan]Генерация package-lock.json...[/]")

    # Способ 1: через Docker
    if docker_utils.generate_lock_file(ui_dir):
        console.print("[green]✓ package-lock.json создан (Docker)[/]")
        return
    else:
        console.print("[yellow]⚠ Генерация через Docker не удалась (проблемы с сетью или Docker Hub)[/]")

    # Способ 2: локальный npm
    if shutil.which("npm"):
        try:
            subprocess.run(["npm", "install"], cwd=str(ui_dir), check=True, capture_output=True)
            console.print("[green]✓ package-lock.json создан (локальный npm)[/]")
            return
        except subprocess.CalledProcessError:
            console.print("[yellow]⚠ Локальный npm также не сработал[/]")
    else:
        console.print("[yellow]⚠ Локальный npm не найден[/]")

    # Всё плохо
    console.print(
        "\n[bold red]Ошибка:[/] не удалось создать package-lock.json.\n"
        "Это необходимо для сборки WebUI.\n"
        "Причина: Docker не может загрузить образ node:22-slim (нет доступа к Docker Hub) "
        "и локальный npm не установлен.\n"
        "[bold]Варианты решения:[/]\n"
        "  1. Проверьте интернет-соединение и права Docker на загрузку образов.\n"
        "  2. Установите Node.js и npm локально (sudo apt install nodejs npm).\n"
        "  3. Вручную выполните в папке проекта:\n"
        f"       cd {ui_dir} && npm install\n"
        "     после чего повторите установку.\n"
    )
    import sys
    sys.exit(1)
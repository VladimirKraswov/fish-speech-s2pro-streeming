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
    Порядок:
      1. Попробовать через Docker (node:22-slim).
      2. Если не вышло – локальный npm.
      3. Иначе завершить установку с сообщением.
    """
    ui_dir = PROJECT_ROOT / "fish_speech_web_ui" / "ui"
    lock_file = ui_dir / "package-lock.json"

    if lock_file.exists():
        console.print("[green]✓ package-lock.json уже существует[/]")
        return

    console.print("[cyan]Генерация package-lock.json...[/]")

    # Способ 1: Docker
    if docker_utils.generate_lock_file(ui_dir):
        console.print("[green]✓ package-lock.json создан (Docker)[/]")
        return
    else:
        console.print("[yellow]⚠ Генерация через Docker не удалась[/]")

    # Способ 2: локальный npm
    if shutil.which("npm"):
        try:
            subprocess.run(["npm", "install"], cwd=str(ui_dir), check=True, capture_output=True)
            console.print("[green]✓ package-lock.json создан (локальный npm)[/]")
            return
        except subprocess.CalledProcessError:
            console.print("[yellow]⚠ Локальный npm не сработал[/]")
    else:
        console.print("[yellow]⚠ Локальный npm не найден[/]")

    console.print(
        "\n[bold red]Ошибка:[/] не удалось создать package-lock.json.\n"
        "Это необходимо для сборки WebUI.\n"
        "Варианты решения:\n"
        "  1. Проверьте интернет и права Docker.\n"
        "  2. Установите Node.js локально: sudo apt install nodejs npm.\n"
        f"  3. Вручную: cd {ui_dir} && npm install\n"
    )
    sys.exit(1)
"""Вспомогательные функции для WebUI."""

from pathlib import Path
from . import docker_utils
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def ensure_package_lock():
    """Если нет package-lock.json, создать его через npm install в контейнере."""
    ui_dir = PROJECT_ROOT / "fish_speech_web_ui" / "ui"
    lock_file = ui_dir / "package-lock.json"
    if lock_file.exists():
        console.print("[green]✓ package-lock.json уже существует[/]")
        return
    console.print("[cyan]Генерация package-lock.json...[/]")
    docker_utils.generate_lock_file(ui_dir)
    console.print("[green]✓ package-lock.json создан[/]")
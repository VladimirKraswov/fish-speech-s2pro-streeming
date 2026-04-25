#!/usr/bin/env python3
"""
Fish Speech Installer – полный цикл управления сервисами.
"""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from .config import get_config_value
from .docker_utils import build_images, try_build_service, up_detached, down, logs
from .model_manager import download_models_parallel
from .health import wait_healthy
from .ui_utils import ensure_package_lock

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_URL = f"http://127.0.0.1:{get_config_value('network.server.port', 8080)}/v1/health"
WEBUI_URL = "http://127.0.0.1:9001/health"
PROXY_URL = f"http://127.0.0.1:{get_config_value('network.proxy.port', 9000)}/health"


def print_banner(text: str):
    console.print(Panel(text, style="bold magenta"))


def check_prereqs():
    """Проверка Docker и nvidia-smi."""
    import subprocess
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
        console.print("[green]✓ Docker доступен[/]")
    except Exception:
        console.print("[red]❌ Docker не работает[/]")
        sys.exit(1)

    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        console.print("[green]✓ GPU обнаружен[/]")
    except Exception:
        console.print("[yellow]⚠ nvidia-smi не найден (GPU может быть недоступен)[/]")


def step(message: str):
    console.print(f"\n[bold cyan]▶ {message}[/]")


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        show_menu()


@cli.command()
def install():
    """Полная установка: окружение, модели, сборка, запуск."""
    print_banner("🐟 УСТАНОВКА FISH SPEECH")

    step("1. Проверка окружения")
    check_prereqs()

    step("2. Подготовка WebUI (lock-файл)")
    ensure_package_lock()

    step("3. Загрузка моделей")
    download_models_parallel()

    step("4. Сборка Docker-образов")
    # Обязательные сервисы
    build_images(["server", "proxy"], no_cache=True)
    console.print("[green]✓ Сервер и прокси собраны[/]")

    # WebUI — опционально, не прерывает установку при ошибке
    if try_build_service("webui", no_cache=True):
        console.print("[green]✓ WebUI собран[/]")
    else:
        console.print("[yellow]⚠ Сборка WebUI не удалась (возможные проблемы с исходным кодом)[/]")
        console.print("   Сервер и прокси будут работать. WebUI можно собрать позже.")

    step("5. Запуск сервисов")
    up_detached()

    step("6. Ожидание API-сервера")
    if wait_healthy(API_URL, timeout=180):
        console.print("[green]✓ API готов[/]")
    else:
        console.print("[red]❌ Таймаут API[/]")
        sys.exit(1)

    console.print("\n[bold green]Установка завершена![/]")
    console.print(f"API: http://localhost:{get_config_value('network.server.port', 8080)}")
    console.print(f"WebUI: http://localhost:9001 (если был собран)")


@cli.command()
def run():
    """Быстрый запуск сервисов (без пересборки)."""
    print_banner("🚀 ЗАПУСК сервисов")
    up_detached()
    wait_healthy(API_URL, timeout=60)
    console.print("[green]Сервисы запущены[/]")


@cli.command()
def restart():
    """Перезапуск."""
    print_banner("🔄 ПЕРЕЗАПУСК")
    down()
    run()


@cli.command()
def stop():
    """Остановка."""
    print_banner("🛑 ОСТАНОВКА")
    down()
    console.print("[green]Контейнеры остановлены[/]")


@cli.command()
def clear():
    """Полная очистка, кроме моделей и references."""
    print_banner("🧹 ПОЛНАЯ ОЧИСТКА")
    if not Confirm.ask("Удалить контейнеры, образы, кэш? Модели и references не пострадают", default=False):
        return
    down()
    import subprocess
    subprocess.run(["docker", "images", "-q", "fish-speech-*", "|", "xargs", "-r", "docker", "rmi", "-f"], shell=True)
    subprocess.run(["docker", "builder", "prune", "-a", "-f"])
    console.print("[green]Очистка завершена[/]")


def show_menu():
    """Интерактивное меню с пунктами."""
    table = Table(title="Доступные действия")
    table.add_column("Ключ", style="cyan")
    table.add_column("Описание")
    table.add_row("1", "install   – полная установка")
    table.add_row("2", "run       – запуск")
    table.add_row("3", "restart   – перезапуск")
    table.add_row("4", "stop      – остановка")
    table.add_row("5", "clear     – очистка")
    table.add_row("0", "выход")
    console.print(table)

    choice = Prompt.ask("Выберите действие", choices=["1", "2", "3", "4", "5", "0"], default="0")
    if choice == "1":
        install()
    elif choice == "2":
        run()
    elif choice == "3":
        restart()
    elif choice == "4":
        stop()
    elif choice == "5":
        clear()
    else:
        sys.exit(0)


if __name__ == "__main__":
    cli()
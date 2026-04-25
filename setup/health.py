"""Проверка здоровья API."""

import time
import requests
from rich.console import Console

console = Console()

def wait_healthy(url: str, timeout: int = 180) -> bool:
    """Ожидание, пока GET url не вернёт 200."""
    start = time.time()
    with console.status(f"[cyan]Ожидаю {url}...[/]") as status:
        while time.time() - start < timeout:
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
    return False
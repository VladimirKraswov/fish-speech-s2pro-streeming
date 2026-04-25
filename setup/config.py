"""Загрузка конфигурации из runtime.json и переменных окружения."""

import json
import os
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_CFG_PATH = PROJECT_ROOT / "config" / "runtime.json"

def load_runtime_config() -> Dict[str, Any]:
    if not RUNTIME_CFG_PATH.exists():
        raise FileNotFoundError(f"Не найден {RUNTIME_CFG_PATH}")
    with open(RUNTIME_CFG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_config_value(dotted_path: str, default=None) -> Any:
    """Получить вложенное значение, например 'proxy.tts.reference_id'."""
    cfg = load_runtime_config()
    parts = dotted_path.split(".")
    cur = cfg
    for p in parts:
        cur = cur.get(p)
        if cur is None:
            return default
    return cur

def get_docker_sudo() -> bool:
    return os.environ.get("DOCKER_USE_SUDO", "1") == "1"
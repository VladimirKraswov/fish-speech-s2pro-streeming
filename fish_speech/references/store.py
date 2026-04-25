from __future__ import annotations

from pathlib import Path

from fish_speech.driver.config import load_runtime_config


def get_references_dir() -> Path:
    return Path(load_runtime_config().paths.references_dir)

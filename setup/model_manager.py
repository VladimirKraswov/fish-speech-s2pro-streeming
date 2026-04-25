"""Загрузка чекпоинтов Fish Speech с HuggingFace."""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from huggingface_hub import snapshot_download, hf_hub_download
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def download_llama_checkpoint():
    repo = "fishaudio/fs-1.2-int8-s2-pro-int8"
    local_dir = PROJECT_ROOT / "checkpoints/fs-1.2-int8-s2-pro-int8"
    if not local_dir.exists() or not any(local_dir.iterdir()):
        console.print("[cyan]Загрузка LLAMA чекпоинта...[/]")
        snapshot_download(repo, local_dir=str(local_dir), resume=True)
        console.print("[green]✓ LLAMA загружен[/]")

def download_dac_decoder():
    repo = "fishaudio/s2-pro"
    filename = "codec.pth"
    local_dir = PROJECT_ROOT / "checkpoints/s2-pro"
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / filename
    if not target.exists():
        console.print("[cyan]Загрузка DAC декодера...[/]")
        hf_hub_download(repo, filename=filename, local_dir=str(local_dir), resume=True)
        console.print("[green]✓ DAC декодер загружен[/]")

def download_models_parallel():
    """Загрузить обе модели одновременно."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(download_llama_checkpoint)
        pool.submit(download_dac_decoder)
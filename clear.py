# clear.py
#!/usr/bin/env python3
"""
Скрипт для вычистки проекта Fish Speech до минимальной версии,
необходимой для работы streaming API и прокси (скрипты up_5090.sh, run_proxy.sh).

Все удаляемые файлы и папки перемещаются в папку `temp` в корне проекта.
Для каждого перемещаемого текстового файла в начало добавляется комментарий
с его исходным полным путём от корня проекта перед перемещением.
Файлы, которые нужно отредактировать (pyproject.toml, __init__.py и др.),
копируются в `temp` (с добавлением пути), а затем заменяются минимальным содержимым.
Для всех оставшихся текстовых файлов (не удаляемых и не редактируемых)
также добавляется комментарий с путём (на месте).
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Set


# ----------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------
# Список относительных путей (файлов и директорий), которые нужно ПОЛНОСТЬЮ УДАЛИТЬ (переместить в temp)
DELETE_PATHS: List[str] = [
    # Директории
    "fish_speech/callbacks",
    "fish_speech/datasets",
    "fish_speech/text",
    "fish_speech/configs/lora",
    "docs",
    "tests",
    "tools/llama",
    "tools/vqgan",
    "tools/webui",
    ".vscode",
    ".github",
    "__pycache__",       # удалить все __pycache__
    # Файлы
    "fish_speech/scheduler.py",
    "fish_speech/train.py",
    "fish_speech/utils/braceexpand.py",
    "fish_speech/utils/rich_utils.py",
    "fish_speech/utils/logging_utils.py",
    "fish_speech/utils/instantiators.py",
    "fish_speech/utils/spectrogram.py",
    "fish_speech/configs/base.yaml",
    "fish_speech/configs/text2semantic_finetune.yaml",
    "inference.ipynb",
    "mkdocs.yml",
    ".readthedocs.yaml",
    ".pre-commit-config.yaml",
    "tools/run_webui.py",
    "entrypoint.sh",
    "dockerfile.dev",
    "Makefile",
    ".env",
    ".project-root",
    "pyrightconfig.json",
    "scripts/e2e_smoke.sh",
    "scripts/e2e_memory.sh",
    "scripts/preencode.sh",
    "scripts/upload_references.sh",
    "scripts/ttfa_smoke.py",
    "fish_speech/i18n.py",   # если не нужен webui
]

# Список файлов, которые нужно ОТРЕДАКТИРОВАТЬ (заменить содержимым из констант)
EDIT_PATHS: List[str] = [
    "pyproject.toml",
    "fish_speech/utils/__init__.py",
    "fish_speech/__init__.py",
]

# Минимальное содержимое для редактируемых файлов
NEW_CONTENTS = {
    "pyproject.toml": '''[project]
name = "fish-speech"
version = "2.0.0"
description = "Fish Speech (streaming-only)"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "transformers<=4.57.3",
    "hydra-core>=1.3.2",
    "omegaconf>=2.3.0",
    "einops>=0.7.0",
    "loguru>=0.6.0",
    "pydantic==2.9.2",
    "ormsgpack",
    "uvicorn>=0.30.0",
    "kui>=1.6.0",
    "cachetools",
    "safetensors",
    "tiktoken>=0.8.0",
    "click>=8.0.0",
    "pyrootutils>=1.0.4",
    "natsort>=8.4.0",
    "soundfile>=0.12.0",
    "fastapi",
    "httpx",
]

[project.optional-dependencies]
cpu = ["torch==2.8.0", "torchaudio"]
cu126 = ["torch==2.8.0", "torchaudio"]
cu128 = ["torch==2.8.0", "torchaudio"]
cu129 = ["torch==2.8.0", "torchaudio"]

[tool.uv]
conflicts = [
  [
    { extra = "cpu" },
    { extra = "cu126" },
    { extra = "cu128" },
    { extra = "cu129" },
  ],
]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", extra = "cpu" },
  { index = "pytorch-cu126", extra = "cu126" },
  { index = "pytorch-cu128", extra = "cu128" },
  { index = "pytorch-cu129", extra = "cu129" },
]
torchaudio = [
  { index = "pytorch-cpu", extra = "cpu" },
  { index = "pytorch-cu126", extra = "cu126" },
  { index = "pytorch-cu128", extra = "cu128" },
  { index = "pytorch-cu129", extra = "cu129" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu129"
url = "https://download.pytorch.org/whl/cu129"
explicit = true

[build-system]
requires = ["setuptools", "setuptools-scm"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["fish_speech", "tools"]
''',

    "fish_speech/utils/__init__.py": '''from .context import autocast_exclude_mps
from .file import get_latest_checkpoint
from .logger import RankedLogger
from .utils import extras, get_metric_value, set_seed, task_wrapper

__all__ = [
    "extras",
    "get_metric_value",
    "RankedLogger",
    "task_wrapper",
    "get_latest_checkpoint",
    "autocast_exclude_mps",
    "set_seed",
]
''',

    "fish_speech/__init__.py": "# Minimal __init__ for streaming\n",
}

# Расширения текстовых файлов, в которые можно безопасно добавить комментарий
TEXT_EXTENSIONS = {
    ".py", ".yaml", ".yml", ".sh", ".md", ".txt", ".toml",
    ".gitignore", ".dockerignore", ".env", "Dockerfile", "compose.yml", "compose.base.yml"
}
# Дополнительные имена файлов без расширения (например, Dockerfile)
SPECIFIC_FILES = {"Dockerfile", "compose.yml", "compose.base.yml", ".gitignore", ".dockerignore", ".env"}


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def is_text_file(file_path: Path) -> bool:
    """Определяет, является ли файл текстовым (по расширению или имени)."""
    if file_path.name in SPECIFIC_FILES:
        return True
    if file_path.suffix in TEXT_EXTENSIONS:
        return True
    return False


def add_path_comment(file_path: Path, root_dir: Path) -> None:
    """Добавляет комментарий с относительным путём в начало текстового файла."""
    if not is_text_file(file_path):
        return

    rel_path = file_path.relative_to(root_dir)
    comment_line = f"# {rel_path}\n"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
    except (UnicodeDecodeError, OSError):
        # Не текстовый или не читается – пропускаем
        return

    # Если уже есть такой комментарий в первой строке – пропускаем (чтобы не дублировать)
    if first_line.rstrip("\n") == comment_line.rstrip("\n"):
        return

    # Читаем всё содержимое
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Записываем комментарий + оригинальное содержимое
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(comment_line)
        f.write(content)


def move_to_temp_with_path_comment(src_rel: str, root_dir: Path, temp_dir: Path) -> None:
    """
    Перемещает файл или директорию в temp_dir, сохраняя относительную структуру.
    Для текстовых файлов перед перемещением добавляет комментарий с исходным путём.
    """
    src_path = root_dir / src_rel
    if not src_path.exists():
        print(f"⚠️  Пропуск: {src_rel} не существует")
        return

    dst_path = temp_dir / src_rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Если это файл и текстовый – добавить комментарий перед перемещением
    if src_path.is_file() and is_text_file(src_path):
        # Временно копируем, добавляем комментарий, затем перемещаем?
        # Проще: прочитать оригинал, создать новый файл в temp с комментарием, затем удалить оригинал.
        # Но по условию нужно переместить с подписью. Сделаем:
        # 1. Прочитаем оригинал
        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()
        # 2. Сформируем новый контент с комментарием
        rel_path = src_path.relative_to(root_dir)
        comment_line = f"# {rel_path}\n"
        new_content = comment_line + content
        # 3. Запишем в целевой файл (уже внутри temp)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        # 4. Удалим оригинал
        src_path.unlink()
        print(f"🗑️  Перемещён с подписью: {src_rel} -> {dst_path}")
    else:
        # Для бинарных или директорий – просто перемещаем
        shutil.move(str(src_path), str(dst_path))
        print(f"🗑️  Перемещён: {src_rel}")


def remove_empty_dirs(root_dir: Path) -> None:
    """Рекурсивно удаляет пустые директории."""
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        try:
            if not dirnames and not filenames:
                os.rmdir(dirpath)
                print(f"🧹 Удалена пустая папка: {Path(dirpath).relative_to(root_dir)}")
        except OSError:
            pass


# ----------------------------------------------------------------------
# Основная функция
# ----------------------------------------------------------------------
def main():
    # Определяем корень проекта (скрипт должен лежать в корне, либо передан аргумент)
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1]).resolve()
    else:
        root_dir = Path(__file__).parent.resolve()

    if not (root_dir / "fish_speech").exists():
        print(f"❌ Ошибка: {root_dir} не похож на корень проекта Fish Speech (нет папки fish_speech)")
        sys.exit(1)

    print(f"📁 Корень проекта: {root_dir}")

    # Создаём папку temp в корне
    temp_dir = root_dir / "temp"
    if temp_dir.exists():
        print(f"⚠️  Папка {temp_dir} уже существует. Очищаем её...")
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    print(f"📦 Создана временная папка: {temp_dir}")

    # 1. Перемещаем все удаляемые пути в temp (с добавлением подписи)
    print("\n--- 1. Перемещение ненужных файлов и папок в temp ---")
    for rel_path in DELETE_PATHS:
        # Если путь оканчивается на __pycache__, обработаем все такие папки рекурсивно
        if rel_path == "__pycache__":
            for pycache in root_dir.rglob("__pycache__"):
                rel = pycache.relative_to(root_dir)
                move_to_temp_with_path_comment(str(rel), root_dir, temp_dir)
            continue
        move_to_temp_with_path_comment(rel_path, root_dir, temp_dir)

    # 2. Редактируем выбранные файлы (с сохранением копии в temp с подписью)
    print("\n--- 2. Редактирование файлов ---")
    for rel_path in EDIT_PATHS:
        src_path = root_dir / rel_path
        if not src_path.exists():
            print(f"⚠️  Файл {rel_path} не существует, пропуск")
            continue

        # Создаём копию в temp с добавлением комментария (как при перемещении)
        dst_backup = temp_dir / rel_path
        dst_backup.parent.mkdir(parents=True, exist_ok=True)

        # Копируем оригинал в temp (добавляем комментарий)
        with open(src_path, "r", encoding="utf-8") as f:
            original_content = f.read()
        rel_for_comment = Path(rel_path)
        comment_line = f"# {rel_for_comment}\n"
        backup_content = comment_line + original_content
        with open(dst_backup, "w", encoding="utf-8") as f:
            f.write(backup_content)
        print(f"📑 Копия с подписью сохранена: {rel_path} -> {dst_backup}")

        # Заменяем содержимое оригинала минимальным содержимым
        new_content = NEW_CONTENTS.get(rel_path)
        if new_content is None:
            print(f"⚠️  Не найдено новое содержимое для {rel_path}, пропуск")
            continue

        with open(src_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"✏️  Обновлён: {rel_path}")

    # 2b. Удаляем пустые папки, которые могли образоваться после перемещения
    print("\n--- 2b. Очистка пустых директорий ---")
    remove_empty_dirs(root_dir)

    # 3. Добавляем комментарий-путь во все оставшиеся текстовые файлы (которые не были перемещены и не редактировались)
    print("\n--- 3. Добавление комментария с путём в оставшиеся файлы ---")
    # Получаем множество путей, которые были обработаны (удалены или отредактированы)
    processed = set(DELETE_PATHS) | set(EDIT_PATHS)
    # Также добавим все сработавшие __pycache__ директории (их не нужно комментировать на месте, т.к. они удалены)
    # Просто обойдём все файлы в корне, кроме temp
    for file_path in root_dir.rglob("*"):
        if file_path.is_dir():
            continue
        # Пропускаем файлы внутри temp
        if temp_dir in file_path.parents:
            continue
        # Пропускаем .git
        if ".git" in str(file_path):
            continue
        # Получаем относительный путь
        try:
            rel = file_path.relative_to(root_dir)
        except ValueError:
            continue
        # Если файл в списке удаляемых или редактируемых – он уже обработан (удалён или обновлён)
        # Но для отредактированных файлов мы уже заменили содержимое, и новое содержимое не содержит комментарий.
        # Поэтому нужно добавить комментарий и для них (после замены).
        if str(rel) in processed:
            continue
        # Для всех остальных – добавить комментарий
        add_path_comment(file_path, root_dir)
        print(f"📝 Добавлен путь: {rel}")

    print("\n✅ Готово!")
    print(f"   Все удалённые/заменённые файлы находятся в: {temp_dir}")
    print("   Проект теперь содержит только минимальный набор для streaming.")
    print("   *Внимание:* перед повторным запуском скрипта удалите папку temp или скрипт пересоздаст её заново.")


if __name__ == "__main__":
    main()
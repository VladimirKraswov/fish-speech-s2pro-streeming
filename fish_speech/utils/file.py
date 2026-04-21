import os
from pathlib import Path
from typing import Union

from loguru import logger
from natsort import natsorted

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
}


def get_latest_checkpoint(path: Path | str) -> Path | None:
    ckpt_dir = Path(path)

    if ckpt_dir.exists() is False:
        return None

    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime)
    if len(ckpts) == 0:
        return None

    return ckpts[-1]


def audio_to_bytes(file_path):
    if not file_path or not Path(file_path).exists():
        return None
    with open(file_path, "rb") as wav_file:
        wav = wav_file.read()
    return wav


def read_ref_text(ref_text):
    path = Path(ref_text)
    if path.exists() and path.is_file():
        with path.open("r", encoding="utf-8") as file:
            return file.read()
    return ref_text


def list_files(
    path: Union[Path, str],
    extensions: set[str] | None = None,
    recursive: bool = False,
    sort: bool = True,
) -> list[Path]:
    """
    List files in a directory.

    Args:
        path: Path to the directory.
        extensions: Extensions to filter. If empty or None, returns all files.
        recursive: Whether to search recursively.
        sort: Whether to sort the files.

    Returns:
        List of files.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist.")

    files: list[Path] = []

    if not extensions:
        iterator = path.rglob("*") if recursive else path.glob("*")
        files = [file for file in iterator if file.is_file()]
    else:
        for ext in extensions:
            pattern = f"*{ext}"
            iterator = path.rglob(pattern) if recursive else path.glob(pattern)
            files.extend(file for file in iterator if file.is_file())

    if sort:
        files = natsorted(files)

    return files


def load_filelist(path: Path | str) -> list[tuple[Path, str, str, str]]:
    """
    Load a Bert-VITS2 style filelist.
    """

    files = set()
    results = []
    count_duplicated, count_not_found = 0, 0

    LANGUAGE_TO_LANGUAGES = {
        "zh": ["zh", "en"],
        "jp": ["jp", "en"],
        "en": ["en"],
    }

    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            splits = line.strip().split("|", maxsplit=3)
            if len(splits) != 4:
                logger.warning(f"Invalid line: {line}")
                continue

            filename, speaker, language, text = splits
            file = Path(filename)
            language = language.strip().lower()

            if language == "ja":
                language = "jp"

            assert language in ["zh", "jp", "en"], f"Invalid language {language}"
            languages = LANGUAGE_TO_LANGUAGES[language]

            if file in files:
                logger.warning(f"Duplicated file: {file}")
                count_duplicated += 1
                continue

            if not file.exists():
                logger.warning(f"File not found: {file}")
                count_not_found += 1
                continue

            results.append((file, speaker, languages, text))

    if count_duplicated > 0:
        logger.warning(f"Total duplicated files: {count_duplicated}")

    if count_not_found > 0:
        logger.warning(f"Total files not found: {count_not_found}")

    return results
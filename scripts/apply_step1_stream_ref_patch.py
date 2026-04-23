#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_step1")
    if not backup.exists():
        shutil.copy2(path, backup)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def replace_all_or_fail(text: str, old: str, new: str, expected_count: int | None = None) -> str:
    count = text.count(old)
    if count == 0:
        raise RuntimeError(f"Не найден фрагмент для замены:\n{old}")
    if expected_count is not None and count != expected_count:
        raise RuntimeError(
            f"Ожидалось {expected_count} вхождений, найдено {count} для:\n{old}"
        )
    return text.replace(old, new)


def replace_regex_or_fail(text: str, pattern: str, repl: str, flags: int = re.MULTILINE | re.DOTALL, expected_count: int = 1) -> str:
    new_text, count = re.subn(pattern, repl, text, flags=flags)
    if count != expected_count:
        raise RuntimeError(
            f"Regex-замена сработала {count} раз(а), ожидалось {expected_count}.\nPATTERN:\n{pattern}"
        )
    return new_text


def patch_inference_engine(repo_root: Path) -> None:
    path = repo_root / "fish_speech" / "inference_engine" / "__init__.py"
    backup_file(path)
    text = read_text(path)

    old = 'getattr(req, "stream_tokens", False) or req.streaming'
    new = 'bool(getattr(req, "stream_tokens", False))'
    text = replace_all_or_fail(text, old, new, expected_count=2)

    write_text(path, text)
    print(f"OK: {path}")


def patch_views(repo_root: Path) -> None:
    path = repo_root / "tools" / "server" / "views.py"
    backup_file(path)
    text = read_text(path)

    block_to_remove = """    # Подстановка референса по умолчанию
    if req.reference_id is None:
        req.reference_id = "ref"

"""
    if block_to_remove in text:
        text = text.replace(block_to_remove, "")
    else:
        print("WARN: блок автоподстановки reference_id уже удалён или изменён")

    anchor = """        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine
        sample_rate = engine.decoder_model.sample_rate
"""
    insert = """        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        if not req.reference_id and not req.references:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="Reference is required for this test mode: provide reference_id or references",
            )

        sample_rate = engine.decoder_model.sample_rate
"""
    if 'Reference is required for this test mode' not in text:
        if anchor not in text:
            raise RuntimeError("Не найден якорь для вставки проверки референса в views.py")
        text = text.replace(anchor, insert)
    else:
        print("WARN: проверка обязательного референса уже вставлена")

    write_text(path, text)
    print(f"OK: {path}")


def patch_reference_loader(repo_root: Path) -> None:
    path = repo_root / "fish_speech" / "inference_engine" / "reference_loader.py"
    backup_file(path)
    text = read_text(path)

    load_by_id_pattern = r'''
def\ load_by_id\(
        self,
        id:\ str,
        use_cache:\ Literal\["on",\ "off"\],
    \)\ ->\ Tuple:
        \#\ Load\ the\ references\ audio\ and\ text\ by\ id\.
        \#\ Each\ reference\ can\ be:\ \(a\)\ \.wav\ \+\ \.lab\ →\ encode\ at\ load,\ or\ \(b\)\ \.codes\.pt\ \+\ \.lab\ →\ load\ pre-encoded\ \(no\ encoder\ run\)\.
        ref_folder\ =\ Path\("references"\)\ /\ id
        ref_folder\.mkdir\(parents=True,\ exist_ok=True\)
        ref_audios\ =\ list_files\(
            ref_folder,\ AUDIO_EXTENSIONS,\ recursive=True,\ sort=False
        \)
        ref_codes\ =\ list\(ref_folder\.glob\("\*\.codes\.pt"\)\)

        \#\ \.codes\.pt\ filename\ stem\ is\ "en\.codes"\ \(Path\.stem\);\ we\ want\ logical\ stem\ "en"\ for\ en\.lab\ /\ en\.codes\.pt
        def\ _stem_for_codes\(p:\ Path\)\ ->\ str:
            s\ =\ p\.stem
            return\ s\.removesuffix\("\.codes"\)\ if\ s\.endswith\("\.codes"\)\ else\ s

        stems\ =\ \{p\.stem\ for\ p\ in\ ref_audios\}\ \|\ \{_stem_for_codes\(p\)\ for\ p\ in\ ref_codes\}
        stems\ =\ sorted\(stems\)
'''

    load_by_id_repl = '''def load_by_id(
        self,
        id: str,
        use_cache: Literal["on", "off"],
    ) -> Tuple:
        ref_folder = Path("references") / id

        if not ref_folder.exists() or not ref_folder.is_dir():
            raise FileNotFoundError(f"Reference ID '{id}' does not exist")

        ref_audios = list_files(
            ref_folder, AUDIO_EXTENSIONS, recursive=True, sort=False
        )
        ref_codes = list(ref_folder.glob("*.codes.pt"))

        def _stem_for_codes(p: Path) -> str:
            s = p.stem
            return s.removesuffix(".codes") if s.endswith(".codes") else s

        stems = {p.stem for p in ref_audios} | {_stem_for_codes(p) for p in ref_codes}
        stems = sorted(stems)

        if not stems:
            raise FileNotFoundError(
                f"Reference ID '{id}' exists but contains no audio or .codes.pt files"
            )
'''

    if "Reference ID '{id}' does not exist" not in text:
        text = replace_regex_or_fail(text, load_by_id_pattern, load_by_id_repl)
    else:
        print("WARN: начало load_by_id уже пропатчено")

    old_cache_line = "            self.ref_by_id[id] = (prompt_tokens, prompt_texts)"
    new_cache_block = """            if not prompt_tokens:
                raise ValueError(
                    f"Reference ID '{id}' has no valid (audio/.codes.pt + .lab) pairs"
                )

            self.ref_by_id[id] = (prompt_tokens, prompt_texts)"""
    if "has no valid (audio/.codes.pt + .lab) pairs" not in text:
        text = replace_all_or_fail(text, old_cache_line, new_cache_block, expected_count=1)
    else:
        print("WARN: проверка prompt_tokens уже вставлена")

    list_ref_pattern = r'''
    def\ list_reference_ids\(self\)\ ->\ list\[str\]:
        """
        List\ all\ valid\ reference\ IDs\ \(subdirectory\ names\ containing\ valid\ audio\ and\ \.lab\ files\)\.

        Returns:
            list\[str\]:\ List\ of\ valid\ reference\ IDs
        """
        ref_base_path\ =\ Path\("references"\)
        if\ not\ ref_base_path\.exists\(\):
            return\ \[\]

        valid_ids\ =\ \[\]
        for\ ref_dir\ in\ ref_base_path\.iterdir\(\):
            if\ not\ ref_dir\.is_dir\(\):
                continue

            \#\ Check\ if\ directory\ contains\ at\ least\ one\ audio\ file\ and\ corresponding\ \.lab\ file
            audio_files\ =\ list_files\(
                ref_dir,\ AUDIO_EXTENSIONS,\ recursive=False,\ sort=False
            \)
            if\ not\ audio_files:
                continue

            \#\ Check\ if\ corresponding\ \.lab\ file\ exists\ for\ at\ least\ one\ audio\ file
            has_valid_pair\ =\ False
            for\ audio_file\ in\ audio_files:
                lab_file\ =\ audio_file\.with_suffix\("\.lab"\)
                if\ lab_file\.exists\(\):
                    has_valid_pair\ =\ True
                    break

            if\ has_valid_pair:
                valid_ids\.append\(ref_dir\.name\)

        return\ sorted\(valid_ids\)
'''

    list_ref_repl = '''    def list_reference_ids(self) -> list[str]:
        """
        List all valid reference IDs.
        Valid reference:
          - audio + .lab
          - or .codes.pt + .lab
        """
        ref_base_path = Path("references")
        if not ref_base_path.exists():
            return []

        valid_ids = []

        for ref_dir in ref_base_path.iterdir():
            if not ref_dir.is_dir():
                continue

            audio_files = list_files(
                ref_dir, AUDIO_EXTENSIONS, recursive=False, sort=False
            )
            code_files = list(ref_dir.glob("*.codes.pt"))

            if not audio_files and not code_files:
                continue

            def _stem_for_codes(p: Path) -> str:
                s = p.stem
                return s.removesuffix(".codes") if s.endswith(".codes") else s

            stems = {p.stem for p in audio_files} | {_stem_for_codes(p) for p in code_files}

            has_valid_pair = any((ref_dir / f"{stem}.lab").exists() for stem in stems)

            if has_valid_pair:
                valid_ids.append(ref_dir.name)

        return sorted(valid_ids)
'''

    if "Valid reference:" not in text:
        text = replace_regex_or_fail(text, list_ref_pattern, list_ref_repl)
    else:
        print("WARN: list_reference_ids уже пропатчен")

    write_text(path, text)
    print(f"OK: {path}")


def main() -> int:
    repo_root = Path.cwd()

    expected = [
        repo_root / "fish_speech",
        repo_root / "tools",
        repo_root / "scripts",
    ]
    if not all(p.exists() for p in expected):
        print("Ошибка: запускай скрипт из корня проекта.", file=sys.stderr)
        return 1

    try:
        patch_inference_engine(repo_root)
        patch_views(repo_root)
        patch_reference_loader(repo_root)
    except Exception as e:
        print(f"\nPATCH FAILED: {e}", file=sys.stderr)
        return 2

    print("\nГотово.")
    print("Бэкапы созданы рядом с файлами: *.bak_step1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# fish_speech/i18n/scan.py
import ast
import json
from collections import OrderedDict
from pathlib import Path

from loguru import logger

from .core import DEFAULT_LANGUAGE, I18N_FILE_PATH


def extract_i18n_strings(node):
    i18n_strings = []

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "i18n"
    ):
        for arg in node.args:
            if isinstance(arg, ast.Str):
                i18n_strings.append(arg.s)

    for child_node in ast.iter_child_nodes(node):
        i18n_strings.extend(extract_i18n_strings(child_node))

    return i18n_strings


def scan_i18n_strings(folders: list[str] | None = None) -> list[str]:
    strings = []
    folders = folders or ["fish_speech", "tools"]

    for folder in folders:
        for f in Path(folder).rglob("*.py"):
            code = f.read_text(encoding="utf-8")
            if "i18n(" in code:
                tree = ast.parse(code)
                i18n_strings = extract_i18n_strings(tree)
                logger.info(f"Found {len(i18n_strings)} i18n strings in {f}")
                strings.extend(i18n_strings)

    return strings


def update_standard_locale(strings: list[str]) -> OrderedDict:
    code_keys = set(strings)
    logger.info(f"Total unique: {len(code_keys)}")

    standard_file = I18N_FILE_PATH / f"{DEFAULT_LANGUAGE}.json"
    with open(standard_file, "r", encoding="utf-8") as f:
        standard_data = json.load(f, object_pairs_hook=OrderedDict)
    standard_keys = set(standard_data.keys())

    unused_keys = standard_keys - code_keys
    logger.info(f"Found {len(unused_keys)} unused keys in {standard_file}")
    for unused_key in unused_keys:
        logger.info(f"\t{unused_key}")

    missing_keys = code_keys - standard_keys
    logger.info(f"Found {len(missing_keys)} missing keys in {standard_file}")
    for missing_key in missing_keys:
        logger.info(f"\t{missing_key}")

    code_keys_dict = OrderedDict()
    for s in strings:
        code_keys_dict[s] = s

    with open(standard_file, "w", encoding="utf-8") as f:
        json.dump(code_keys_dict, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")

    logger.info(f"Updated {standard_file}")
    return code_keys_dict


def sync_locale_files(standard_data: OrderedDict) -> None:
    languages = [
        f for f in I18N_FILE_PATH.glob("*.json") if f.stem != DEFAULT_LANGUAGE
    ]

    for lang_file in languages:
        with open(lang_file, "r", encoding="utf-8") as f:
            lang_data = json.load(f, object_pairs_hook=OrderedDict)

        diff = set(standard_data.keys()) - set(lang_data.keys())
        miss = set(lang_data.keys()) - set(standard_data.keys())

        for key in diff:
            lang_data[key] = "#!" + key
            logger.info(f"Added missing key: {key} to {lang_file}")

        for key in miss:
            del lang_data[key]
            logger.info(f"Del extra key: {key} from {lang_file}")

        lang_data = OrderedDict(
            sorted(
                lang_data.items(),
                key=lambda x: list(standard_data.keys()).index(x[0]),
            )
        )

        with open(lang_file, "w", encoding="utf-8") as f:
            json.dump(lang_data, f, ensure_ascii=False, indent=4, sort_keys=True)
            f.write("\n")

        logger.info(f"Updated {lang_file}")


def main() -> None:
    strings = scan_i18n_strings()
    standard_data = update_standard_locale(strings)
    sync_locale_files(standard_data)
    logger.info("Done")


if __name__ == "__main__":
    main()

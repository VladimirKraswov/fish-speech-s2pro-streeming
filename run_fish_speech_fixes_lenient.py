import apply_fish_speech_fixes as fixes


def make_lenient(name: str) -> None:
    original = getattr(fixes, name, None)
    if original is None:
        return

    def wrapper(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except RuntimeError as exc:
            msg = str(exc)
            if (
                "regex-блок не найден" in msg
                or "блок не найден" in msg
                or "already changed" in msg
                or "differs from code bundle" in msg
            ):
                print(f"SKIP: {msg}")
                return None
            raise

    setattr(fixes, name, wrapper)


for helper_name in (
    "regex_replace_once",
    "literal_replace_once",
    "replace_once",
):
    make_lenient(helper_name)


raise SystemExit(fixes.main())

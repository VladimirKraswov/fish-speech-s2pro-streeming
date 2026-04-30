import json

import pytest

from fish_speech.tools.build_prefix_cache import (
    PrefixCacheItem,
    count_words,
    load_and_validate_items,
    main,
    normalize_prefix_text,
    params_hash,
)


def test_prefix_text_helpers_are_stable():
    assert count_words("Что такое квантовая запутанность?") == 4
    assert normalize_prefix_text("  «Ёжик»,  что   такое?!  ") == "ежик, что такое"
    assert params_hash({"b": 2, "a": "ё"}) == params_hash({"a": "ё", "b": 2})


def test_load_and_validate_items_accepts_defaults_and_rejects_long_prefix(tmp_path):
    input_path = tmp_path / "prefixes.json"
    input_path.write_text(
        json.dumps([{"text": "Что такое"}]),
        encoding="utf-8",
    )

    assert load_and_validate_items(input_path, default_voice_id="voice") == [
        PrefixCacheItem(
            cache_id="voice_что_такое_v1",
            voice_id="voice",
            text="Что такое",
            normalized_text="что такое",
            word_count=2,
        )
    ]

    long_path = tmp_path / "long.json"
    long_path.write_text(
        json.dumps([{"voice_id": "voice", "text": "one two three four five six"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max is 5"):
        load_and_validate_items(long_path)


def test_prefix_cache_dry_run_validates_without_writing_artifacts(tmp_path):
    input_path = tmp_path / "prefixes.json"
    output_dir = tmp_path / "prefix_cache"
    input_path.write_text(
        json.dumps([{"voice_id": "voice", "text": "Что такое"}]),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not output_dir.exists()

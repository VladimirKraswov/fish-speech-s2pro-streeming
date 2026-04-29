import json

import pytest

from fish_speech.tools.build_cached_intros import (
    IntroItem,
    load_and_validate_items,
    main,
)


def test_load_and_validate_items_accepts_list_and_object(tmp_path):
    list_path = tmp_path / "list.json"
    list_path.write_text(
        json.dumps([{"id": "what_is", "text": "What is"}]),
        encoding="utf-8",
    )
    object_path = tmp_path / "object.json"
    object_path.write_text(
        json.dumps({"items": [{"id": "hello", "text": "Hello"}]}),
        encoding="utf-8",
    )

    assert load_and_validate_items(list_path) == [IntroItem("what_is", "What is")]
    assert load_and_validate_items(object_path) == [IntroItem("hello", "Hello")]


def test_load_and_validate_items_rejects_duplicate_and_invalid_ids(tmp_path):
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        json.dumps(
            [
                {"id": "same", "text": "one"},
                {"id": "same", "text": "two"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate intro id"):
        load_and_validate_items(duplicate_path)

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps([{"id": "bad/id", "text": "one"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid intro id"):
        load_and_validate_items(invalid_path)


def test_dry_run_validates_without_writing_artifacts(tmp_path):
    input_path = tmp_path / "intros.json"
    output_dir = tmp_path / "cached_intros"
    input_path.write_text(
        json.dumps([{"id": "what_is", "text": "What is"}]),
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

import pytest
import torch

from fish_speech.content_sequence import TextPart, VQPart
from fish_speech.conversation import Conversation, Message
from fish_speech.generation.prompt_builder import (
    _append_continuation_history,
    _prepare_external_continuation,
)


def test_external_continuation_full_policy_keeps_full_codes():
    codes = torch.arange(90).reshape(9, 10)
    selected = _prepare_external_continuation(
        ["cached intro"],
        [codes],
        expected_codebooks=9,
        policy="full",
        tail_frames=0,
        max_history_segments=1,
    )

    assert len(selected) == 1
    assert selected[0][0] == "cached intro"
    assert selected[0][1].shape == (9, 10)
    assert selected[0][1].dtype == torch.long


def test_external_continuation_tail_policy_crops_codes_only_when_requested():
    codes = torch.arange(90).reshape(9, 10)
    selected = _prepare_external_continuation(
        ["cached intro"],
        [codes],
        expected_codebooks=9,
        policy="tail_frames",
        tail_frames=4,
        max_history_segments=1,
    )

    assert selected[0][0] == "cached intro"
    assert selected[0][1].shape == (9, 4)
    assert torch.equal(selected[0][1], codes[:, -4:])


def test_append_continuation_history_uses_user_then_voice_assistant():
    conversation = Conversation(
        [
            Message(
                role="system",
                parts=[TextPart(text="convert the provided text to speech")],
            )
        ]
    )
    codes = torch.zeros((9, 5), dtype=torch.long)

    _append_continuation_history(conversation, [("cached intro", codes)])

    assert [message.role for message in conversation.messages] == [
        "system",
        "user",
        "assistant",
    ]
    user = conversation.messages[1]
    assistant = conversation.messages[2]
    assert isinstance(user.parts[0], TextPart)
    assert user.parts[0].text == "cached intro"
    assert assistant.modality == "voice"
    assert isinstance(assistant.parts[0], VQPart)
    assert torch.equal(assistant.parts[0].codes, codes)


def test_external_continuation_requires_matching_text_and_tokens():
    with pytest.raises(ValueError, match="provided together"):
        _prepare_external_continuation(
            ["cached intro"],
            None,
            expected_codebooks=9,
            policy="full",
            tail_frames=0,
            max_history_segments=1,
        )

    with pytest.raises(ValueError, match="same length"):
        _prepare_external_continuation(
            ["one", "two"],
            [torch.zeros((9, 5))],
            expected_codebooks=9,
            policy="full",
            tail_frames=0,
            max_history_segments=1,
        )

    with pytest.raises(ValueError, match="non-empty"):
        _prepare_external_continuation(
            [""],
            [torch.zeros((9, 5))],
            expected_codebooks=9,
            policy="full",
            tail_frames=0,
            max_history_segments=1,
        )

    with pytest.raises(ValueError, match="codebook count mismatch"):
        _prepare_external_continuation(
            ["cached intro"],
            [torch.zeros((8, 5))],
            expected_codebooks=9,
            policy="full",
            tail_frames=0,
            max_history_segments=1,
        )

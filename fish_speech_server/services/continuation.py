from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fish_speech_server.services.synthesis_context import (
        SynthesisContext,
        SynthesisTurn,
    )


def select_history_turns_for_continuation(
    context: SynthesisContext,
) -> list[SynthesisTurn]:
    """
    Selects turns from the history that have generated codes.
    Returns them in chronological order (oldest to newest).
    Since SynthesisContext already trims its history, we just filter for codes.
    """
    return [turn for turn in context.history if turn.codes is not None]


def build_continuation_debug_summary(context: SynthesisContext) -> dict[str, Any]:
    """
    Provides a diagnostic summary of what will be used for continuation.
    """
    selected = select_history_turns_for_continuation(context)
    return {
        "selected_turns": len(selected),
        "selected_chars": sum(len(t.text) for t in selected),
        "selected_code_frames": sum(t.code_frames for t in selected),
        "commit_seq_list": [t.commit_seq for t in selected],
        "has_codes_count": len(selected),
        "history_turns": len(context.history),
        "history_with_codes": context.history_with_codes_count(),
        "history_code_frames": context.history_code_frames(),
        "continuation_ready": context.history_with_codes_count() > 0,
    }

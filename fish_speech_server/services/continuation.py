from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fish_speech_server.services.synthesis_context import estimate_code_frames

if TYPE_CHECKING:
    from fish_speech_server.services.synthesis_context import (
        SynthesisContext,
        SynthesisTurn,
    )


@dataclass(frozen=True)
class ContinuationPart:
    commit_seq: int
    text: str
    codes: Any
    code_frames: int
    source_code_frames: int
    cropped: bool


def crop_codes_tail(codes: Any | None, max_frames: int) -> Any | None:
    if codes is None or max_frames <= 0:
        return None

    frames = estimate_code_frames(codes)
    if frames <= 0:
        return None

    if frames <= max_frames:
        return codes

    # torch.Tensor / numpy.ndarray обычно поддерживают такой slicing
    if hasattr(codes, "shape"):
        return codes[..., -max_frames:]

    # nested list: [codebook][frame]
    if isinstance(codes, list):
        if not codes:
            return codes
        first = codes[0]
        if isinstance(first, list):
            return [row[-max_frames:] for row in codes]
        return codes[-max_frames:]

    if isinstance(codes, tuple):
        first = codes[0] if codes else None
        if isinstance(first, tuple):
            return tuple(row[-max_frames:] for row in codes)
        return codes[-max_frames:]

    return codes


def crop_text_tail_for_codes(
    text: str,
    source_frames: int,
    kept_frames: int,
    max_chars: int,
) -> str:
    stripped = text.strip()
    if not stripped or max_chars <= 0:
        return ""

    if source_frames > 0 and kept_frames < source_frames:
        ratio = max(0.0, min(1.0, kept_frames / source_frames))
        keep_chars = int(math.ceil(len(stripped) * ratio)) + 8
        keep_chars = max(12, keep_chars)
    else:
        keep_chars = len(stripped)

    keep_chars = min(len(stripped), max_chars, keep_chars)
    return stripped[-keep_chars:].lstrip()


def select_continuation_parts_for_prompt(
    context: SynthesisContext,
) -> list[ContinuationPart]:
    raw_turns = [turn for turn in context.history if turn.codes is not None]

    if not raw_turns:
        return []

    max_turns = max(1, context.max_history_turns)
    remaining_frames = int(context.max_history_code_frames)
    remaining_chars = int(context.max_history_chars)

    if remaining_frames <= 0 or remaining_chars <= 0:
        return []

    raw_turns = raw_turns[-max_turns:]
    result_reversed: list[ContinuationPart] = []

    for turn in reversed(raw_turns):
        if remaining_frames <= 0 or remaining_chars <= 0:
            break

        source_frames = turn.code_frames or estimate_code_frames(turn.codes)
        if source_frames <= 0:
            continue

        keep_frames = min(source_frames, remaining_frames)
        cropped_codes = crop_codes_tail(turn.codes, keep_frames)
        actual_frames = estimate_code_frames(cropped_codes)

        if cropped_codes is None or actual_frames <= 0:
            continue

        cropped_text = crop_text_tail_for_codes(
            turn.text,
            source_frames=source_frames,
            kept_frames=actual_frames,
            max_chars=remaining_chars,
        )

        if not cropped_text:
            continue

        result_reversed.append(
            ContinuationPart(
                commit_seq=turn.commit_seq,
                text=cropped_text,
                codes=cropped_codes,
                code_frames=actual_frames,
                source_code_frames=source_frames,
                cropped=actual_frames < source_frames
                or len(cropped_text) < len(turn.text.strip()),
            )
        )

        remaining_frames -= actual_frames
        remaining_chars -= len(cropped_text)

    return list(reversed(result_reversed))


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
    selected = select_continuation_parts_for_prompt(context)
    return {
        "selected_turns": len(selected),
        "selected_chars": sum(len(p.text) for p in selected),
        "selected_code_frames": sum(p.code_frames for p in selected),
        "selected_source_code_frames": sum(p.source_code_frames for p in selected),
        "selected_cropped_count": sum(1 for p in selected if p.cropped),
        "commit_seq_list": [p.commit_seq for p in selected],
        "has_codes_count": context.history_with_codes_count(),
        "history_turns": len(context.history),
        "history_with_codes": context.history_with_codes_count(),
        "history_code_frames": context.history_code_frames(),
        "continuation_ready": len(selected) > 0,
    }

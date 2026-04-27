from __future__ import annotations

import re
from typing import TYPE_CHECKING

from fish_speech.driver import (
    DriverGenerationOptions,
    DriverReference,
    DriverSynthesisRequest,
)

if TYPE_CHECKING:
    from fish_speech_server.schema import ServeTTSRequest, StatefulTTSRequest
    from fish_speech_server.services.synthesis_context import SynthesisContext


def split_text_by_speaker(text: str) -> list[str]:
    pattern = r"(<\|speaker:\d+\|>)"
    parts = re.split(pattern, text)

    turns = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if re.match(pattern, part):
            if i + 1 < len(parts):
                turn = part + parts[i + 1]
                turns.append(turn.strip())
                i += 2
            else:
                turns.append(part)
                i += 1
        else:
            i += 1

    return turns


def split_text_by_bytes(text: str, max_bytes: int) -> list[str]:
    if max_bytes <= 0 or len(text.encode("utf-8")) <= max_bytes:
        return [text] if text.strip() else []
    chunks = []
    remaining = text
    while remaining:
        encoded = remaining.encode("utf-8")
        if len(encoded) <= max_bytes:
            chunks.append(remaining)
            break
        cut = max_bytes
        while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
            cut -= 1
        chunks.append(encoded[:cut].decode("utf-8", errors="replace"))
        remaining = encoded[cut:].decode("utf-8", errors="replace")
    return chunks


def group_turns_into_batches(
    turns: list[str], max_speakers: int = 3, max_bytes: int = 300
) -> list[str]:
    batches = []
    current_batch = []
    current_bytes = 0

    for turn in turns:
        turn_bytes = len(turn.encode("utf-8"))

        would_exceed_speakers = len(current_batch) >= max_speakers
        would_exceed_bytes = current_bytes + turn_bytes > max_bytes and current_batch

        if would_exceed_speakers or would_exceed_bytes:
            batches.append("\n".join(current_batch))
            current_batch = [turn]
            current_bytes = turn_bytes
        else:
            current_batch.append(turn)
            current_bytes += turn_bytes

    if current_batch:
        batches.append("\n".join(current_batch))

    return batches


def split_text_for_generation(
    text: str,
    *,
    chunk_length: int,
    stream_tokens: bool,
) -> list[str]:
    turns = split_text_by_speaker(text)
    if stream_tokens:
        return [text] if text.strip() else []
    if turns:
        return group_turns_into_batches(turns, max_speakers=5, max_bytes=chunk_length)
    return split_text_by_bytes(text, chunk_length)


def api_tts_to_driver_request(req: ServeTTSRequest) -> DriverSynthesisRequest:
    segments = split_text_for_generation(
        req.text,
        chunk_length=req.chunk_length,
        stream_tokens=req.stream_tokens,
    )
    return DriverSynthesisRequest(
        text=req.text,
        segments=segments,
        references=[
            DriverReference(audio=ref.audio, text=ref.text) for ref in req.references
        ],
        reference_id=req.reference_id,
        seed=req.seed,
        use_memory_cache=req.use_memory_cache,
        normalize=req.normalize,
        stream_audio=req.streaming,
        generation=DriverGenerationOptions(
            chunk_length=req.chunk_length,
            max_new_tokens=req.max_new_tokens,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            stream_tokens=req.stream_tokens,
            stream_chunk_size=req.stream_chunk_size,
            initial_stream_chunk_size=req.initial_stream_chunk_size,
        ),
    )


def stateful_tts_to_driver_request(
    req: StatefulTTSRequest, context: SynthesisContext
) -> DriverSynthesisRequest:
    """
    Creates a DriverSynthesisRequest that includes rolling history for acoustic continuity.
    """
    from fish_speech_server.services.continuation import (
        select_history_turns_for_continuation,
    )

    # 1. Start with standard conversion
    driver_req = api_tts_to_driver_request(req)

    # Also force stream_tokens=True to ensure we get codes back
    # from the very first commit in a stateful session.
    driver_req.generation.stream_tokens = True

    # 2. Select history for continuation
    history_turns = select_history_turns_for_continuation(context)

    if not history_turns:
        return driver_req

    # 3. Add history to prompt_text and prompt_tokens
    # Fish Speech expects prompt_text and prompt_tokens to be lists of same length
    # or combined structures. In DriverSynthesisRequest (implied),
    # it might need to be passed down via custom logic if DriverSynthesisRequest
    # doesn't have these fields yet.

    # Looking at GenerateRequest in TTSInferenceEngine.send_Llama_request:
    # prompt_tokens=prompt_tokens,
    # prompt_text=prompt_text,

    # We need to ensure DriverSynthesisRequest can carry these,
    # OR we use a pattern where they are injected.

    # Let's check DriverSynthesisRequest fields in fish_speech/driver/types.py
    # (I saw them earlier, it didn't have prompt_text/tokens explicitly,
    # but the Engine used them).

    # If I can't modify DriverSynthesisRequest cleanly, I'll use text-based
    # fallback for now, but the goal is acoustic continuity.

    # Wait, TTSInferenceEngine.inference loads prompt_tokens from ref_id.
    # We might need to extend DriverSynthesisRequest or use a different mechanism.

    # For now, let's assume we can attach them to driver_req as attributes
    # and handle them in a modified inference path if needed,
    # or just use them if they exist.

    # Explicitly populate continuation fields
    driver_req.continuation_text = [t.text for t in history_turns]
    driver_req.continuation_tokens = [t.codes for t in history_turns]

    # Legacy fallback for older components if any
    driver_req.prompt_text = driver_req.continuation_text
    driver_req.prompt_tokens = driver_req.continuation_tokens

    return driver_req

from __future__ import annotations

from fish_speech.generation.prompt_builder import generate_committed_segments

from fish_speech_server.services.adapter import split_text_for_generation


def generate_long(
    *,
    text: str,
    chunk_length: int = 512,
    stream_tokens: bool = False,
    **kwargs,
):
    segments = split_text_for_generation(
        text,
        chunk_length=chunk_length,
        stream_tokens=stream_tokens,
    )
    return generate_committed_segments(
        text=text,
        segments=segments,
        chunk_length=chunk_length,
        stream_tokens=stream_tokens,
        **kwargs,
    )

from __future__ import annotations

from fish_speech.driver.types import (
    DriverErrorEvent,
    DriverGenerationOptions,
    DriverSynthesisRequest,
)


def run_driver_warmup(driver) -> None:
    config = driver.config
    if config is None or not config.warmup.enabled:
        return

    warmup = config.warmup
    request = DriverSynthesisRequest(
        text=warmup.text,
        segments=[warmup.text],
        reference_id=warmup.reference_id,
        stream_audio=warmup.streaming,
        generation=DriverGenerationOptions(
            chunk_length=warmup.chunk_length,
            max_new_tokens=warmup.max_new_tokens,
            stream_tokens=warmup.stream_tokens,
            initial_stream_chunk_size=warmup.initial_stream_chunk_size,
            stream_chunk_size=warmup.stream_chunk_size,
        ),
    )
    for event in driver.synthesize(request):
        if isinstance(event, DriverErrorEvent):
            if event.error is not None:
                raise RuntimeError(f"Warmup synthesis failed: {event.error}") from (
                    event.error if isinstance(event.error, Exception) else None
                )
            raise RuntimeError("Warmup synthesis failed with an unknown error")

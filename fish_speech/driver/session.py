from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from fish_speech.driver.types import (
    DriverAudioChunkEvent,
    DriverErrorEvent,
    DriverFinalAudioEvent,
    DriverGenerationOptions,
    DriverReference,
    DriverSegmentRequest,
    DriverSynthesisRequest,
    DriverTokenChunkEvent,
)

if TYPE_CHECKING:
    from fish_speech.driver.api import FishSpeechDriver


@dataclass
class DriverSessionConfig:
    reference_id: str | None = None
    references: list[DriverReference] = field(default_factory=list)
    generation: DriverGenerationOptions = field(default_factory=DriverGenerationOptions)
    seed: int | None = None
    use_memory_cache: str = "off"
    normalize: bool = True
    stream_audio: bool = False


class DriverSession:
    def __init__(
        self,
        driver: FishSpeechDriver,
        config: DriverSessionConfig | None = None,
    ) -> None:
        self.driver = driver
        self.config = config or DriverSessionConfig()
        self.closed = False
        self.segments_submitted = 0
        self.audio_events_emitted = 0

    def synthesize_segment(
        self,
        text: str,
        options: DriverGenerationOptions | None = None,
        *,
        stream_audio: bool | None = None,
    ) -> Iterator[
        DriverAudioChunkEvent
        | DriverFinalAudioEvent
        | DriverTokenChunkEvent
        | DriverErrorEvent
    ]:
        request = DriverSegmentRequest(
            text=text,
            options=options,
            stream_audio=stream_audio,
        )
        yield from self.synthesize_request(self._segment_to_synthesis_request(request))

    def synthesize_request(
        self,
        request: DriverSynthesisRequest,
    ) -> Iterator[
        DriverAudioChunkEvent
        | DriverFinalAudioEvent
        | DriverTokenChunkEvent
        | DriverErrorEvent
    ]:
        if self.closed:
            raise RuntimeError("driver synthesis session is closed")

        self.segments_submitted += len(request.committed_segments())
        for result in self.driver.engine.inference(request):
            if result.code == "tokens":
                yield DriverTokenChunkEvent(codes=result.tokens)
            elif result.code == "segment" and isinstance(result.audio, tuple):
                self.audio_events_emitted += 1
                sample_rate, audio = result.audio
                yield DriverAudioChunkEvent(
                    sample_rate=sample_rate,
                    audio=audio,
                    segment_index=self.audio_events_emitted,
                )
            elif result.code == "final" and isinstance(result.audio, tuple):
                sample_rate, audio = result.audio
                yield DriverFinalAudioEvent(sample_rate=sample_rate, audio=audio)
            elif result.code == "error":
                yield DriverErrorEvent(error=result.error)

    def finish(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True

    def _segment_to_synthesis_request(
        self,
        request: DriverSegmentRequest,
    ) -> DriverSynthesisRequest:
        return DriverSynthesisRequest(
            text=request.text,
            segments=[request.text],
            references=list(self.config.references),
            reference_id=self.config.reference_id,
            seed=self.config.seed,
            use_memory_cache=self.config.use_memory_cache,  # type: ignore[arg-type]
            normalize=self.config.normalize,
            stream_audio=(
                self.config.stream_audio
                if request.stream_audio is None
                else request.stream_audio
            ),
            generation=request.options or self.config.generation,
        )


SynthesisContext = DriverSession

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DriverReference:
    audio: bytes
    text: str


@dataclass
class DriverGenerationOptions:
    chunk_length: int = 200
    max_new_tokens: int = 512
    top_p: float = 0.8
    repetition_penalty: float = 1.1
    temperature: float = 0.8
    stream_tokens: bool = False
    stream_chunk_size: int = 8
    initial_stream_chunk_size: int = 10


@dataclass
class DriverSynthesisRequest:
    text: str = ""
    segments: list[str] = field(default_factory=list)
    references: list[DriverReference] = field(default_factory=list)
    reference_id: str | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    normalize: bool = True
    stream_audio: bool = False
    generation: DriverGenerationOptions = field(default_factory=DriverGenerationOptions)

    def committed_segments(self) -> list[str]:
        if self.segments:
            return [segment for segment in self.segments if segment.strip()]
        return [self.text] if self.text.strip() else []

    @property
    def chunk_length(self) -> int:
        return self.generation.chunk_length

    @property
    def max_new_tokens(self) -> int:
        return self.generation.max_new_tokens

    @property
    def top_p(self) -> float:
        return self.generation.top_p

    @property
    def repetition_penalty(self) -> float:
        return self.generation.repetition_penalty

    @property
    def temperature(self) -> float:
        return self.generation.temperature

    @property
    def stream_tokens(self) -> bool:
        return self.generation.stream_tokens

    @property
    def stream_chunk_size(self) -> int:
        return self.generation.stream_chunk_size

    @property
    def initial_stream_chunk_size(self) -> int:
        return self.generation.initial_stream_chunk_size


@dataclass
class DriverSegmentRequest:
    text: str
    options: DriverGenerationOptions | None = None
    stream_audio: bool | None = None


@dataclass
class DriverEvent:
    type: str = field(init=False, default="event")


@dataclass
class DriverAudioChunkEvent(DriverEvent):
    sample_rate: int
    audio: Any
    segment_index: int | None = None
    type: str = field(init=False, default="audio_chunk")


@dataclass
class DriverFinalAudioEvent(DriverEvent):
    sample_rate: int
    audio: Any
    type: str = field(init=False, default="final_audio")


@dataclass
class DriverTokenChunkEvent(DriverEvent):
    codes: Any
    text: str | None = None
    type: str = field(init=False, default="token_chunk")


@dataclass
class DriverErrorEvent(DriverEvent):
    error: Exception | None = None
    type: str = field(init=False, default="error")


@dataclass
class DriverContext:
    closed: bool = False


@dataclass
class DriverHealth:
    ok: bool
    message: str = "ok"


@dataclass
class DriverStats:
    sessions_opened: int = 0
    generated_segments: int = 0
    audio_events: int = 0


GenerationOptions = DriverGenerationOptions

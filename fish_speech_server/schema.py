import base64
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, conint, model_validator
from pydantic.functional_validators import SkipValidation
from typing_extensions import Annotated


class ServeVQPart(BaseModel):
    type: Literal["vq"] = "vq"
    codes: SkipValidation[list[list[int]]]


class ServeTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ServeAudioPart(BaseModel):
    type: Literal["audio"] = "audio"
    audio: bytes


class ServeRequest(BaseModel):
    content: dict
    max_new_tokens: int = 600
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    streaming: bool = False
    num_samples: int = 1
    early_stop_threshold: float = 1.0


class ServeVQGANEncodeRequest(BaseModel):
    audios: list[bytes]


class ServeVQGANEncodeResponse(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeRequest(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeResponse(BaseModel):
    audios: list[bytes]


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    @model_validator(mode="before")
    @classmethod
    def decode_audio(cls, values):
        if not isinstance(values, dict):
            return values

        audio = values.get("audio")
        if isinstance(audio, str) and len(audio) > 255:
            try:
                values = dict(values)
                values["audio"] = base64.b64decode(audio)
            except Exception:
                pass

        return values

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeTTSRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    format: Literal["wav", "pcm", "mp3", "flac"] = "wav"
    references: list[ServeReferenceAudio] = Field(default_factory=list)
    reference_id: str | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    normalize: bool = True
    streaming: bool = False
    stream_tokens: bool = False
    stream_chunk_size: Annotated[int, conint(ge=1, le=200, strict=True)] = 8
    initial_stream_chunk_size: Annotated[int, conint(ge=1, le=200, strict=True)] = 10
    max_new_tokens: Annotated[int, conint(ge=1, le=4096, strict=True)] = 512
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.1
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8

    @model_validator(mode="after")
    def validate_stream_chunk_sizes(self) -> "ServeTTSRequest":
        if self.initial_stream_chunk_size < self.stream_chunk_size:
            raise ValueError(
                "initial_stream_chunk_size must be >= stream_chunk_size"
            )
        return self


class AddReferenceRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-zA-Z0-9\-_ ]+$")
    audio: bytes
    text: str = Field(..., min_length=1)


class AddReferenceResponse(BaseModel):
    success: bool
    message: str
    reference_id: str


class AddEncodedReferenceResponse(BaseModel):
    success: bool
    status: str
    message: str
    reference_id: str


class ListReferencesResponse(BaseModel):
    success: bool
    reference_ids: list[str]
    message: str = "Success"


class DeleteReferenceResponse(BaseModel):
    success: bool
    message: str
    reference_id: str


class UpdateReferenceResponse(BaseModel):
    success: bool
    message: str
    old_reference_id: str
    new_reference_id: str


class OpenSynthesisSessionRequest(BaseModel):
    reference_id: str = Field(..., min_length=1)
    max_history_turns: Annotated[int, conint(ge=1)] = 2
    max_history_chars: Annotated[int, conint(ge=1)] = 220
    max_history_code_frames: Annotated[int, conint(ge=0)] = 400


class OpenSynthesisSessionResponse(BaseModel):
    ok: bool
    synthesis_session_id: str
    reference_id: str
    context: dict | None = None


class CloseSynthesisSessionResponse(BaseModel):
    ok: bool
    closed: bool
    synthesis_session_id: str


class SynthesisSessionInfoResponse(BaseModel):
    ok: bool
    synthesis_session_id: str
    context: dict


class StatefulTTSRequest(ServeTTSRequest):
    synthesis_session_id: str
    commit_seq: Annotated[int, conint(ge=1, strict=True)]
    commit_reason: str = "unknown"


class AppendHistoryRequest(BaseModel):
    text: str
    codes: SkipValidation[list[list[int]]]
    reason: str = "unknown"
    commit_seq: int = 0


class GenerateIntroCacheResponse(BaseModel):
    ok: bool
    text: str
    audio_meta: dict[str, int]
    pcm_b64: str
    pcm_bytes: int
    codes: SkipValidation[list[list[int]]] | None = None
    code_frames: int = 0


class PreloadSynthesisTurnRequest(BaseModel):
    synthesis_session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    codes: SkipValidation[list[list[int]]]
    commit_seq: int = Field(0, ge=0)
    commit_reason: str = Field("intro_cache", min_length=1)


class PreloadSynthesisTurnResponse(BaseModel):
    ok: bool
    synthesis_session_id: str
    code_frames: int
    context: dict
import base64
from typing import Literal

from pydantic import BaseModel, Field, conint, model_validator
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
    def decode_audio(cls, values):
        audio = values.get("audio")
        if isinstance(audio, str) and len(audio) > 255:
            try:
                values["audio"] = base64.b64decode(audio)
            except Exception:
                pass
        return values

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    format: Literal["wav", "pcm", "mp3"] = "wav"
    references: list[ServeReferenceAudio] = []
    reference_id: str | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    normalize: bool = True
    streaming: bool = False
    stream_tokens: bool = False
    stream_chunk_size: Annotated[int, conint(ge=1, le=200, strict=True)] = 8
    initial_stream_chunk_size: Annotated[int, conint(ge=1, le=200, strict=True)] = 10
    max_new_tokens: int = 512
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.1
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8

    class Config:
        arbitrary_types_allowed = True


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

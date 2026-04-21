from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SessionBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class BufferConfig(SessionBaseModel):
    """
    Настройки накопления текста перед отправкой в TTS.
    """

    min_words: int = Field(default=3, ge=1, le=64)
    soft_limit_chars: int = Field(default=120, ge=16, le=4000)
    hard_limit_chars: int = Field(default=220, ge=16, le=8000)

    @model_validator(mode="after")
    def validate_limits(self) -> "BufferConfig":
        if self.hard_limit_chars < self.soft_limit_chars:
            raise ValueError("hard_limit_chars must be >= soft_limit_chars")
        return self


class TTSForwardConfig(SessionBaseModel):
    """
    Что и как отправляем в основной TTS backend.
    """

    base_url: str = Field(default="http://127.0.0.1:8080")
    endpoint: str = Field(default="/v1/tts")

    format: Literal["wav", "pcm"] = "pcm"
    streaming: bool = True
    stream_tokens: bool = True
    stream_chunk_size: int = Field(default=8, ge=1, le=200)

    max_new_tokens: int = Field(default=512, ge=16, le=4096)
    top_p: float = Field(default=0.8, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=0.9, le=2.0)
    temperature: float = Field(default=0.8, ge=0.1, le=1.2)

    cleanup_mode: Literal["request_end", "session_idle", "none"] = "session_idle"
    stream_empty_cache: bool | None = None

    use_memory_cache: Literal["on", "off"] = "off"
    normalize: bool = True
    reference_id: str | None = None

    connect_timeout_sec: float = Field(default=10.0, gt=0.0, le=120.0)
    read_timeout_sec: float = Field(default=120.0, gt=0.0, le=3600.0)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("endpoint")
    @classmethod
    def normalize_endpoint(cls, v: str) -> str:
        if not v.startswith("/"):
            v = "/" + v
        return v


class SessionPolicyConfig(SessionBaseModel):
    """
    Настройки жизненного цикла session_mode.
    """

    session_idle_timeout_sec: float = Field(default=15.0, ge=1.0, le=3600.0)
    cleanup_after_idle_sec: float = Field(default=2.0, ge=0.0, le=300.0)
    force_flush_after_sec: float = Field(default=0.35, ge=0.0, le=10.0)
    max_pending_emit_chunks: int = Field(default=32, ge=1, le=4096)
    close_tts_stream_on_new_text: bool = False

    @model_validator(mode="after")
    def validate_policy(self) -> "SessionPolicyConfig":
        if self.cleanup_after_idle_sec > self.session_idle_timeout_sec:
            raise ValueError(
                "cleanup_after_idle_sec must be <= session_idle_timeout_sec"
            )
        return self


class SessionModeConfig(SessionBaseModel):
    """
    Полный runtime-конфиг одной сессии.
    """

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    sample_rate: int = Field(default=44100, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=2)

    buffer: BufferConfig = Field(default_factory=BufferConfig)
    tts: TTSForwardConfig = Field(default_factory=TTSForwardConfig)
    policy: SessionPolicyConfig = Field(default_factory=SessionPolicyConfig)


class ClientStartSession(SessionBaseModel):
    type: Literal["start_session"] = "start_session"
    config: SessionModeConfig | None = None


class ClientPatchConfig(SessionBaseModel):
    type: Literal["patch_config"] = "patch_config"
    patch: dict[str, Any]


class ClientTextDelta(SessionBaseModel):
    """
    Очередная дельта текста от LLM/оркестратора.
    """

    type: Literal["text_delta"] = "text_delta"
    text: str = Field(min_length=1, max_length=20000)
    final: bool = False
    trace_id: str | None = None


class ClientFlush(SessionBaseModel):
    type: Literal["flush"] = "flush"
    final: bool = False
    reason: str | None = None


class ClientClear(SessionBaseModel):
    """
    Сбросить только текстовый буфер session_mode, не трогая саму TTS модель.
    """

    type: Literal["clear"] = "clear"
    reason: str | None = None


class ClientCleanup(SessionBaseModel):
    """
    Попросить backend выполнить heavy cleanup.
    """

    type: Literal["cleanup"] = "cleanup"
    reason: str | None = None


class ClientCloseSession(SessionBaseModel):
    type: Literal["close_session"] = "close_session"
    reason: str | None = None


class ClientPing(SessionBaseModel):
    type: Literal["ping"] = "ping"
    ts_ms: int | None = None


ClientMessage = (
    ClientStartSession
    | ClientPatchConfig
    | ClientTextDelta
    | ClientFlush
    | ClientClear
    | ClientCleanup
    | ClientCloseSession
    | ClientPing
)


class TTSChunkRequest(SessionBaseModel):
    """
    Внутреннее описание одного отправляемого в backend текстового куска.
    """

    session_id: str
    chunk_id: str = Field(default_factory=lambda: uuid4().hex)

    text: str = Field(min_length=1)
    reason: Literal["punct", "hard_limit", "force", "final"]

    words: int = Field(ge=0)
    chars: int = Field(ge=0)

    trace_id: str | None = None
    final: bool = False

    @model_validator(mode="after")
    def validate_text_metrics(self) -> "TTSChunkRequest":
        if self.chars <= 0:
            self.chars = len(self.text)
        return self


class TTSAudioMeta(SessionBaseModel):
    sample_rate: int = Field(default=44100, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=2)
    format: Literal["pcm", "wav"] = "pcm"


class ServerSessionStarted(SessionBaseModel):
    event: Literal["session_started"] = "session_started"
    session_id: str
    config: SessionModeConfig


class ServerConfigPatched(SessionBaseModel):
    event: Literal["config_patched"] = "config_patched"
    session_id: str
    config: SessionModeConfig


class ServerTextAccepted(SessionBaseModel):
    event: Literal["text_accepted"] = "text_accepted"
    session_id: str
    buffered_text_len: int = Field(ge=0)
    buffered_words: int = Field(ge=0)
    trace_id: str | None = None
    final: bool = False


class ServerChunkQueued(SessionBaseModel):
    event: Literal["chunk_queued"] = "chunk_queued"
    session_id: str
    chunk: TTSChunkRequest
    queue_size: int = Field(ge=0)


class ServerTTSStarted(SessionBaseModel):
    event: Literal["tts_started"] = "tts_started"
    session_id: str
    chunk_id: str
    text: str
    trace_id: str | None = None


class ServerAudioMeta(SessionBaseModel):
    event: Literal["audio_meta"] = "audio_meta"
    session_id: str
    chunk_id: str
    meta: TTSAudioMeta


class ServerAudioChunk(SessionBaseModel):
    """
    PCM/WAV chunk в base64 тут специально не упаковываем.
    По websocket обычно удобнее слать бинарным frame отдельно,
    но если понадобится JSON-only транспорт — можно добавить payload_b64.
    """

    event: Literal["audio_chunk"] = "audio_chunk"
    session_id: str
    chunk_id: str
    seq: int = Field(ge=1)
    size_bytes: int = Field(ge=0)
    trace_id: str | None = None


class ServerTTSFinished(SessionBaseModel):
    event: Literal["tts_finished"] = "tts_finished"
    session_id: str
    chunk_id: str
    total_chunks: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    trace_id: str | None = None
    final: bool = False


class ServerBufferCleared(SessionBaseModel):
    event: Literal["buffer_cleared"] = "buffer_cleared"
    session_id: str
    reason: str | None = None


class ServerCleanupDone(SessionBaseModel):
    event: Literal["cleanup_done"] = "cleanup_done"
    session_id: str
    reason: str | None = None


class ServerPong(SessionBaseModel):
    event: Literal["pong"] = "pong"
    session_id: str | None = None
    ts_ms: int | None = None


class ServerSessionClosed(SessionBaseModel):
    event: Literal["session_closed"] = "session_closed"
    session_id: str
    reason: str | None = None


class ServerError(SessionBaseModel):
    event: Literal["error"] = "error"
    code: str
    message: str
    session_id: str | None = None
    details: dict[str, Any] | None = None
    fatal: bool = False


ServerEvent = (
    ServerSessionStarted
    | ServerConfigPatched
    | ServerTextAccepted
    | ServerChunkQueued
    | ServerTTSStarted
    | ServerAudioMeta
    | ServerAudioChunk
    | ServerTTSFinished
    | ServerBufferCleared
    | ServerCleanupDone
    | ServerPong
    | ServerSessionClosed
    | ServerError
)


def parse_client_message(payload: dict[str, Any] | str) -> ClientMessage:
    """
    Явный парсер входящих websocket/json сообщений.
    Поддерживает:
    - plain text -> text_delta
    - {"text": "..."} без type -> text_delta
    - обычные typed messages
    """
    if isinstance(payload, str):
        return ClientTextDelta.model_validate(
            {
                "type": "text_delta",
                "text": payload,
            }
        )

    if not isinstance(payload, dict):
        raise ValueError("Client message must be an object or plain text string")

    msg_type = payload.get("type")

    if msg_type is None and "text" in payload:
        payload = {"type": "text_delta", **payload}
        msg_type = "text_delta"

    mapping: dict[str, type[SessionBaseModel]] = {
        "start_session": ClientStartSession,
        "patch_config": ClientPatchConfig,
        "text_delta": ClientTextDelta,
        "flush": ClientFlush,
        "clear": ClientClear,
        "cleanup": ClientCleanup,
        "close_session": ClientCloseSession,
        "ping": ClientPing,
    }

    model_cls = mapping.get(msg_type)
    if model_cls is None:
        raise ValueError(f"Unsupported client message type: {msg_type!r}")

    return model_cls.model_validate(payload)  # type: ignore[return-value]


def event_to_dict(event: ServerEvent) -> dict[str, Any]:
    return event.model_dump(mode="json")
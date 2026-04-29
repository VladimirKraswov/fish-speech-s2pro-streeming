from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _default_runtime_config_path() -> Path:
    env_path = (
        os.getenv("FISH_RUNTIME_CONFIG")
        or os.getenv("FISH_RUNTIME_CONFIG_PATH")
        or os.getenv("FISH_SERVER_CONFIG")
        or os.getenv("FISH_CONFIG")
    )
    if env_path:
        return Path(env_path)

    candidates = (
        Path("/app/config/runtime.json"),
        Path("config/runtime.json"),
        Path("runtime.json"),
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path("config/runtime.json")


DEFAULT_RUNTIME_CONFIG_PATH = _default_runtime_config_path()


# =============================================================================
# Proxy-safe copies of driver runtime config models.
#
# ВАЖНО:
# Не импортируем fish_speech.driver.config здесь.
# Proxy-контейнер импортирует fish_speech_server.config, а fish_speech.* тянет torch.
# В proxy torch не установлен и не должен быть установлен.
# =============================================================================
class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    llama_checkpoint_path: str
    decoder_checkpoint_path: str
    decoder_config_name: str


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    device: str = "cuda"
    precision: str = "bfloat16"
    compile: bool = False
    record_memory_history: bool = False
    memory_history_max_entries: int = Field(100_000, ge=1)


class WarmupConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    reference_id: str | None = None
    text: str = "Hello."
    max_new_tokens: int = Field(125, ge=1, le=1024)
    chunk_length: int = Field(160, ge=100, le=300)
    streaming: bool = True
    stream_tokens: bool = True
    initial_stream_chunk_size: int = Field(24, ge=1, le=200)
    stream_chunk_size: int = Field(18, ge=1, le=200)


class NetworkEndpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str
    port: int = Field(..., ge=1, le=65535)


class NetworkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: NetworkEndpointConfig
    proxy: NetworkEndpointConfig
    upstream_tts_url: str


class CommitStageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_chars: int = Field(..., ge=1)
    target_chars: int = Field(..., ge=1)
    max_chars: int = Field(..., ge=1)
    max_wait_ms: int = Field(..., ge=1)
    allow_partial_after_ms: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_lengths(self) -> "CommitStageConfig":
        if self.min_chars > self.target_chars:
            raise ValueError("min_chars must be <= target_chars")
        if self.target_chars > self.max_chars:
            raise ValueError("target_chars must be <= max_chars")
        if self.max_wait_ms > self.allow_partial_after_ms:
            raise ValueError("max_wait_ms must be <= allow_partial_after_ms")
        return self


class CommitPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    first: CommitStageConfig
    next: CommitStageConfig
    flush_on_sentence_punctuation: bool = True
    flush_on_clause_punctuation: bool = False
    flush_on_newline: bool = True
    carry_incomplete_tail: bool = True


class ProxyTTSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_id: str = "voice"
    format: str = "wav"
    normalize: bool = True
    use_memory_cache: str = "on"
    seed: int | None = None

    max_new_tokens: int = Field(300, ge=1, le=512)
    chunk_length: int = Field(160, ge=100, le=300)

    top_p: float = Field(0.8, ge=0.1, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.9, le=2.0)
    temperature: float = Field(0.66, ge=0.1, le=1.0)

    stream_tokens: bool = True
    initial_stream_chunk_size: int = Field(10, ge=1, le=200)
    stream_chunk_size: int = Field(8, ge=1, le=200)
    first_initial_stream_chunk_size: int | None = Field(None, ge=1, le=200)
    first_stream_chunk_size: int | None = Field(None, ge=1, le=200)

    stateful_synthesis: bool = True
    stateful_fallback_to_stateless: bool = False

    stateful_history_turns: int = Field(1, ge=1, le=4)
    stateful_history_chars: int = Field(160, ge=1, le=1000)
    stateful_history_code_frames: int = Field(260, ge=0, le=2000)
    stateful_reset_every_commits: int = Field(0, ge=0, le=1000)
    stateful_reset_every_chars: int = Field(0, ge=0, le=100_000)

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        value = value.strip().lower()
        if value != "wav":
            raise ValueError("only format='wav' is supported in proxy runtime config")
        return value

    @field_validator("use_memory_cache")
    @classmethod
    def validate_use_memory_cache(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"on", "off"}:
            raise ValueError("use_memory_cache must be 'on' or 'off'")
        return value

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "ProxyTTSConfig":
        if self.initial_stream_chunk_size < self.stream_chunk_size:
            raise ValueError("initial_stream_chunk_size must be >= stream_chunk_size")

        first_initial = (
            self.first_initial_stream_chunk_size or self.initial_stream_chunk_size
        )
        first_stream = self.first_stream_chunk_size or self.stream_chunk_size

        if first_initial < first_stream:
            raise ValueError(
                "first initial_stream_chunk_size must be >= first stream_chunk_size"
            )

        return self


class PlaybackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_emit_bytes: int = Field(8192, ge=512, le=65536)
    first_commit_target_emit_bytes: int = Field(8192, ge=512, le=65536)
    start_buffer_ms: int = Field(180, ge=0, le=5000)
    first_commit_start_buffer_ms: int = Field(120, ge=0, le=5000)
    client_start_buffer_ms: int = Field(160, ge=0, le=5000)
    client_initial_start_delay_ms: int = Field(40, ge=0, le=1000)
    stop_grace_ms: int = Field(100, ge=0, le=5000)

    boundary_smoothing_enabled: bool = True
    punctuation_pauses_enabled: bool = True

    fade_in_ms: int = Field(8, ge=0, le=100)
    fade_out_ms: int = Field(12, ge=0, le=100)

    pause_after_clause_ms: int = Field(110, ge=0, le=2000)
    pause_after_sentence_ms: int = Field(280, ge=0, le=3000)
    pause_after_newline_ms: int = Field(520, ge=0, le=5000)
    pause_after_force_ms: int = Field(220, ge=0, le=3000)
    pause_after_hard_limit_ms: int = Field(40, ge=0, le=1000)

    @field_validator("target_emit_bytes", "first_commit_target_emit_bytes")
    @classmethod
    def validate_even_bytes(cls, value: int) -> int:
        if value % 2 != 0:
            raise ValueError("PCM emit byte sizes must be even for PCM16")
        return value


class SessionRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_buffer_chars: int = Field(12000, ge=256, le=100_000)
    auto_close_on_finish: bool = False


class IntroCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False

    # Текст, который будет синтезироваться и кэшироваться.
    # Например: короткое приветствие или стабильный intro-фрагмент.
    text: str = ""

    # Максимум записей в памяти proxy.
    max_entries: int = Field(16, ge=1, le=256)

    # TTL записи в секундах.
    ttl_sec: int = Field(3600, ge=1, le=86400)

    # Если true — при /session/open сразу прогревать intro cache.
    # Если false — lazy generate при первом /pcm-stream.
    warm_on_session_open: bool = False

    # Если true — при ошибке генерации intro не валить session,
    # а просто пропустить intro и продолжить обычный stream.
    ignore_errors: bool = True

    # Размер PCM чанка для отдачи intro.
    emit_bytes: int = Field(8192, ge=512, le=65536)

    # Пауза после intro перед первым реальным commit.
    pause_after_ms: int = Field(0, ge=0, le=3000)

    @field_validator("emit_bytes")
    @classmethod
    def validate_even_emit_bytes(cls, value: int) -> int:
        if value % 2 != 0:
            raise ValueError("intro_cache.emit_bytes must be even for PCM16")
        return value


class ProxyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_reference_id: str = "voice"
    session_ttl_sec: int = Field(1800, ge=1)
    session_max_count: int = Field(128, ge=1)

    version: int = 1
    commit: CommitPolicyConfig
    tts: ProxyTTSConfig
    playback: PlaybackConfig
    session: SessionRuntimeConfig
    intro_cache: IntroCacheConfig = Field(default_factory=IntroCacheConfig)

    @model_validator(mode="after")
    def normalize_reference_id(self) -> "ProxyConfig":
        if not self.tts.reference_id.strip():
            self.tts.reference_id = self.default_reference_id
        return self


class FrontendOverridesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    allowed_paths: list[str] = Field(default_factory=list)


class ServerRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    paths: PathsConfig
    network: NetworkConfig
    model: ModelConfig
    warmup: WarmupConfig
    proxy: ProxyConfig
    frontend_overrides: FrontendOverridesConfig


AppConfig = ServerRuntimeConfig


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))

    for key, value in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value

    return out


def _iter_leaf_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []

        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_iter_leaf_paths(child, child_prefix))

        return paths

    return [prefix] if prefix else []


def _normalize_allowed_path(path: str) -> str:
    path = path.strip().strip(".")

    if path.startswith("proxy."):
        path = path[len("proxy.") :]

    if path.startswith("session."):
        path = path[len("session.") :]

    return path


def merge_frontend_proxy_override(
    override_patch: dict[str, Any],
    runtime: ServerRuntimeConfig | None = None,
) -> ProxyConfig:
    runtime = runtime or load_runtime_config()

    if not isinstance(override_patch, dict):
        raise ValueError("frontend override must be a JSON object")

    if runtime.frontend_overrides.enabled:
        requested = {
            _normalize_allowed_path(path)
            for path in _iter_leaf_paths(override_patch)
            if path.strip()
        }

        allowed = {
            _normalize_allowed_path(path)
            for path in runtime.frontend_overrides.allowed_paths
            if path.strip()
        }

        disallowed = sorted(path for path in requested if path not in allowed)

        if disallowed:
            raise ValueError(
                "frontend override contains disallowed paths: " + ", ".join(disallowed)
            )

    merged = deep_merge(runtime.proxy.model_dump(mode="python"), override_patch)
    return ProxyConfig.model_validate(merged)


@lru_cache(maxsize=4)
def load_runtime_config(path: str | Path | None = None) -> ServerRuntimeConfig:
    config_path = Path(path) if path is not None else DEFAULT_RUNTIME_CONFIG_PATH

    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return ServerRuntimeConfig.model_validate(payload)


load_server_config = load_runtime_config
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_CONFIG_PATH = ROOT_DIR / "config" / "runtime.json"


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llama_checkpoint_path: str = "checkpoints/s2-pro"
    decoder_checkpoint_path: str = "checkpoints/s2-pro/codec.pth"
    decoder_config_name: str = "modded_dac_vq"
    references_dir: str = "references"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "cuda"
    precision: str = "bfloat16"
    compile: bool = True
    cache_max_seq_len: int = Field(768, ge=1)
    max_new_tokens_cap: int = Field(128, ge=1)
    sdpa_math: bool = False

    cleanup_after_request: bool = False
    cleanup_every_n_requests: int = Field(0, ge=0)
    cleanup_on_error: bool = True
    cleanup_on_abort: bool = True
    empty_cache_per_stream_chunk: bool = False
    empty_cache_per_segment: bool = False

    profile_inference: bool = False
    record_memory_history: bool = False
    memory_history_max_entries: int = Field(100000, ge=1)

    # Long-form settings
    long_form_auto_split: bool = True
    long_form_target_chars: int = Field(220, ge=40)
    long_form_max_chars: int = Field(320, ge=80)
    long_form_context_policy: str = "tail_frames"
    long_form_tail_frames: int = Field(48, ge=0)
    long_form_max_history_segments: int = Field(1, ge=0)
    long_form_tokens_per_char: float = Field(1.3, ge=0.1, le=5.0)
    long_form_token_overhead: int = Field(32, ge=0)
    long_form_min_new_tokens: int = Field(48, ge=1)
    long_form_max_new_tokens_per_segment: int = Field(256, ge=1)

    @field_validator("long_form_context_policy")
    @classmethod
    def validate_long_form_context_policy(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"none", "last_segment", "tail_frames"}:
            raise ValueError(
                "long_form_context_policy must be one of: none, last_segment, tail_frames"
            )
        return value

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"bfloat16", "float16", "float32"}:
            raise ValueError("precision must be one of: bfloat16, float16, float32")
        return value


class ProxyConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_ttl_sec: int = Field(1800, ge=1)
    session_max_count: int = Field(128, ge=1)


class WarmupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    reference_id: str | None = "voice"
    text: str = "Привет. Это дополнительный прогрев стримингового режима для Fish Speech."
    streaming: bool = True
    stream_tokens: bool = True
    max_new_tokens: int = Field(96, ge=1)
    chunk_length: int = Field(180, ge=100, le=300)
    initial_stream_chunk_size: int = Field(8, ge=1, le=200)
    stream_chunk_size: int = Field(8, ge=1, le=200)

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "WarmupConfig":
        if self.initial_stream_chunk_size < self.stream_chunk_size:
            raise ValueError("initial_stream_chunk_size must be >= stream_chunk_size")
        return self


class DriverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    paths: PathsConfig
    model: ModelConfig
    warmup: WarmupConfig
    proxy: ProxyConfig


def _driver_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": payload.get("version", 1),
        "paths": payload.get("paths", {}),
        "model": payload.get("model", {}),
        "warmup": payload.get("warmup", {}),
        "proxy": payload.get("proxy", {}),
    }


@lru_cache(maxsize=4)
def load_runtime_config(path: str | Path | None = None) -> DriverConfig:
    config_path = Path(path) if path is not None else DEFAULT_RUNTIME_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return DriverConfig.model_validate(_driver_payload(payload))


load_driver_config = load_runtime_config

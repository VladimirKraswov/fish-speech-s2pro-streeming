from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import os
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional, Tuple

import httpx
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

UPSTREAM_TTS_URL = os.environ.get("UPSTREAM_TTS_URL", "http://127.0.0.1:8080/v1/tts")
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
DEFAULT_REFERENCE_ID = os.environ.get("DEFAULT_REFERENCE_ID", "voice")
SESSION_TTL_SEC = int(os.environ.get("SESSION_TTL_SEC", "1800"))
SESSION_MAX_COUNT = int(os.environ.get("SESSION_MAX_COUNT", "128"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("fish-proxy")

app = FastAPI(title="Fish Speech PCM Proxy", version="0.7.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CommitterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    soft_min_chars: int = Field(40, ge=1, le=500)
    target_chars: int = Field(120, ge=1, le=800)
    hard_max_chars: int = Field(180, ge=1, le=1200)
    flush_on_punctuation: bool = True
    flush_on_clause_punctuation: bool = True
    flush_on_newline: bool = True
    max_wait_ms: int = Field(900, ge=50, le=20_000)
    allow_partial_after_ms: int = Field(1400, ge=50, le=60_000)
    carry_incomplete_tail: bool = True

    @model_validator(mode="after")
    def validate_lengths(self) -> "CommitterConfig":
        if self.soft_min_chars > self.target_chars:
            raise ValueError("soft_min_chars must be <= target_chars")
        if self.target_chars > self.hard_max_chars:
            raise ValueError("target_chars must be <= hard_max_chars")
        if self.max_wait_ms > self.allow_partial_after_ms:
            raise ValueError("max_wait_ms must be <= allow_partial_after_ms")
        return self


class TTSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_id: str = Field(DEFAULT_REFERENCE_ID, min_length=1, max_length=255)
    format: str = Field("wav")
    normalize: bool = True
    use_memory_cache: str = Field("on")
    seed: int | None = None
    max_new_tokens: int = Field(96, ge=1, le=512)
    chunk_length: int = Field(200, ge=100, le=300)
    top_p: float = Field(0.8, ge=0.1, le=1.0)
    repetition_penalty: float = Field(1.1, ge=0.9, le=2.0)
    temperature: float = Field(0.8, ge=0.1, le=1.0)
    stream_tokens: bool = True
    initial_stream_chunk_size: int = Field(18, ge=1, le=200)
    stream_chunk_size: int = Field(8, ge=1, le=200)

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        value = value.lower().strip()
        if value != "wav":
            raise ValueError("proxy currently supports only format='wav'")
        return value

    @field_validator("use_memory_cache")
    @classmethod
    def validate_cache(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"on", "off"}:
            raise ValueError("use_memory_cache must be 'on' or 'off'")
        return value

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "TTSConfig":
        if self.initial_stream_chunk_size < self.stream_chunk_size:
            raise ValueError(
                "initial_stream_chunk_size must be >= stream_chunk_size"
            )
        return self


class PlaybackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_emit_bytes: int = Field(4096, ge=512, le=32768)
    start_buffer_ms: int = Field(120, ge=0, le=5000)
    stop_grace_ms: int = Field(150, ge=0, le=5000)

    @field_validator("target_emit_bytes")
    @classmethod
    def validate_even_bytes(cls, value: int) -> int:
        if value % 2 != 0:
            raise ValueError("target_emit_bytes must be even for PCM16")
        return value


class SessionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    committer: CommitterConfig = Field(default_factory=CommitterConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    playback: PlaybackConfig = Field(default_factory=PlaybackConfig)


DEFAULT_SESSION_CONFIG = SessionConfig().model_dump(mode="python")


class SessionOpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_text: str = Field(..., min_length=2, max_length=100_000)


class SessionCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str


@dataclass
class SessionRecord:
    session_id: str
    config: dict[str, Any]
    raw_config_text: str
    created_at: float
    updated_at: float
    expires_at: float


class SessionStore:
    def __init__(self) -> None:
        self._items: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()

    async def cleanup(self) -> None:
        now = time.time()
        async with self._lock:
            expired = [sid for sid, rec in self._items.items() if rec.expires_at <= now]
            for sid in expired:
                self._items.pop(sid, None)
            if expired:
                logger.info("session cleanup removed=%s", len(expired))

    async def create(self, config: dict[str, Any], raw_config_text: str) -> SessionRecord:
        await self.cleanup()
        async with self._lock:
            if len(self._items) >= SESSION_MAX_COUNT:
                raise HTTPException(429, detail="too many active sessions")
            now = time.time()
            session_id = uuid.uuid4().hex
            rec = SessionRecord(
                session_id=session_id,
                config=config,
                raw_config_text=raw_config_text,
                created_at=now,
                updated_at=now,
                expires_at=now + SESSION_TTL_SEC,
            )
            self._items[session_id] = rec
            return rec

    async def get(self, session_id: str, touch: bool = True) -> SessionRecord | None:
        await self.cleanup()
        async with self._lock:
            rec = self._items.get(session_id)
            if rec is None:
                return None
            if touch:
                now = time.time()
                rec.updated_at = now
                rec.expires_at = now + SESSION_TTL_SEC
            return rec

    async def close(self, session_id: str) -> bool:
        async with self._lock:
            return self._items.pop(session_id, None) is not None

    async def stats(self) -> dict[str, Any]:
        await self.cleanup()
        async with self._lock:
            now = time.time()
            return {
                "active_sessions": len(self._items),
                "ttl_sec": SESSION_TTL_SEC,
                "max_sessions": SESSION_MAX_COUNT,
                "sessions": [
                    {
                        "session_id": rec.session_id,
                        "age_sec": round(now - rec.created_at, 1),
                        "idle_sec": round(now - rec.updated_at, 1),
                        "expires_in_sec": max(0, round(rec.expires_at - now, 1)),
                    }
                    for rec in self._items.values()
                ],
            }


session_store = SessionStore()


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_config_text(config_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, detail=f"invalid config json: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, detail="config json must be an object")
    return payload


def normalize_config(config_text: str) -> SessionConfig:
    payload = _parse_config_text(config_text)
    merged = _deep_merge(DEFAULT_SESSION_CONFIG, payload)
    try:
        return SessionConfig.model_validate(merged)
    except ValidationError as exc:
        raise HTTPException(400, detail=json.loads(exc.json())) from exc


def build_upstream_payload(
    text: str,
    config: SessionConfig,
    reference_id_override: str | None = None,
) -> dict[str, Any]:
    tts = config.tts
    payload = {
        "text": text,
        "streaming": True,
        "stream_tokens": tts.stream_tokens,
        "initial_stream_chunk_size": tts.initial_stream_chunk_size,
        "stream_chunk_size": tts.stream_chunk_size,
        "reference_id": (reference_id_override or tts.reference_id).strip(),
        "format": tts.format,
        "normalize": tts.normalize,
        "use_memory_cache": tts.use_memory_cache,
        "max_new_tokens": tts.max_new_tokens,
        "chunk_length": tts.chunk_length,
        "top_p": tts.top_p,
        "repetition_penalty": tts.repetition_penalty,
        "temperature": tts.temperature,
    }
    if tts.seed is not None:
        payload["seed"] = tts.seed
    return payload


def _parse_wav_header(buf: bytes) -> Optional[Tuple[int, int, int, int]]:
    if len(buf) < 12:
        return None

    if buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError("upstream did not return a RIFF/WAVE stream")

    pos = 12
    channels = None
    sample_rate = None
    bits_per_sample = None

    while True:
        if len(buf) < pos + 8:
            return None

        chunk_id = buf[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", buf, pos + 4)[0]
        chunk_data_start = pos + 8

        if chunk_id == b"fmt ":
            needed = chunk_data_start + chunk_size + (chunk_size % 2)
            if len(buf) < needed:
                return None
            if chunk_size < 16:
                raise ValueError("invalid WAV fmt chunk")

            audio_format, channels, sample_rate = struct.unpack_from(
                "<HHI", buf, chunk_data_start
            )
            bits_per_sample = struct.unpack_from("<H", buf, chunk_data_start + 14)[0]
            if audio_format != 1:
                raise ValueError(f"only PCM WAV is supported, got format={audio_format}")
            pos = needed
            continue

        if chunk_id == b"data":
            if channels is None or sample_rate is None or bits_per_sample is None:
                raise ValueError("WAV data chunk arrived before fmt chunk")
            return sample_rate, channels, bits_per_sample, chunk_data_start

        needed = chunk_data_start + chunk_size + (chunk_size % 2)
        if len(buf) < needed:
            return None
        pos = needed


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "fish-pcm-proxy",
        "version": app.version,
        "upstream_tts_url": UPSTREAM_TTS_URL,
        "default_reference_id": DEFAULT_REFERENCE_ID,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    stats = await session_store.stats()
    return {"ok": True, **stats}


@app.post("/session/open")
async def session_open(req: SessionOpenRequest) -> JSONResponse:
    config = normalize_config(req.config_text)
    rec = await session_store.create(config.model_dump(mode="python"), req.config_text)
    logger.info(
        "session open id=%s ref=%s target_emit_bytes=%s",
        rec.session_id[:8],
        config.tts.reference_id,
        config.playback.target_emit_bytes,
    )
    return JSONResponse(
        {
            "ok": True,
            "session_id": rec.session_id,
            "config": rec.config,
            "ttl_sec": SESSION_TTL_SEC,
            "server_defaults": {
                "upstream_tts_url": UPSTREAM_TTS_URL,
                "default_reference_id": DEFAULT_REFERENCE_ID,
            },
        }
    )


@app.get("/session/{session_id}")
async def session_get(session_id: str) -> JSONResponse:
    rec = await session_store.get(session_id, touch=False)
    if rec is None:
        raise HTTPException(404, detail="session not found")
    return JSONResponse(
        {
            "ok": True,
            "session_id": rec.session_id,
            "config": rec.config,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
            "expires_at": rec.expires_at,
        }
    )


@app.post("/session/close")
async def session_close(req: SessionCloseRequest) -> JSONResponse:
    removed = await session_store.close(req.session_id)
    return JSONResponse({"ok": True, "closed": removed, "session_id": req.session_id})


@app.get("/pcm-stream")
async def pcm_stream(
    text: str = Query(..., min_length=1, max_length=2000),
    session_id: str | None = Query(default=None),
    reference_id: str | None = Query(default=None),
):
    req_id = uuid.uuid4().hex[:8]

    if session_id:
        rec = await session_store.get(session_id, touch=True)
        if rec is None:
            raise HTTPException(404, detail="session not found")
        config = SessionConfig.model_validate(rec.config)
        config_source = f"session:{session_id[:8]}"
    else:
        config = SessionConfig()
        config_source = "defaults"

    payload = build_upstream_payload(text=text, config=config, reference_id_override=reference_id)
    target_emit_bytes = config.playback.target_emit_bytes

    logger.info(
        "REQ %s start text_len=%s config=%s ref=%s emit_bytes=%s",
        req_id,
        len(text),
        config_source,
        payload["reference_id"],
        target_emit_bytes,
    )

    async def emit(obj: dict[str, Any]) -> bytes:
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    async def body_iter() -> AsyncGenerator[bytes, None]:
        started_at = time.perf_counter()
        header_buffer = bytearray()
        pending_pcm = bytearray()

        header_sent = False
        upstream_bytes = 0
        pcm_bytes_sent = 0
        pcm_chunks_sent = 0

        async def flush_pending(force: bool = False) -> AsyncGenerator[bytes, None]:
            nonlocal pcm_chunks_sent, pcm_bytes_sent
            while len(pending_pcm) >= target_emit_bytes or (force and pending_pcm):
                if len(pending_pcm) >= target_emit_bytes:
                    out = bytes(pending_pcm[:target_emit_bytes])
                    del pending_pcm[:target_emit_bytes]
                else:
                    out = bytes(pending_pcm)
                    pending_pcm.clear()

                pcm_chunks_sent += 1
                pcm_bytes_sent += len(out)
                yield await emit(
                    {
                        "type": "pcm",
                        "req_id": req_id,
                        "seq": pcm_chunks_sent,
                        "data": base64.b64encode(out).decode("ascii"),
                    }
                )
                await asyncio.sleep(0)

        yield await emit(
            {
                "type": "proxy_start",
                "req_id": req_id,
                "session_id": session_id,
                "config_source": config_source,
                "target_emit_bytes": target_emit_bytes,
            }
        )
        await asyncio.sleep(0)

        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                logger.info("REQ %s connecting upstream %s", req_id, UPSTREAM_TTS_URL)
                async with client.stream("POST", UPSTREAM_TTS_URL, json=payload) as upstream:
                    logger.info(
                        "REQ %s upstream connected status=%s content_type=%s",
                        req_id,
                        upstream.status_code,
                        upstream.headers.get("content-type"),
                    )

                    if upstream.status_code != 200:
                        text_body = await upstream.aread()
                        msg = text_body.decode("utf-8", errors="replace")
                        logger.error(
                            "REQ %s upstream status=%s body=%s",
                            req_id,
                            upstream.status_code,
                            msg,
                        )
                        yield await emit(
                            {
                                "type": "error",
                                "req_id": req_id,
                                "status_code": upstream.status_code,
                                "message": msg,
                            }
                        )
                        return

                    async for chunk in upstream.aiter_raw(1024):
                        if not chunk:
                            continue
                        upstream_bytes += len(chunk)

                        if not header_sent:
                            header_buffer.extend(chunk)
                            try:
                                parsed = _parse_wav_header(header_buffer)
                            except Exception as exc:  # noqa: BLE001
                                logger.exception("REQ %s wav parse failed: %s", req_id, exc)
                                yield await emit(
                                    {
                                        "type": "error",
                                        "req_id": req_id,
                                        "message": f"wav parse failed: {exc}",
                                    }
                                )
                                return

                            if parsed is None:
                                continue

                            sample_rate, channels, bits_per_sample, data_offset = parsed
                            if bits_per_sample != 16:
                                yield await emit(
                                    {
                                        "type": "error",
                                        "req_id": req_id,
                                        "message": f"unsupported bits_per_sample={bits_per_sample}",
                                    }
                                )
                                return

                            header_sent = True
                            yield await emit(
                                {
                                    "type": "meta",
                                    "req_id": req_id,
                                    "sample_rate": sample_rate,
                                    "channels": channels,
                                    "sample_width": bits_per_sample // 8,
                                }
                            )
                            await asyncio.sleep(0)

                            pcm = bytes(header_buffer[data_offset:])
                            header_buffer.clear()
                            if pcm:
                                pending_pcm.extend(pcm)
                                async for out_event in flush_pending(force=False):
                                    yield out_event
                        else:
                            pending_pcm.extend(chunk)
                            async for out_event in flush_pending(force=False):
                                yield out_event

        except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
            logger.warning("REQ %s upstream ended imperfectly: %s", req_id, exc)
            if not header_sent:
                yield await emit(
                    {
                        "type": "error",
                        "req_id": req_id,
                        "message": "upstream closed before WAV header completed",
                    }
                )
                return
        except asyncio.CancelledError:
            logger.warning("REQ %s cancelled by client", req_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("REQ %s stream failed: %s", req_id, exc)
            yield await emit(
                {
                    "type": "error",
                    "req_id": req_id,
                    "message": f"stream failed: {exc}",
                }
            )
            return

        async for out_event in flush_pending(force=True):
            yield out_event

        elapsed = time.perf_counter() - started_at
        logger.info(
            "REQ %s done elapsed=%.3fs upstream_bytes=%s pcm_bytes_sent=%s pcm_chunks_sent=%s",
            req_id,
            elapsed,
            upstream_bytes,
            pcm_bytes_sent,
            pcm_chunks_sent,
        )
        yield await emit({"type": "done", "req_id": req_id})

    return StreamingResponse(
        body_iter(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )

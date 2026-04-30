from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from fish_speech_server.config import (
    CommitPolicyConfig,
    ProxyConfig,
    load_runtime_config,
    merge_frontend_proxy_override,
)

_RUNTIME = load_runtime_config()
UPSTREAM_TTS_URL = os.getenv("FISH_UPSTREAM_TTS_URL", _RUNTIME.network.upstream_tts_url)

if os.getenv("UPSTREAM_TTS_URL"):
    UPSTREAM_TTS_URL = os.getenv("UPSTREAM_TTS_URL")

UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
FISH_API_KEY = os.getenv("FISH_API_KEY", "").strip()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


PREFIX_CACHE_SALT = os.getenv("FISH_PREFIX_CACHE_SALT", "").strip()
PREFIX_CACHE_SCHEMA_VERSION = 2

PREFIX_CACHE_MAX_ENTRIES = int(os.getenv("FISH_PREFIX_CACHE_MAX_ENTRIES", "4096"))
PREFIX_CACHE_EMIT_BYTES = int(os.getenv("FISH_PREFIX_CACHE_EMIT_BYTES", "8192"))

# Пауза после озвученного prefix-cache перед генерацией хвоста.
# Не портит append_sent_to_first_pcm: первый звук уже отдан из cache.
PREFIX_CACHE_PAUSE_AFTER_MS = int(os.getenv("FISH_PREFIX_CACHE_PAUSE_AFTER_MS", "90"))

# Затухание конца prefix-cache, чтобы склейка не щёлкала.
PREFIX_CACHE_FADE_OUT_MS = int(os.getenv("FISH_PREFIX_CACHE_FADE_OUT_MS", "14"))

# Мягкий вход генерации после prefix-cache.
PREFIX_CACHE_GENERATION_FADE_IN_MS = int(
    os.getenv("FISH_PREFIX_CACHE_GENERATION_FADE_IN_MS", "6")
)
PREFIX_CACHE_FULL_COMMIT_MODE = _env_flag("FISH_PREFIX_CACHE_FULL_COMMIT_MODE", True)
PREFIX_CACHE_SKIP_ADJUST_MS = int(os.getenv("FISH_PREFIX_CACHE_SKIP_ADJUST_MS", "0"))
PREFIX_CACHE_DISABLE_PRELOAD_IN_FULL_COMMIT_MODE = _env_flag(
    "FISH_PREFIX_CACHE_DISABLE_PRELOAD_IN_FULL_COMMIT_MODE",
    True,
)

DEFAULT_REFERENCE_ID = _RUNTIME.proxy.default_reference_id
SESSION_TTL_SEC = _RUNTIME.proxy.session_ttl_sec
SESSION_MAX_COUNT = _RUNTIME.proxy.session_max_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("fish-proxy")

app = FastAPI(title="Fish Speech PCM Proxy", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SENTENCE_BOUNDARY_RE = re.compile(r'.+?[.!?…](?:["»”)\]]+)?(?:\s+|$)', re.S)
CLAUSE_BOUNDARY_RE = re.compile(r'.+?[,:;—\-](?:\s+|$)', re.S)

SHORT_COMPLETE_SENTENCE_MIN_CHARS = 24
FIRST_COMMIT_TTFA_MIN_CHARS = 6

DEFAULT_SESSION_CONFIG = _RUNTIME.proxy.model_dump(mode="python")


def _auth_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}

    if FISH_API_KEY:
        headers["Authorization"] = f"Bearer {FISH_API_KEY}"

    if extra:
        headers.update(extra)

    return headers


class SessionOpenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config_text: str = Field(..., min_length=2, max_length=100_000)


class SessionCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str


class SessionAppendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=1, max_length=4000)
    source_ts_ms: int | None = None
    cache: str | None = Field(None, min_length=1, max_length=4000)
    cash: str | None = Field(None, min_length=1, max_length=4000)


class PrefixCacheAddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config_text: str = Field(..., min_length=2, max_length=100_000)
    text: str | None = Field(None, min_length=1, max_length=4000)
    texts: list[str] | None = Field(None, min_length=1, max_length=4096)
    lookahead_text: str | None = Field(None, min_length=1, max_length=4000)
    full_text: str | None = Field(None, min_length=1, max_length=8000)
    prefix_cut_adjust_ms: int = Field(0, ge=-1000, le=1000)
    clear_existing: bool = False
    fail_fast: bool = False


class PrefixCacheKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config_text: str = Field(..., min_length=2, max_length=100_000)
    text: str = Field(..., min_length=1, max_length=4000)


class SessionFlushRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field("manual_flush", min_length=1, max_length=200)


class SessionFinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field("input_finished", min_length=1, max_length=200)


@dataclass
class IntroCacheEntry:
    key: str
    created_at: float
    expires_at: float
    audio_meta: dict[str, Any]
    pcm: bytes
    text: str
    codes: list[list[int]] | None = None
    code_frames: int = 0


@dataclass
class PrefixCacheEntry:
    key: str
    created_at: float
    expires_at: float
    audio_meta: dict[str, Any]
    pcm: bytes
    text: str
    codes: list[list[int]] | None = None
    code_frames: int = 0
    cache_mode: str = "standalone"
    generation_text: str | None = None
    lookahead_text: str | None = None
    full_pcm_bytes: int = 0
    prefix_audio_skip_bytes: int | None = None
    prefix_cut_adjust_ms: int = 0
    boundary_method: str = "standalone_duration"


class IntroCache:
    def __init__(self) -> None:
        self._items: dict[str, IntroCacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        *,
        key: str,
        text: str,
        config: ProxyConfig,
        client: httpx.AsyncClient,
    ) -> IntroCacheEntry | None:
        if not config.intro_cache.enabled or not text.strip():
            return None

        now = time.time()

        async with self._lock:
            entry = self._items.get(key)
            if entry and entry.expires_at > now:
                return entry

            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            lock = self._locks[key]

        async with lock:
            async with self._lock:
                entry = self._items.get(key)
                if entry and entry.expires_at > now:
                    return entry

            try:
                entry = await _generate_intro_cache_entry(
                    key=key,
                    text=text,
                    config=config,
                    client=client,
                )
            except Exception as exc:
                logger.error(
                    "failed to generate intro cache for key=%s: %s",
                    key[:8],
                    exc,
                )
                raise

            async with self._lock:
                if (
                    len(self._items) >= config.intro_cache.max_entries
                    and key not in self._items
                ):
                    oldest_key = min(
                        self._items.keys(),
                        key=lambda k: self._items[k].created_at,
                    )
                    self._items.pop(oldest_key, None)
                    self._locks.pop(oldest_key, None)

                self._items[key] = entry
                return entry

    async def clear(self) -> None:
        async with self._lock:
            self._items.clear()
            self._locks.clear()
            logger.info("intro cache cleared")

    def list_entries(self) -> list[IntroCacheEntry]:
        return list(self._items.values())

    def count(self) -> int:
        return len(self._items)


class PrefixCacheLibrary:
    def __init__(self, *, max_entries: int = PREFIX_CACHE_MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self._items: dict[str, PrefixCacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> PrefixCacheEntry | None:
        async with self._lock:
            return self._items.get(key)

    async def get_or_create(
        self,
        *,
        key: str,
        text: str,
        config: ProxyConfig,
        client: httpx.AsyncClient,
        lookahead_text: str | None = None,
        full_text: str | None = None,
        prefix_cut_adjust_ms: int = 0,
    ) -> tuple[PrefixCacheEntry, bool]:
        async with self._lock:
            entry = self._items.get(key)
            if entry is not None:
                return entry, False

            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            lock = self._locks[key]

        async with lock:
            async with self._lock:
                entry = self._items.get(key)
                if entry is not None:
                    return entry, False

            entry = await _generate_prefix_cache_entry(
                key=key,
                text=text,
                config=config,
                client=client,
                lookahead_text=lookahead_text,
                full_text=full_text,
                prefix_cut_adjust_ms=prefix_cut_adjust_ms,
            )

            async with self._lock:
                if len(self._items) >= self.max_entries and key not in self._items:
                    oldest_key = min(
                        self._items.keys(),
                        key=lambda k: self._items[k].created_at,
                    )
                    self._items.pop(oldest_key, None)
                    self._locks.pop(oldest_key, None)

                self._items[key] = entry
                return entry, True

    async def clear(self) -> None:
        async with self._lock:
            self._items.clear()
            self._locks.clear()
            logger.info("prefix cache cleared")

    async def list_entries(self) -> list[PrefixCacheEntry]:
        async with self._lock:
            return list(self._items.values())

    async def count(self) -> int:
        async with self._lock:
            return len(self._items)


@dataclass
class SessionRecord:
    session_id: str
    config: dict[str, Any]
    raw_config_text: str
    created_at: float
    updated_at: float
    expires_at: float
    append_chars: int = 0
    commit_chars: int = 0
    next_commit_seq: int = 1
    next_pcm_seq: int = 1
    buffer_text: str = ""
    input_closed: bool = False
    synthesis_session_id: str | None = None
    intro_preloaded_cache_key: str | None = None
    prefix_preloaded_cache_key: str | None = None
    pending_prefix_cache_text: str | None = None
    pending_prefix_cache_key: str | None = None
    audio_meta: dict[str, Any] | None = None
    stream_started: bool = False
    stream_finished: bool = False
    closed: bool = False
    buffer_started_at: float | None = None
    commit_timer_task: asyncio.Task | None = field(default=None, repr=False)
    chars_since_upstream_reset: int = 0
    last_upstream_reset_commit_seq: int = 0
    commit_history: list[dict[str, Any]] = field(default_factory=list)
    commit_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stream_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    def __init__(self) -> None:
        self._items: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()

    async def cleanup(self) -> None:
        now = time.time()
        expired_records: list[SessionRecord] = []

        async with self._lock:
            expired_ids = []

            for sid, rec in list(self._items.items()):
                should_expire = False

                if rec.expires_at <= now and not rec.stream_started:
                    should_expire = True
                elif rec.expires_at <= now and rec.stream_finished:
                    should_expire = True

                if should_expire:
                    expired_ids.append(sid)
                    expired_records.append(rec)

            for sid in expired_ids:
                self._items.pop(sid, None)

            if expired_ids:
                logger.info("session cleanup removed=%s", len(expired_ids))

        for rec in expired_records:
            rec.closed = True
            _cancel_commit_timer(rec)

            try:
                rec.commit_queue.put_nowait({"type": "abort"})
            except Exception:
                pass

            if rec.synthesis_session_id:
                _schedule_close_upstream_synthesis_session(
                    rec.synthesis_session_id,
                    reason="expired",
                )

    async def create(
        self,
        config: dict[str, Any],
        raw_config_text: str,
    ) -> SessionRecord:
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
            rec = self._items.pop(session_id, None)

        if rec is None:
            return False

        rec.closed = True
        _cancel_commit_timer(rec)

        try:
            rec.commit_queue.put_nowait({"type": "abort"})
        except Exception:
            pass

        return True

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
                        "buffer_chars": len(rec.buffer_text),
                        "append_chars": rec.append_chars,
                        "commit_chars": rec.commit_chars,
                        "input_closed": rec.input_closed,
                        "stream_started": rec.stream_started,
                        "stream_finished": rec.stream_finished,
                        "queued_commits": rec.commit_queue.qsize(),
                        "synthesis_session_id": rec.synthesis_session_id,
                        "pending_prefix_cache_text": rec.pending_prefix_cache_text,
                    }
                    for rec in self._items.values()
                ],
            }


session_store = SessionStore()
intro_cache = IntroCache()
prefix_cache_library = PrefixCacheLibrary()


def _schedule_close_upstream_synthesis_session(
    synthesis_session_id: str | None,
    *,
    reason: str,
) -> None:
    if not synthesis_session_id:
        return

    async def runner() -> None:
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                await _close_upstream_synthesis_session(
                    client,
                    synthesis_session_id,
                    reason=reason,
                )
        except Exception as exc:
            logger.warning(
                "failed to schedule-close upstream synthesis session %s reason=%s: %s",
                synthesis_session_id[:8],
                reason,
                exc,
            )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    loop.create_task(runner())


def _cancel_commit_timer(rec: SessionRecord) -> None:
    task = rec.commit_timer_task
    rec.commit_timer_task = None

    if task is not None and not task.done():
        task.cancel()


def _ensure_commit_timer(rec: SessionRecord) -> None:
    if rec.closed or rec.input_closed:
        return
    if not rec.buffer_text.strip():
        return
    if rec.commit_timer_task is not None and not rec.commit_timer_task.done():
        return

    rec.commit_timer_task = asyncio.create_task(_commit_timer_loop(rec.session_id))


def _extract_timed_commit(
    buffer_text: str,
    cfg: CommitPolicyConfig,
    *,
    next_commit_seq: int,
    elapsed_ms: int,
) -> tuple[list[tuple[str, str]], str]:
    if not buffer_text.strip():
        return [], ""

    stage = cfg.first if next_commit_seq == 1 else cfg.next
    text_len = len(buffer_text)
    is_first_commit = next_commit_seq == 1
    ttfa_min_chars = _first_commit_min_chars(stage)

    wait_min_chars = ttfa_min_chars if is_first_commit else stage.min_chars

    if elapsed_ms >= stage.max_wait_ms and text_len >= wait_min_chars:
        if is_first_commit:
            cut = _safe_partial_cut_for_ttfa(
                buffer_text,
                min_chars=ttfa_min_chars,
                max_chars=stage.max_chars,
                allow_inside_word=False,
            )

            if cut is None:
                if elapsed_ms < stage.allow_partial_after_ms:
                    return [], buffer_text
            else:
                end, reason = cut
                piece = _right_trim_committed(buffer_text[:end])
                remainder = _normalize_tail_after_commit(buffer_text[end:])

                if piece:
                    return [(piece, reason)], remainder

                if elapsed_ms < stage.allow_partial_after_ms:
                    return [], buffer_text

        else:
            boundary = _last_boundary_before(
                buffer_text,
                text_len,
                include_clause=cfg.flush_on_clause_punctuation,
                include_newline=cfg.flush_on_newline,
            )

            if boundary is not None:
                end, reason = boundary
            else:
                end = _safe_hard_cut(
                    buffer_text,
                    text_len,
                    min_keep=max(1, min(stage.min_chars, text_len)),
                )
                reason = "max_wait_timeout"

            piece = _right_trim_committed(buffer_text[:end])
            remainder = _normalize_tail_after_commit(buffer_text[end:])

            if piece:
                return [(piece, reason)], remainder

    if elapsed_ms >= stage.allow_partial_after_ms:
        if is_first_commit:
            cut = _safe_partial_cut_for_ttfa(
                buffer_text,
                min_chars=ttfa_min_chars,
                max_chars=stage.max_chars,
                allow_inside_word=True,
            )

            if cut is None:
                return [], buffer_text

            end, reason = cut
            piece = _right_trim_committed(buffer_text[:end])
            remainder = _normalize_tail_after_commit(buffer_text[end:])

            if piece:
                return [(piece, reason)], remainder

            return [], buffer_text

        piece = _right_trim_committed(buffer_text)
        if piece:
            return [(piece, "allow_partial_timeout")], ""

    return [], buffer_text


async def _commit_timer_loop(session_id: str) -> None:
    try:
        while True:
            rec = await session_store.get(session_id, touch=False)
            if rec is None or rec.closed or rec.input_closed:
                return

            sleep_sec = 0.1

            async with rec.lock:
                if rec.closed or rec.input_closed or not rec.buffer_text.strip():
                    rec.buffer_started_at = None
                    rec.commit_timer_task = None
                    return

                now = time.time()
                if rec.buffer_started_at is None:
                    rec.buffer_started_at = now

                config = ProxyConfig.model_validate(rec.config)
                stage = (
                    config.commit.first
                    if rec.next_commit_seq == 1
                    else config.commit.next
                )
                elapsed_ms = int((now - rec.buffer_started_at) * 1000)

                commits, remainder = _extract_timed_commit(
                    rec.buffer_text,
                    config.commit,
                    next_commit_seq=rec.next_commit_seq,
                    elapsed_ms=elapsed_ms,
                )

                if commits:
                    rec.buffer_text = remainder
                    await _queue_commits(rec, commits)
                    await _touch_session(rec)

                    if rec.buffer_text.strip():
                        rec.buffer_started_at = time.time()
                        continue

                    rec.buffer_started_at = None
                    rec.commit_timer_task = None
                    return

                next_ms = min(
                    max(25, stage.max_wait_ms - elapsed_ms),
                    max(25, stage.allow_partial_after_ms - elapsed_ms),
                )
                sleep_sec = max(0.025, next_ms / 1000.0)

            await asyncio.sleep(sleep_sec)

    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("commit timer failed session_id=%s", session_id)


def _parse_config_text(config_text: str) -> dict[str, Any]:
    try:
        payload = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, detail=f"invalid config json: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(400, detail="config json must be an object")

    return payload


def normalize_config(config_text: str) -> ProxyConfig:
    payload = _parse_config_text(config_text)

    try:
        return merge_frontend_proxy_override(payload, runtime=load_runtime_config())
    except Exception as exc:
        raise HTTPException(400, detail=str(exc)) from exc


def _normalized_reference_id(config: ProxyConfig) -> str:
    return (config.tts.reference_id or config.default_reference_id or "").strip()


def build_upstream_payload(
    text: str,
    config: ProxyConfig,
    *,
    commit_seq: int | None = None,
) -> dict[str, Any]:
    tts = config.tts
    reference_id = _normalized_reference_id(config)

    payload = {
        "text": text,
        "streaming": True,
        "stream_tokens": tts.stream_tokens,
        "initial_stream_chunk_size": tts.initial_stream_chunk_size,
        "stream_chunk_size": tts.stream_chunk_size,
        "reference_id": reference_id,
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

    if commit_seq == 1:
        if tts.first_initial_stream_chunk_size is not None:
            payload["initial_stream_chunk_size"] = tts.first_initial_stream_chunk_size

        if tts.first_stream_chunk_size is not None:
            payload["stream_chunk_size"] = tts.first_stream_chunk_size

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
                "<HHI",
                buf,
                chunk_data_start,
            )
            bits_per_sample = struct.unpack_from(
                "<H",
                buf,
                chunk_data_start + 14,
            )[0]

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


def _right_trim_committed(text: str) -> str:
    return text.rstrip()


def _normalize_tail_after_commit(text: str) -> str:
    return text.lstrip()


def _intro_cache_key(config: ProxyConfig) -> str:
    tts = config.tts

    payload = {
        "cache_schema": PREFIX_CACHE_SCHEMA_VERSION,
        "prefix_cache_salt": PREFIX_CACHE_SALT,
        "proxy_version": config.version,
        "runtime_version": _RUNTIME.version,
        "llama_checkpoint_path": _RUNTIME.paths.llama_checkpoint_path,
        "decoder_checkpoint_path": _RUNTIME.paths.decoder_checkpoint_path,
        "decoder_config_name": _RUNTIME.paths.decoder_config_name,
        "model_device": _RUNTIME.model.device,
        "model_precision": _RUNTIME.model.precision,
        "model_compile": _RUNTIME.model.compile,
        "upstream_url": UPSTREAM_TTS_URL,
        "intro_text": config.intro_cache.text,
        "reference_id": _normalized_reference_id(config),
        "format": tts.format,
        "seed": tts.seed,
        "normalize": tts.normalize,
        "use_memory_cache": tts.use_memory_cache,
        "max_new_tokens": tts.max_new_tokens,
        "chunk_length": tts.chunk_length,
        "top_p": tts.top_p,
        "repetition_penalty": tts.repetition_penalty,
        "temperature": tts.temperature,
        "stream_tokens": True,
        "initial_stream_chunk_size": tts.initial_stream_chunk_size,
        "stream_chunk_size": tts.stream_chunk_size,
        "first_initial_stream_chunk_size": tts.first_initial_stream_chunk_size,
        "first_stream_chunk_size": tts.first_stream_chunk_size,
        "stateful_synthesis": tts.stateful_synthesis,
        "stateful_history_turns": tts.stateful_history_turns,
        "stateful_history_chars": tts.stateful_history_chars,
        "stateful_history_code_frames": tts.stateful_history_code_frames,
    }

    dump = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


def _prefix_cache_key(config: ProxyConfig, text: str) -> str:
    tts = config.tts
    prefix_text = text.strip()

    payload = {
        "cache_schema": PREFIX_CACHE_SCHEMA_VERSION,
        "cache_kind": "prefix_cache",
        "prefix_cache_salt": PREFIX_CACHE_SALT,
        "proxy_version": config.version,
        "runtime_version": _RUNTIME.version,
        "llama_checkpoint_path": _RUNTIME.paths.llama_checkpoint_path,
        "decoder_checkpoint_path": _RUNTIME.paths.decoder_checkpoint_path,
        "decoder_config_name": _RUNTIME.paths.decoder_config_name,
        "model_device": _RUNTIME.model.device,
        "model_precision": _RUNTIME.model.precision,
        "model_compile": _RUNTIME.model.compile,
        "upstream_url": UPSTREAM_TTS_URL,
        "prefix_text": prefix_text,
        "reference_id": _normalized_reference_id(config),
        "format": "wav",
        "seed": tts.seed,
        "normalize": tts.normalize,
        "use_memory_cache": tts.use_memory_cache,
        "max_new_tokens": tts.max_new_tokens,
        "chunk_length": tts.chunk_length,
        "top_p": tts.top_p,
        "repetition_penalty": tts.repetition_penalty,
        "temperature": tts.temperature,
        "streaming": True,
        "stream_tokens": True,
        "initial_stream_chunk_size": tts.initial_stream_chunk_size,
        "stream_chunk_size": tts.stream_chunk_size,
        "first_initial_stream_chunk_size": tts.first_initial_stream_chunk_size,
        "first_stream_chunk_size": tts.first_stream_chunk_size,
        "stateful_synthesis": tts.stateful_synthesis,
        "stateful_history_turns": tts.stateful_history_turns,
        "stateful_history_chars": tts.stateful_history_chars,
        "stateful_history_code_frames": tts.stateful_history_code_frames,
    }

    dump = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()


async def _generate_intro_cache_entry(
    *,
    key: str,
    text: str,
    config: ProxyConfig,
    client: httpx.AsyncClient,
) -> IntroCacheEntry:
    payload = build_upstream_payload(text=text, config=config, commit_seq=1)
    payload["streaming"] = True
    payload["stream_tokens"] = True
    payload["format"] = "wav"

    url = UPSTREAM_TTS_URL.replace(
        "/v1/tts",
        "/v1/synthesis/intro-cache/generate",
    )

    logger.info(
        "generating intro cache key=%s text=%r ref=%s url=%s",
        key[:8],
        text[:100],
        payload["reference_id"],
        url,
    )

    resp = await client.post(
        url,
        json=payload,
        headers=_auth_headers({"Accept": "application/json"}),
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"intro generation failed status={resp.status_code}: {resp.text}"
        )

    data = resp.json()
    pcm_b64 = data.get("pcm_b64") or ""
    audio_meta = data.get("audio_meta") or {}
    codes = data.get("codes")
    code_frames = int(data.get("code_frames") or 0)

    if not pcm_b64:
        raise RuntimeError("intro generation returned empty pcm_b64")

    pcm = base64.b64decode(pcm_b64)

    if not pcm:
        raise RuntimeError("intro generation returned empty PCM")

    if config.tts.stateful_synthesis and (not codes or code_frames <= 0):
        raise RuntimeError(
            "intro generation returned PCM but no continuation codes; "
            "seamless cached intro requires codes"
        )

    now = time.time()
    return IntroCacheEntry(
        key=key,
        created_at=now,
        expires_at=now + config.intro_cache.ttl_sec,
        audio_meta=audio_meta,
        pcm=pcm,
        text=text,
        codes=codes,
        code_frames=code_frames,
    )


async def _generate_prefix_cache_entry(
    *,
    key: str,
    text: str,
    config: ProxyConfig,
    client: httpx.AsyncClient,
    lookahead_text: str | None = None,
    full_text: str | None = None,
    prefix_cut_adjust_ms: int = 0,
) -> PrefixCacheEntry:
    cache_mode = "standalone"
    generation_text = text
    lookahead = lookahead_text.strip() if lookahead_text else None
    explicit_full_text = full_text.strip() if full_text else None

    if explicit_full_text:
        generation_text = explicit_full_text
        cache_mode = "lookahead"
    elif lookahead:
        generation_text = _join_prefix_and_tail(text, lookahead)
        cache_mode = "lookahead"

    payload = build_upstream_payload(text=text, config=config, commit_seq=1)
    payload["streaming"] = True
    payload["stream_tokens"] = True
    payload["format"] = "wav"

    url = UPSTREAM_TTS_URL.replace(
        "/v1/tts",
        "/v1/synthesis/intro-cache/generate",
    )

    logger.info(
        "generating prefix cache key=%s text=%r mode=%s generation_text=%r ref=%s url=%s",
        key[:8],
        text[:100],
        cache_mode,
        generation_text[:100],
        payload["reference_id"],
        url,
    )

    async def generate_cache_payload(payload_text: str) -> tuple[bytes, dict[str, Any], list[list[int]] | None, int]:
        request_payload = dict(payload)
        request_payload["text"] = payload_text

        resp = await client.post(
            url,
            json=request_payload,
            headers=_auth_headers({"Accept": "application/json"}),
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"prefix-cache generation failed status={resp.status_code}: {resp.text}"
            )

        data = resp.json()
        pcm_b64 = data.get("pcm_b64") or ""
        audio_meta = data.get("audio_meta") or {}
        codes = data.get("codes")
        code_frames = int(data.get("code_frames") or 0)

        if not pcm_b64:
            raise RuntimeError("prefix-cache generation returned empty pcm_b64")

        pcm = base64.b64decode(pcm_b64)

        if not pcm:
            raise RuntimeError("prefix-cache generation returned empty PCM")

        return pcm, audio_meta, codes, code_frames

    pcm, audio_meta, codes, code_frames = await generate_cache_payload(text)
    full_pcm_bytes = len(pcm)
    prefix_audio_skip_bytes: int | None = len(pcm)
    boundary_method = "standalone_duration"

    if cache_mode == "lookahead":
        prefix_pcm = pcm
        prefix_audio_meta = audio_meta
        prefix_codes = codes
        prefix_code_frames = code_frames
        full_pcm, full_audio_meta, _full_codes, _full_code_frames = (
            await generate_cache_payload(generation_text)
        )

        if prefix_audio_meta != full_audio_meta:
            raise RuntimeError(
                "lookahead prefix-cache audio format mismatch: "
                f"prefix={prefix_audio_meta}, full={full_audio_meta}"
            )

        sample_rate = int(prefix_audio_meta["sample_rate"])
        channels = int(prefix_audio_meta["channels"])
        sample_width = int(prefix_audio_meta["sample_width"])
        frame_bytes = channels * sample_width
        adjust_bytes = _pcm_bytes_for_duration_ms_allow_negative(
            sample_rate,
            channels,
            sample_width * 8,
            prefix_cut_adjust_ms,
        )
        cut_bytes = max(0, len(prefix_pcm) + adjust_bytes)
        cut_bytes = min(len(full_pcm), _align_down(cut_bytes, frame_bytes))

        if cut_bytes <= 0:
            raise RuntimeError("lookahead prefix-cache produced empty prefix PCM")

        pcm = full_pcm[:cut_bytes]
        audio_meta = full_audio_meta
        codes = prefix_codes
        code_frames = prefix_code_frames
        full_pcm_bytes = len(full_pcm)
        prefix_audio_skip_bytes = cut_bytes
        boundary_method = "lookahead_full_phrase_standalone_duration"

    if config.tts.stateful_synthesis and (not codes or code_frames <= 0):
        raise RuntimeError(
            "prefix-cache generation returned PCM but no continuation codes; "
            "seamless prefix-cache requires codes"
        )

    now = time.time()
    return PrefixCacheEntry(
        key=key,
        created_at=now,
        expires_at=float("inf"),
        audio_meta=audio_meta,
        pcm=pcm,
        text=text,
        codes=codes,
        code_frames=code_frames,
        cache_mode=cache_mode,
        generation_text=generation_text,
        lookahead_text=lookahead,
        full_pcm_bytes=full_pcm_bytes,
        prefix_audio_skip_bytes=prefix_audio_skip_bytes,
        prefix_cut_adjust_ms=prefix_cut_adjust_ms,
        boundary_method=boundary_method,
    )


async def _preload_intro_cache_context(
    *,
    rec: SessionRecord,
    config: ProxyConfig,
    entry: IntroCacheEntry | PrefixCacheEntry,
    client: httpx.AsyncClient,
    commit_seq: int = 0,
    commit_reason: str = "intro_cache",
    cache_kind: str = "intro_cache",
) -> bool:
    if not config.tts.stateful_synthesis:
        return False

    if not rec.synthesis_session_id:
        raise RuntimeError(
            "cannot emit seamless cache: upstream synthesis session is missing"
        )

    preloaded_key = (
        rec.prefix_preloaded_cache_key
        if cache_kind == "prefix_cache"
        else rec.intro_preloaded_cache_key
    )

    if preloaded_key == entry.key:
        return True

    if not entry.codes or entry.code_frames <= 0:
        raise RuntimeError("cannot preload cache context: cached entry has no codes")

    url = UPSTREAM_TTS_URL.replace(
        "/v1/tts",
        "/v1/synthesis/sessions/preload",
    )

    payload = {
        "synthesis_session_id": rec.synthesis_session_id,
        "text": entry.text,
        "codes": entry.codes,
        "commit_seq": commit_seq,
        "commit_reason": commit_reason,
    }

    resp = await client.post(
        url,
        json=payload,
        headers=_auth_headers({"Accept": "application/json"}),
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"{cache_kind} context preload failed status={resp.status_code}: {resp.text}"
        )

    if cache_kind == "prefix_cache":
        rec.prefix_preloaded_cache_key = entry.key
    else:
        rec.intro_preloaded_cache_key = entry.key

    logger.info(
        "%s context preloaded session=%s synthesis=%s key=%s code_frames=%s",
        cache_kind,
        rec.session_id[:8],
        rec.synthesis_session_id[:8],
        entry.key[:8],
        entry.code_frames,
    )

    return True


async def emit_intro_cache(
    *,
    rec: SessionRecord,
    config: ProxyConfig,
    client: httpx.AsyncClient,
    req_id: str,
) -> AsyncGenerator[bytes, None]:
    async def emit(obj: dict[str, Any]) -> bytes:
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    key = _intro_cache_key(config)
    entry = await intro_cache.get_or_create(
        key=key,
        text=config.intro_cache.text,
        config=config,
        client=client,
    )

    if entry is None:
        return

    preloaded = await _preload_intro_cache_context(
        rec=rec,
        config=config,
        entry=entry,
        client=client,
    )

    if preloaded:
        yield await emit(
            {
                "type": "intro_context_preloaded",
                "req_id": req_id,
                "session_id": rec.session_id,
                "synthesis_session_id": rec.synthesis_session_id,
                "cache_key": entry.key,
                "code_frames": entry.code_frames,
            }
        )

    if rec.audio_meta is not None:
        for k, v in entry.audio_meta.items():
            if rec.audio_meta.get(k) != v:
                logger.warning("intro meta mismatch session=%s", rec.session_id[:8])
                yield await emit(
                    {
                        "type": "intro_error",
                        "session_id": rec.session_id,
                        "message": "intro audio format mismatch with session format",
                    }
                )
                return

    yield await emit(
        {
            "type": "intro_start",
            "req_id": req_id,
            "session_id": rec.session_id,
            "cache_key": entry.key,
            "text_preview": entry.text[:180],
            "text_len": len(entry.text),
        }
    )

    if rec.audio_meta is None:
        rec.audio_meta = entry.audio_meta
        yield await emit(
            {
                "type": "meta",
                "session_id": rec.session_id,
                "commit_seq": 0,
                **rec.audio_meta,
            }
        )

    pcm = entry.pcm
    frame_bytes = rec.audio_meta["channels"] * rec.audio_meta["sample_width"]
    chunk_size = _align_down(config.intro_cache.emit_bytes, frame_bytes)
    if chunk_size <= 0:
        chunk_size = frame_bytes

    total_emitted = 0
    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i : i + chunk_size]
        if not chunk:
            continue

        pcm_seq = rec.next_pcm_seq
        rec.next_pcm_seq += 1
        total_emitted += len(chunk)

        yield await emit(
            {
                "type": "pcm",
                "req_id": req_id,
                "session_id": rec.session_id,
                "commit_seq": 0,
                "seq": pcm_seq,
                "first_pcm_for_commit": (i == 0),
                "intro": True,
                "data": base64.b64encode(chunk).decode("ascii"),
            }
        )
        await asyncio.sleep(0)

    if config.intro_cache.pause_after_ms > 0:
        pause_ms = config.intro_cache.pause_after_ms
        silence = _pcm_silence_bytes(
            sample_rate=rec.audio_meta["sample_rate"],
            channels=rec.audio_meta["channels"],
            bits_per_sample=rec.audio_meta["sample_width"] * 8,
            duration_ms=pause_ms,
        )
        if silence:
            pcm_seq = rec.next_pcm_seq
            rec.next_pcm_seq += 1
            yield await emit(
                {
                    "type": "pcm",
                    "req_id": req_id,
                    "session_id": rec.session_id,
                    "commit_seq": 0,
                    "seq": pcm_seq,
                    "first_pcm_for_commit": False,
                    "intro": True,
                    "data": base64.b64encode(silence).decode("ascii"),
                }
            )

        yield await emit(
            {
                "type": "pause",
                "req_id": req_id,
                "session_id": rec.session_id,
                "commit_seq": 0,
                "boundary": "intro",
                "pause_ms": pause_ms,
            }
        )

    yield await emit(
        {
            "type": "intro_done",
            "req_id": req_id,
            "session_id": rec.session_id,
            "cache_key": entry.key,
            "pcm_bytes": total_emitted,
        }
    )


def _resolve_prefix_cache_alias(
    cache: str | None,
    cash: str | None,
) -> str | None:
    cache_text = cache.strip() if cache is not None else None
    cash_text = cash.strip() if cash is not None else None

    if cache_text is not None and cash_text is not None and cache_text != cash_text:
        raise HTTPException(400, detail="cache and cash must match when both are set")

    return cache_text if cache_text is not None else cash_text


def _join_prefix_and_tail(prefix: str, tail: str) -> str:
    prefix = prefix or ""
    tail = tail or ""

    if not prefix:
        return tail
    if not tail:
        return prefix
    if prefix[-1].isspace() or tail[0].isspace():
        return prefix + tail
    if tail[0] in ".,!?;:…)]}»”":
        return prefix + tail
    return f"{prefix} {tail}"


async def _get_prefix_cache_entry_for_commit(
    config: ProxyConfig,
    commit_item: dict[str, Any],
) -> PrefixCacheEntry:
    cache_text = (commit_item.get("prefix_cache_text") or "").strip()
    if not cache_text:
        raise RuntimeError("prefix-cache commit is missing prefix_cache_text")

    cache_key = commit_item.get("prefix_cache_key") or _prefix_cache_key(
        config,
        cache_text,
    )

    entry = await prefix_cache_library.get(cache_key)
    if entry is None:
        raise RuntimeError(
            f"prefix-cache entry not found for text={cache_text!r} key={cache_key[:8]}"
        )

    return entry


def _prefix_cache_public_item(entry: PrefixCacheEntry) -> dict[str, Any]:
    return {
        "key": entry.key,
        "key_short": entry.key[:8],
        "text": entry.text,
        "pcm_bytes": len(entry.pcm),
        "code_frames": entry.code_frames,
        "audio_meta": entry.audio_meta,
        "cache_mode": entry.cache_mode,
        "generation_text": entry.generation_text,
        "lookahead_text": entry.lookahead_text,
        "full_pcm_bytes": entry.full_pcm_bytes,
        "prefix_audio_skip_bytes": entry.prefix_audio_skip_bytes,
        "prefix_cut_adjust_ms": entry.prefix_cut_adjust_ms,
        "boundary_method": entry.boundary_method,
    }


def _prefix_cache_event_base(
    *,
    req_id: str,
    rec: SessionRecord,
    commit_seq: int,
    entry: PrefixCacheEntry,
) -> dict[str, Any]:
    return {
        "req_id": req_id,
        "session_id": rec.session_id,
        "commit_seq": commit_seq,
        "cache_key": entry.key,
        "cache_key_short": entry.key[:8],
        "cache_text": entry.text,
    }


async def emit_prefix_cache(
    *,
    rec: SessionRecord,
    config: ProxyConfig,
    client: httpx.AsyncClient,
    req_id: str,
    commit_item: dict[str, Any],
    entry: PrefixCacheEntry,
    full_commit_mode: bool,
    preload_context: bool,
) -> AsyncGenerator[bytes, None]:
    async def emit(obj: dict[str, Any]) -> bytes:
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    commit_seq = int(commit_item["seq"])
    preloaded = False
    if preload_context:
        preloaded = await _preload_intro_cache_context(
            rec=rec,
            config=config,
            entry=entry,
            client=client,
            commit_seq=0,
            commit_reason="prefix_cache",
            cache_kind="prefix_cache",
        )

    event_base = _prefix_cache_event_base(
        req_id=req_id,
        rec=rec,
        commit_seq=commit_seq,
        entry=entry,
    )

    if preloaded:
        yield await emit(
            {
                "type": "prefix_cache_context_preloaded",
                **event_base,
                "synthesis_session_id": rec.synthesis_session_id,
                "code_frames": entry.code_frames,
            }
        )

    if rec.audio_meta is not None:
        for k, v in entry.audio_meta.items():
            if rec.audio_meta.get(k) != v:
                logger.warning(
                    "prefix-cache meta mismatch session=%s",
                    rec.session_id[:8],
                )
                raise RuntimeError(
                    "prefix-cache audio format mismatch with session format"
                )

    yield await emit(
        {
            "type": "prefix_cache_start",
            **event_base,
            "text_preview": entry.text[:180],
            "text_len": len(entry.text),
            "code_frames": entry.code_frames,
            "pcm_bytes": len(entry.pcm),
            "full_commit_mode": full_commit_mode,
            "preload_context": preload_context,
            "pause_after_ms": PREFIX_CACHE_PAUSE_AFTER_MS,
            "fade_out_ms": PREFIX_CACHE_FADE_OUT_MS,
        }
    )

    if rec.audio_meta is None:
        rec.audio_meta = entry.audio_meta
        yield await emit(
            {
                "type": "meta",
                "session_id": rec.session_id,
                "commit_seq": commit_seq,
                **rec.audio_meta,
            }
        )

    pcm = bytes(entry.pcm)

    sample_rate = int(rec.audio_meta["sample_rate"])
    channels = int(rec.audio_meta["channels"])
    sample_width = int(rec.audio_meta["sample_width"])
    bits_per_sample = sample_width * 8
    frame_bytes = channels * sample_width

    if PREFIX_CACHE_FADE_OUT_MS > 0 and pcm:
        fade_out_bytes = _align_down(
            _pcm_bytes_for_duration_ms(
                sample_rate,
                channels,
                bits_per_sample,
                PREFIX_CACHE_FADE_OUT_MS,
            ),
            frame_bytes,
        )

        tail_len = min(len(pcm), fade_out_bytes)
        tail_len = _align_down(tail_len, frame_bytes)

        if tail_len > 0:
            start = len(pcm) - tail_len
            buf = bytearray(pcm)
            buf[start:] = _apply_pcm16_fade(
                bytes(buf[start:]),
                channels=channels,
                fade_in_frames=0,
                fade_out_frames=int(sample_rate * PREFIX_CACHE_FADE_OUT_MS / 1000.0),
            )
            pcm = bytes(buf)

    if PREFIX_CACHE_PAUSE_AFTER_MS > 0:
        pcm += _pcm_silence_bytes(
            sample_rate=sample_rate,
            channels=channels,
            bits_per_sample=bits_per_sample,
            duration_ms=PREFIX_CACHE_PAUSE_AFTER_MS,
        )

    chunk_size = _align_down(PREFIX_CACHE_EMIT_BYTES, frame_bytes)
    if chunk_size <= 0:
        chunk_size = frame_bytes

    total_emitted = 0
    for i in range(0, len(pcm), chunk_size):
        chunk = pcm[i : i + chunk_size]
        if not chunk:
            continue

        pcm_seq = rec.next_pcm_seq
        rec.next_pcm_seq += 1
        total_emitted += len(chunk)

        yield await emit(
            {
                "type": "pcm",
                **event_base,
                "seq": pcm_seq,
                "first_pcm_for_commit": (i == 0),
                "prefix_cache": True,
                "pcm_bytes": len(chunk),
                "data": base64.b64encode(chunk).decode("ascii"),
            }
        )
        await asyncio.sleep(0)

    yield await emit(
        {
            "type": "prefix_cache_done",
            **event_base,
            "code_frames": entry.code_frames,
            "pcm_bytes": total_emitted,
            "full_commit_mode": full_commit_mode,
            "preload_context": preload_context,
            "pause_after_ms": PREFIX_CACHE_PAUSE_AFTER_MS,
            "fade_out_ms": PREFIX_CACHE_FADE_OUT_MS,
        }
    )


def _safe_hard_cut(text: str, limit: int, min_keep: int) -> int:
    if len(text) <= limit:
        return len(text)

    search_area = text[:limit]
    candidates = [
        search_area.rfind(" "),
        search_area.rfind("\n"),
        search_area.rfind("\t"),
    ]
    cut = max(candidates)

    if cut >= max(1, min_keep):
        return cut + 1

    return limit


def _first_commit_min_chars(stage) -> int:
    return max(1, min(stage.min_chars, FIRST_COMMIT_TTFA_MIN_CHARS))


def _safe_partial_cut_for_ttfa(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    allow_inside_word: bool = False,
) -> tuple[int, str] | None:
    if not text:
        return None

    min_chars = max(1, int(min_chars))
    limit = max(1, min(len(text), int(max_chars)))
    prefix = text[:limit]

    if limit < min_chars and not allow_inside_word:
        return None

    punctuation_candidates: list[int] = []
    for match in SENTENCE_BOUNDARY_RE.finditer(prefix):
        punctuation_candidates.append(match.end())
    for match in CLAUSE_BOUNDARY_RE.finditer(prefix):
        punctuation_candidates.append(match.end())

    valid_punctuation = [
        end for end in punctuation_candidates if min_chars <= end <= limit
    ]
    if valid_punctuation:
        return min(valid_punctuation), "ttfa_boundary"

    for match in re.finditer(r"\s+", prefix):
        end = match.end()
        if min_chars <= end <= limit:
            return end, "ttfa_boundary"

    if allow_inside_word:
        end = _safe_hard_cut(text, limit, min_keep=min_chars)
        if end > 0:
            return end, "ttfa_partial"

    return None


def _pcm_bytes_for_duration_ms(
    sample_rate: int,
    channels: int,
    bits_per_sample: int,
    duration_ms: int,
) -> int:
    if duration_ms <= 0:
        return 0

    bytes_per_second = sample_rate * channels * (bits_per_sample // 8)
    raw = int(bytes_per_second * (duration_ms / 1000.0))

    if raw % 2 != 0:
        raw += 1

    return raw


def _pcm_bytes_for_duration_ms_allow_negative(
    sample_rate: int,
    channels: int,
    bits_per_sample: int,
    duration_ms: int,
) -> int:
    if duration_ms == 0:
        return 0

    bytes_per_second = sample_rate * channels * (bits_per_sample // 8)
    raw = int(bytes_per_second * (duration_ms / 1000.0))

    if raw % 2 != 0:
        raw += 1 if raw > 0 else -1

    return raw


def _pcm_ms_estimate(
    pcm_bytes: int,
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> float:
    bytes_per_second = sample_rate * channels * sample_width
    if bytes_per_second <= 0:
        return 0.0
    return round((pcm_bytes / bytes_per_second) * 1000.0, 2)


def _align_down(value: int, frame_bytes: int) -> int:
    if frame_bytes <= 0:
        return value
    return value - (value % frame_bytes)


def _pcm_silence_bytes(
    sample_rate: int,
    channels: int,
    bits_per_sample: int,
    duration_ms: int,
) -> bytes:
    if duration_ms <= 0:
        return b""

    frame_bytes = channels * (bits_per_sample // 8)
    frames = int(sample_rate * duration_ms / 1000.0)
    return b"\x00" * (frames * frame_bytes)


SENTENCE_END_RE = re.compile(r'[.!?…]+(?:["»”)\]]+)?\s*$')
CLAUSE_END_RE = re.compile(r'[,;:—–-]+(?:["»”)\]]+)?\s*$')


def _classify_commit_boundary(text: str, reason: str) -> str:
    stripped = text.rstrip()

    if reason == "newline":
        return "newline"

    if "\n" in text[-3:]:
        return "newline"

    if SENTENCE_END_RE.search(stripped):
        return "sentence"

    if CLAUSE_END_RE.search(stripped):
        return "clause"

    if reason in {"force", "manual_flush", "input_finished", "llm_input_finished"}:
        return "force"

    if reason.startswith("hard_limit"):
        return "hard_limit"

    if reason == "sentence":
        return "sentence"

    if reason == "clause":
        return "clause"

    return "soft"


def _pause_ms_for_commit(text: str, reason: str, playback) -> tuple[str, int]:
    if not playback.punctuation_pauses_enabled:
        return "disabled", 0

    boundary = _classify_commit_boundary(text, reason)

    if boundary == "newline":
        return boundary, playback.pause_after_newline_ms
    if boundary == "sentence":
        return boundary, playback.pause_after_sentence_ms
    if boundary == "clause":
        return boundary, playback.pause_after_clause_ms
    if boundary == "force":
        return boundary, playback.pause_after_force_ms
    if boundary == "hard_limit":
        return boundary, playback.pause_after_hard_limit_ms

    return boundary, 0


def _apply_pcm16_fade(
    pcm: bytes,
    *,
    channels: int,
    fade_in_frames: int = 0,
    fade_out_frames: int = 0,
) -> bytes:
    if not pcm:
        return pcm

    if channels <= 0:
        return pcm

    frame_bytes = channels * 2
    usable_len = len(pcm) - (len(pcm) % frame_bytes)

    if usable_len <= 0:
        return pcm

    out = bytearray(pcm)
    frame_count = usable_len // frame_bytes

    def clamp_i16(value: float) -> int:
        ivalue = int(round(value))
        if ivalue > 32767:
            return 32767
        if ivalue < -32768:
            return -32768
        return ivalue

    def scale_frame(frame_index: int, gain: float) -> None:
        base = frame_index * frame_bytes
        for ch in range(channels):
            offset = base + ch * 2
            sample = struct.unpack_from("<h", out, offset)[0]
            struct.pack_into("<h", out, offset, clamp_i16(sample * gain))

    if fade_in_frames > 0:
        n = min(frame_count, fade_in_frames)
        if n == 1:
            scale_frame(0, 1.0)
        elif n > 1:
            denom = n - 1
            for i in range(n):
                scale_frame(i, i / denom)

    if fade_out_frames > 0:
        n = min(frame_count, fade_out_frames)
        if n == 1:
            scale_frame(frame_count - 1, 0.0)
        elif n > 1:
            denom = n - 1
            start = frame_count - n
            for i in range(n):
                scale_frame(start + i, 1.0 - (i / denom))

    return bytes(out)


def _last_boundary_before(
    text: str,
    limit: int,
    *,
    include_clause: bool,
    include_newline: bool,
) -> tuple[int, str] | None:
    candidates: list[tuple[int, str]] = []

    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end()
        if end <= limit:
            candidates.append((end, "sentence"))

    if include_clause:
        for match in CLAUSE_BOUNDARY_RE.finditer(text):
            end = match.end()
            if end <= limit:
                candidates.append((end, "clause"))

    if include_newline:
        idx = 0
        while True:
            idx = text.find("\n", idx)
            if idx == -1:
                break

            end = idx + 1
            if end <= limit:
                candidates.append((end, "newline"))

            idx = end

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def _extract_commits(
    buffer_text: str,
    cfg: CommitPolicyConfig,
    *,
    next_commit_seq: int,
    force: bool = False,
) -> tuple[list[tuple[str, str]], str]:
    commits: list[tuple[str, str]] = []
    text = buffer_text

    while True:
        text_len = len(text)

        if text_len == 0:
            break

        is_first_commit = (next_commit_seq + len(commits)) == 1
        stage = cfg.first if is_first_commit else cfg.next

        if force:
            piece = _right_trim_committed(text)
            if piece:
                commits.append((piece, "force"))
            text = ""
            break

        if is_first_commit:
            cut = _safe_partial_cut_for_ttfa(
                text,
                min_chars=_first_commit_min_chars(stage),
                max_chars=stage.max_chars,
                allow_inside_word=False,
            )

            if cut is None:
                break

            end, reason = cut
            piece = _right_trim_committed(text[:end])
            text = _normalize_tail_after_commit(text[end:])

            if piece:
                commits.append((piece, reason))
                continue

            break

        if (
            cfg.flush_on_sentence_punctuation
            and text_len < stage.min_chars
            and text_len >= SHORT_COMPLETE_SENTENCE_MIN_CHARS
        ):
            sentence_boundary = _last_boundary_before(
                text,
                text_len,
                include_clause=False,
                include_newline=cfg.flush_on_newline,
            )

            if sentence_boundary and sentence_boundary[0] == text_len:
                piece = _right_trim_committed(text)
                if piece:
                    commits.append((piece, "sentence"))
                text = ""
                break

        if text_len < stage.min_chars:
            break

        reason = None
        end = None

        sentence_boundary = _last_boundary_before(
            text,
            stage.target_chars,
            include_clause=False,
            include_newline=cfg.flush_on_newline,
        )

        if sentence_boundary and cfg.flush_on_sentence_punctuation:
            end, reason = sentence_boundary

        if end is None and text_len >= stage.target_chars:
            any_boundary = _last_boundary_before(
                text,
                stage.max_chars,
                include_clause=cfg.flush_on_clause_punctuation,
                include_newline=cfg.flush_on_newline,
            )

            if any_boundary:
                end, reason = any_boundary

        if end is None and text_len >= stage.max_chars:
            hard_boundary = _last_boundary_before(
                text,
                stage.max_chars,
                include_clause=cfg.flush_on_clause_punctuation,
                include_newline=cfg.flush_on_newline,
            )

            if hard_boundary:
                end, reason = hard_boundary
            else:
                end = _safe_hard_cut(
                    text,
                    stage.max_chars,
                    min_keep=max(1, min(stage.min_chars, stage.max_chars)),
                )
                reason = "hard_limit_word_safe"

        if end is None:
            break

        piece = _right_trim_committed(text[:end])
        text = _normalize_tail_after_commit(text[end:])

        if piece:
            commits.append((piece, reason or "unknown"))
        else:
            break

    return commits, text


async def _open_upstream_synthesis_session(
    client: httpx.AsyncClient,
    rec: SessionRecord,
    config: ProxyConfig,
) -> str:
    open_payload = {
        "reference_id": _normalized_reference_id(config),
        "max_history_turns": config.tts.stateful_history_turns,
        "max_history_chars": config.tts.stateful_history_chars,
        "max_history_code_frames": config.tts.stateful_history_code_frames,
    }

    resp = await client.post(
        UPSTREAM_TTS_URL.replace("/v1/tts", "/v1/synthesis/sessions/open"),
        json=open_payload,
        headers=_auth_headers({"Accept": "application/json"}),
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"upstream synthesis open failed status={resp.status_code}: {resp.text}"
        )

    data = resp.json()
    synthesis_session_id = data.get("synthesis_session_id")
    if not synthesis_session_id:
        raise RuntimeError("upstream did not return synthesis_session_id")

    return synthesis_session_id


async def _close_upstream_synthesis_session(
    client: httpx.AsyncClient,
    synthesis_session_id: str | None,
    *,
    reason: str = "close",
) -> None:
    if not synthesis_session_id:
        return

    try:
        await client.post(
            UPSTREAM_TTS_URL.replace("/v1/tts", "/v1/synthesis/sessions/close"),
            json=synthesis_session_id,
            headers=_auth_headers({"Accept": "application/json"}),
        )
    except Exception as exc:
        logger.warning(
            "failed to close upstream synthesis session %s reason=%s: %s",
            synthesis_session_id[:8],
            reason,
            exc,
        )


def _should_reset_upstream_session(
    rec: SessionRecord,
    config: ProxyConfig,
    commit_item: dict[str, Any],
) -> tuple[bool, str]:
    if not config.tts.stateful_synthesis:
        return False, ""

    if not rec.synthesis_session_id:
        return False, ""

    seq = int(commit_item.get("seq") or 0)
    if seq <= 1:
        return False, ""

    every_commits = int(getattr(config.tts, "stateful_reset_every_commits", 0) or 0)
    every_chars = int(getattr(config.tts, "stateful_reset_every_chars", 0) or 0)

    if every_commits > 0 and (seq - 1) % every_commits == 0:
        return True, f"every_{every_commits}_commits"

    if every_chars > 0 and rec.chars_since_upstream_reset >= every_chars:
        return True, f"after_{every_chars}_chars"

    return False, ""


async def _touch_session(rec: SessionRecord) -> None:
    now = time.time()
    rec.updated_at = now
    rec.expires_at = now + SESSION_TTL_SEC


async def _queue_commits(
    rec: SessionRecord,
    commits: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    items = []
    now = time.time()

    for text, reason in commits:
        seq = rec.next_commit_seq
        rec.next_commit_seq += 1
        rec.commit_chars += len(text)

        item = {
            "seq": seq,
            "text": text,
            "reason": reason,
            "created_at": now,
            "text_len": len(text),
        }

        if seq == 1 and rec.pending_prefix_cache_text:
            item["prefix_cache_text"] = rec.pending_prefix_cache_text
            item["prefix_cache_key"] = rec.pending_prefix_cache_key
            rec.pending_prefix_cache_text = None
            rec.pending_prefix_cache_key = None

        rec.commit_history.append(item)
        await rec.commit_queue.put({"type": "commit", **item})
        items.append(item)

        logger.info(
            "commit queued session=%s seq=%s reason=%s text_len=%s text=%r",
            rec.session_id[:8],
            seq,
            reason,
            len(text),
            text[:180],
        )

    return items


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "fish-pcm-proxy",
        "version": app.version,
        "upstream_tts_url": UPSTREAM_TTS_URL,
        "default_reference_id": DEFAULT_REFERENCE_ID,
        "auth_enabled": bool(FISH_API_KEY),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    stats = await session_store.stats()
    return {"ok": True, **stats}


@app.post("/intro-cache/key")
async def intro_cache_key(req: SessionOpenRequest) -> JSONResponse:
    config = normalize_config(req.config_text)
    key = _intro_cache_key(config)

    return JSONResponse(
        {
            "ok": True,
            "key": key,
            "key_short": key[:8],
            "schema": PREFIX_CACHE_SCHEMA_VERSION,
            "salt_set": bool(PREFIX_CACHE_SALT),
            "reference_id": _normalized_reference_id(config),
            "intro_text_len": len(config.intro_cache.text),
            "intro_enabled": config.intro_cache.enabled,
            "warm_on_session_open": config.intro_cache.warm_on_session_open,
        }
    )


@app.get("/intro-cache/stats")
async def intro_cache_stats() -> JSONResponse:
    now = time.time()
    entries = intro_cache.list_entries()
    return JSONResponse(
        {
            "ok": True,
            "entries": len(entries),
            "max_entries": _RUNTIME.proxy.intro_cache.max_entries,
            "items": [
                {
                    "key": entry.key,
                    "age_sec": round(now - entry.created_at, 1),
                    "expires_in_sec": round(max(0, entry.expires_at - now), 1),
                    "pcm_bytes": len(entry.pcm),
                    "text_preview": entry.text[:180],
                    "audio_meta": entry.audio_meta,
                }
                for entry in entries
            ],
        }
    )


@app.post("/intro-cache/clear")
async def intro_cache_clear() -> JSONResponse:
    await intro_cache.clear()
    return JSONResponse({"ok": True, "message": "intro cache cleared"})


@app.post("/prefix-cache/add")
async def prefix_cache_add(req: PrefixCacheAddRequest) -> JSONResponse:
    config = normalize_config(req.config_text)

    if req.text is not None and req.texts is not None:
        raise HTTPException(400, detail="provide either text or texts, not both")

    if req.text is None and req.texts is None:
        raise HTTPException(400, detail="provide text or texts")

    texts = [req.text] if req.text is not None else list(req.texts or [])
    normalized_texts = [text.strip() for text in texts]
    if any(not text for text in normalized_texts):
        raise HTTPException(400, detail="prefix-cache text must not be empty")

    if req.clear_existing:
        await prefix_cache_library.clear()

    created: list[dict[str, Any]] = []
    existed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        for text in normalized_texts:
            key = _prefix_cache_key(config, text)

            try:
                entry, was_created = await prefix_cache_library.get_or_create(
                    key=key,
                    text=text,
                    config=config,
                    client=client,
                )
            except Exception as exc:
                logger.error(
                    "failed to add prefix-cache text=%r key=%s: %s",
                    text[:100],
                    key[:8],
                    exc,
                )
                failed_item = {
                    "status": "failed",
                    "text": text,
                    "key": key,
                    "key_short": key[:8],
                    "error": str(exc),
                }
                failed.append(failed_item)
                items.append(failed_item)
                if req.fail_fast:
                    break
                continue

            public = {
                "status": "created" if was_created else "existed",
                **_prefix_cache_public_item(entry),
            }

            if was_created:
                created.append(public)
            else:
                existed.append(public)
            items.append(public)

    return JSONResponse(
        {
            "ok": not failed,
            "created": created,
            "existed": existed,
            "failed": failed,
            "created_count": len(created),
            "existed_count": len(existed),
            "failed_count": len(failed),
            "items": items,
        }
    )


@app.get("/prefix-cache/stats")
async def prefix_cache_stats() -> JSONResponse:
    entries = await prefix_cache_library.list_entries()
    return JSONResponse(
        {
            "ok": True,
            "entries": len(entries),
            "max_entries": prefix_cache_library.max_entries,
            "items": [_prefix_cache_public_item(entry) for entry in entries],
        }
    )


@app.post("/prefix-cache/clear")
async def prefix_cache_clear() -> JSONResponse:
    await prefix_cache_library.clear()
    return JSONResponse({"ok": True, "message": "prefix cache cleared"})


@app.post("/prefix-cache/key")
async def prefix_cache_key(req: PrefixCacheKeyRequest) -> JSONResponse:
    config = normalize_config(req.config_text)
    text = req.text.strip()
    if not text:
        raise HTTPException(400, detail="prefix-cache text must not be empty")

    key = _prefix_cache_key(config, text)
    entry = await prefix_cache_library.get(key)

    return JSONResponse(
        {
            "ok": True,
            "key": key,
            "key_short": key[:8],
            "exists": entry is not None,
            "text": text,
        }
    )


@app.post("/session/open")
async def session_open(req: SessionOpenRequest) -> JSONResponse:
    config = normalize_config(req.config_text)
    rec = await session_store.create(config.model_dump(mode="python"), req.config_text)

    if config.tts.stateful_synthesis:
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                rec.synthesis_session_id = await _open_upstream_synthesis_session(
                    client,
                    rec,
                    config,
                )

                logger.info(
                    "proxy session %s linked to upstream synthesis session %s "
                    "history_turns=%s history_chars=%s history_code_frames=%s",
                    rec.session_id[:8],
                    rec.synthesis_session_id[:8],
                    config.tts.stateful_history_turns,
                    config.tts.stateful_history_chars,
                    config.tts.stateful_history_code_frames,
                )
        except Exception as exc:
            if config.tts.stateful_fallback_to_stateless:
                logger.warning(
                    "failed to open upstream synthesis session; "
                    "falling back to stateless: %s",
                    exc,
                )
                rec.synthesis_session_id = None
            else:
                await session_store.close(rec.session_id)
                raise HTTPException(
                    502,
                    detail=f"failed to open upstream synthesis session: {exc}",
                )

    if (
        config.intro_cache.enabled
        and config.intro_cache.warm_on_session_open
        and config.intro_cache.text.strip()
    ):
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                entry = await intro_cache.get_or_create(
                    key=_intro_cache_key(config),
                    text=config.intro_cache.text,
                    config=config,
                    client=client,
                )

                if entry is not None:
                    await _preload_intro_cache_context(
                        rec=rec,
                        config=config,
                        entry=entry,
                        client=client,
                    )
        except Exception as exc:
            if not config.intro_cache.ignore_errors:
                if rec.synthesis_session_id:
                    async with httpx.AsyncClient(
                        timeout=UPSTREAM_TIMEOUT
                    ) as cleanup_client:
                        await _close_upstream_synthesis_session(
                            cleanup_client,
                            rec.synthesis_session_id,
                            reason="warm_intro_failed",
                        )
                await session_store.close(rec.session_id)
                raise HTTPException(
                    502,
                    detail=f"failed to warm intro cache: {exc}",
                )
            logger.warning(
                "failed to warm intro cache for session %s: %s",
                rec.session_id[:8],
                exc,
            )

    logger.info(
        "session open id=%s ref=%s emit_bytes=%s stateful=%s intro_warm=%s",
        rec.session_id[:8],
        config.tts.reference_id,
        config.playback.target_emit_bytes,
        bool(rec.synthesis_session_id),
        config.intro_cache.warm_on_session_open,
    )

    return JSONResponse(
        {
            "ok": True,
            "session_id": rec.session_id,
            "config": rec.config,
            "ttl_sec": SESSION_TTL_SEC,
            "synthesis_session_id": rec.synthesis_session_id,
        }
    )


@app.get("/session/{session_id}")
async def session_get(session_id: str) -> JSONResponse:
    rec = await session_store.get(session_id, touch=False)

    if rec is None:
        raise HTTPException(404, detail="session not found")

    config = ProxyConfig.model_validate(rec.config)

    return JSONResponse(
        {
            "ok": True,
            "session_id": rec.session_id,
            "config": rec.config,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
            "expires_at": rec.expires_at,
            "buffer_text": rec.buffer_text,
            "buffer_chars": len(rec.buffer_text),
            "append_chars": rec.append_chars,
            "commit_chars": rec.commit_chars,
            "input_closed": rec.input_closed,
            "synthesis_session_id": rec.synthesis_session_id,
            "stateful_synthesis": config.tts.stateful_synthesis,
            "pending_prefix_cache_text": rec.pending_prefix_cache_text,
            "stream_started": rec.stream_started,
            "stream_finished": rec.stream_finished,
            "commit_history": rec.commit_history[-100:],
        }
    )


@app.post("/session/{session_id}/append")
async def session_append(session_id: str, req: SessionAppendRequest) -> JSONResponse:
    rec = await session_store.get(session_id, touch=True)

    if rec is None:
        raise HTTPException(404, detail="session not found")

    config = ProxyConfig.model_validate(rec.config)
    prefix_cache_text = _resolve_prefix_cache_alias(req.cache, req.cash)
    prefix_cache_key: str | None = None

    if prefix_cache_text is not None:
        if not prefix_cache_text:
            raise HTTPException(400, detail="cache must not be empty")

        prefix_cache_key = _prefix_cache_key(config, prefix_cache_text)

    if len(req.text) + len(rec.buffer_text) > config.session.max_buffer_chars:
        raise HTTPException(400, detail="session buffer limit exceeded")

    async with rec.lock:
        if rec.input_closed:
            raise HTTPException(409, detail="session input already finished")

        if prefix_cache_text is not None:
            if rec.next_commit_seq != 1 or rec.commit_history:
                raise HTTPException(
                    400,
                    detail="cache/cash is allowed only before the first commit",
                )

            if (
                rec.pending_prefix_cache_text is not None
                and rec.pending_prefix_cache_text != prefix_cache_text
            ):
                raise HTTPException(
                    400,
                    detail="a different prefix-cache is already pending for the first commit",
                )

            prefix_cache_entry = await prefix_cache_library.get(prefix_cache_key or "")
            if prefix_cache_entry is None:
                raise HTTPException(
                    404,
                    detail=(
                        "prefix-cache entry not found for "
                        f"text={prefix_cache_text!r} key={(prefix_cache_key or '')[:8]}"
                    ),
                )

            rec.pending_prefix_cache_text = prefix_cache_entry.text
            rec.pending_prefix_cache_key = prefix_cache_entry.key

        rec.append_chars += len(req.text)
        rec.buffer_text += req.text

        commits, remainder = _extract_commits(
            rec.buffer_text,
            config.commit,
            next_commit_seq=rec.next_commit_seq,
            force=False,
        )

        rec.buffer_text = remainder
        queued = await _queue_commits(rec, commits)

        if rec.buffer_text.strip():
            if rec.buffer_started_at is None or commits:
                rec.buffer_started_at = time.time()
            _ensure_commit_timer(rec)
        else:
            rec.buffer_started_at = None
            _cancel_commit_timer(rec)

        await _touch_session(rec)

    return JSONResponse(
        {
            "ok": True,
            "session_id": rec.session_id,
            "accepted_chars": len(req.text),
            "buffer_text": rec.buffer_text,
            "buffer_chars": len(rec.buffer_text),
            "committed": queued,
            "input_closed": rec.input_closed,
        }
    )


@app.post("/session/{session_id}/flush")
async def session_flush(session_id: str, req: SessionFlushRequest) -> JSONResponse:
    rec = await session_store.get(session_id, touch=True)

    if rec is None:
        raise HTTPException(404, detail="session not found")

    async with rec.lock:
        queued = []

        if rec.buffer_text.strip():
            queued = await _queue_commits(rec, [(rec.buffer_text.strip(), req.reason)])
            rec.buffer_text = ""

        rec.buffer_started_at = None
        _cancel_commit_timer(rec)

        await _touch_session(rec)

    logger.info(
        "session flush id=%s committed=%s input_closed=%s",
        rec.session_id[:8],
        len(queued),
        rec.input_closed,
    )

    return JSONResponse(
        {
            "ok": True,
            "session_id": rec.session_id,
            "committed": queued,
            "buffer_text": rec.buffer_text,
            "buffer_chars": len(rec.buffer_text),
            "input_closed": rec.input_closed,
        }
    )


@app.post("/session/{session_id}/finish")
async def session_finish(session_id: str, req: SessionFinishRequest) -> JSONResponse:
    rec = await session_store.get(session_id, touch=True)

    if rec is None:
        raise HTTPException(404, detail="session not found")

    config = ProxyConfig.model_validate(rec.config)

    async with rec.lock:
        if rec.input_closed:
            return JSONResponse(
                {
                    "ok": True,
                    "session_id": rec.session_id,
                    "already_finished": True,
                    "buffer_text": rec.buffer_text,
                    "committed": [],
                }
            )

        rec.input_closed = True
        queued = []

        if rec.buffer_text.strip():
            queued = await _queue_commits(rec, [(rec.buffer_text.strip(), req.reason)])
            rec.buffer_text = ""

        rec.buffer_started_at = None
        _cancel_commit_timer(rec)

        await rec.commit_queue.put({"type": "eof"})
        await _touch_session(rec)

    logger.info(
        "session finish id=%s committed=%s buffer_chars=%s input_closed=%s",
        rec.session_id[:8],
        len(queued),
        len(rec.buffer_text),
        rec.input_closed,
    )

    if config.session.auto_close_on_finish:
        logger.info("session %s marked auto-close-on-finish", rec.session_id[:8])

    return JSONResponse(
        {
            "ok": True,
            "session_id": rec.session_id,
            "committed": queued,
            "buffer_text": rec.buffer_text,
            "buffer_chars": len(rec.buffer_text),
            "input_closed": rec.input_closed,
        }
    )


@app.post("/session/close")
async def session_close(req: SessionCloseRequest) -> JSONResponse:
    rec = await session_store.get(req.session_id, touch=False)

    if rec and rec.synthesis_session_id:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            await _close_upstream_synthesis_session(
                client,
                rec.synthesis_session_id,
                reason="manual_close",
            )

    removed = await session_store.close(req.session_id)

    return JSONResponse(
        {
            "ok": True,
            "closed": removed,
            "session_id": req.session_id,
        }
    )


@app.get("/session/{session_id}/pcm-stream")
async def session_pcm_stream(session_id: str):
    rec = await session_store.get(session_id, touch=True)

    if rec is None:
        raise HTTPException(404, detail="session not found")

    if rec.stream_lock.locked():
        raise HTTPException(409, detail="session audio stream already opened")

    config = ProxyConfig.model_validate(rec.config)
    target_emit_bytes = config.playback.target_emit_bytes

    async def emit(obj: dict[str, Any]) -> bytes:
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    async def stream_one_commit(
        client: httpx.AsyncClient,
        req_id: str,
        commit_item: dict[str, Any],
    ) -> AsyncGenerator[bytes, None]:
        commit_seq = int(commit_item["seq"])
        is_first_commit = commit_seq == 1
        has_prefix_cache = is_first_commit and bool(commit_item.get("prefix_cache_text"))
        full_commit_mode = bool(PREFIX_CACHE_FULL_COMMIT_MODE and has_prefix_cache)
        prefix_cache_entry: PrefixCacheEntry | None = None
        prefix_cache_preload_context = bool(has_prefix_cache)

        if full_commit_mode and PREFIX_CACHE_DISABLE_PRELOAD_IN_FULL_COMMIT_MODE:
            prefix_cache_preload_context = False

        if has_prefix_cache:
            prefix_cache_entry = await _get_prefix_cache_entry_for_commit(
                config,
                commit_item,
            )
            async for event in emit_prefix_cache(
                rec=rec,
                config=config,
                client=client,
                req_id=req_id,
                commit_item=commit_item,
                entry=prefix_cache_entry,
                full_commit_mode=full_commit_mode,
                preload_context=prefix_cache_preload_context,
            ):
                yield event

            await asyncio.sleep(0)

        effective_target_emit_bytes = (
            config.playback.first_commit_target_emit_bytes
            if is_first_commit
            else config.playback.target_emit_bytes
        )
        effective_start_buffer_ms = (
            config.playback.first_commit_start_buffer_ms
            if is_first_commit
            else config.playback.start_buffer_ms
        )
        effective_fade_in_ms = (
            PREFIX_CACHE_GENERATION_FADE_IN_MS
            if has_prefix_cache
            else config.playback.fade_in_ms
        )

        generation_tail_text = commit_item["text"]
        prefix_cache_text = prefix_cache_entry.text if prefix_cache_entry else ""
        upstream_text = (
            _join_prefix_and_tail(prefix_cache_text, generation_tail_text)
            if full_commit_mode
            else generation_tail_text
        )
        planned_prefix_audio_skip_bytes = 0
        prefix_audio_skip_ms_estimate = 0.0

        if full_commit_mode and prefix_cache_entry is not None:
            meta = prefix_cache_entry.audio_meta
            meta_sample_rate = int(meta["sample_rate"])
            meta_channels = int(meta["channels"])
            meta_sample_width = int(meta["sample_width"])
            meta_frame_bytes = meta_channels * meta_sample_width
            adjust_bytes = _pcm_bytes_for_duration_ms_allow_negative(
                meta_sample_rate,
                meta_channels,
                meta_sample_width * 8,
                PREFIX_CACHE_SKIP_ADJUST_MS,
            )
            planned_prefix_audio_skip_bytes = max(
                0,
                len(prefix_cache_entry.pcm) + adjust_bytes,
            )
            planned_prefix_audio_skip_bytes = _align_down(
                planned_prefix_audio_skip_bytes,
                meta_frame_bytes,
            )
            prefix_audio_skip_ms_estimate = _pcm_ms_estimate(
                planned_prefix_audio_skip_bytes,
                sample_rate=meta_sample_rate,
                channels=meta_channels,
                sample_width=meta_sample_width,
            )

        should_reset, reset_reason = _should_reset_upstream_session(
            rec,
            config,
            commit_item,
        )

        if should_reset:
            old_id = rec.synthesis_session_id

            try:
                new_id = await _open_upstream_synthesis_session(client, rec, config)
                rec.synthesis_session_id = new_id
                rec.chars_since_upstream_reset = 0
                rec.last_upstream_reset_commit_seq = commit_item["seq"]

                logger.info(
                    "upstream synthesis reset session=%s commit_seq=%s "
                    "reason=%s old=%s new=%s",
                    rec.session_id[:8],
                    commit_item["seq"],
                    reset_reason,
                    old_id[:8] if old_id else None,
                    rec.synthesis_session_id[:8],
                )

                yield await emit(
                    {
                        "type": "upstream_reset",
                        "session_id": rec.session_id,
                        "commit_seq": commit_item["seq"],
                        "reason": reset_reason,
                        "old_synthesis_session_id": old_id,
                        "new_synthesis_session_id": rec.synthesis_session_id,
                    }
                )

                await _close_upstream_synthesis_session(
                    client,
                    old_id,
                    reason=reset_reason,
                )

            except Exception as exc:
                logger.warning(
                    "upstream synthesis reset FAILED session=%s reason=%s: %s",
                    rec.session_id[:8],
                    reset_reason,
                    exc,
                )
                yield await emit(
                    {
                        "type": "upstream_reset_failed",
                        "session_id": rec.session_id,
                        "commit_seq": commit_item["seq"],
                        "reason": reset_reason,
                        "message": str(exc),
                    }
                )

            await asyncio.sleep(0)

        payload = build_upstream_payload(
            text=upstream_text,
            config=config,
            commit_seq=commit_seq,
        )

        url = UPSTREAM_TTS_URL
        if rec.synthesis_session_id and config.tts.stateful_synthesis:
            url = url.replace("/v1/tts", "/v1/synthesis/synthesize")
            payload.update(
                {
                    "synthesis_session_id": rec.synthesis_session_id,
                    "commit_seq": commit_item["seq"],
                    "commit_reason": commit_item["reason"],
                }
            )

        header_buffer = bytearray()
        pending_pcm = bytearray()
        header_sent = False

        upstream_bytes = 0
        first_emit_done = False
        start_buffer_bytes = 0
        skip_remaining_bytes = planned_prefix_audio_skip_bytes
        skipped_pcm_bytes = 0
        skip_done_emitted = not full_commit_mode

        fade_in_applied = False
        sample_rate = 0
        channels = 0
        bits_per_sample = 0
        frame_bytes = 2
        tail_hold_bytes = 0
        fade_in_bytes = 0
        fade_out_bytes = 0

        async def flush_pending(force: bool = False) -> AsyncGenerator[bytes, None]:
            nonlocal first_emit_done, fade_in_applied

            while True:
                if not pending_pcm:
                    return

                if (
                    config.playback.boundary_smoothing_enabled
                    and not fade_in_applied
                    and fade_in_bytes > 0
                    and len(pending_pcm) >= fade_in_bytes
                ):
                    faded = _apply_pcm16_fade(
                        bytes(pending_pcm[:fade_in_bytes]),
                        channels=channels,
                        fade_in_frames=int(
                            sample_rate * effective_fade_in_ms / 1000.0
                        ),
                        fade_out_frames=0,
                    )
                    pending_pcm[:fade_in_bytes] = faded
                    fade_in_applied = True

                threshold = effective_target_emit_bytes

                if not first_emit_done:
                    threshold = max(effective_target_emit_bytes, start_buffer_bytes)

                threshold = _align_down(threshold, frame_bytes)
                if threshold <= 0:
                    threshold = frame_bytes

                protected_tail = 0 if force else tail_hold_bytes
                available = len(pending_pcm) - protected_tail

                if available <= 0:
                    return

                if not force and available < threshold:
                    return

                if force:
                    out_len = len(pending_pcm)
                else:
                    out_len = min(threshold, available)

                out_len = _align_down(out_len, frame_bytes)

                if out_len <= 0:
                    return

                out = bytes(pending_pcm[:out_len])
                del pending_pcm[:out_len]

                pcm_seq = rec.next_pcm_seq
                rec.next_pcm_seq += 1
                is_first_pcm_for_this_commit = not first_emit_done

                yield await emit(
                    {
                        "type": "pcm",
                        "req_id": req_id,
                        "session_id": rec.session_id,
                        "commit_seq": commit_item["seq"],
                        "seq": pcm_seq,
                        "first_pcm_for_commit": is_first_pcm_for_this_commit,
                        "data": base64.b64encode(out).decode("ascii"),
                    }
                )

                first_emit_done = True
                await asyncio.sleep(0)

        async def append_pcm_after_prefix_skip(
            pcm: bytes,
        ) -> AsyncGenerator[bytes, None]:
            nonlocal skip_remaining_bytes, skipped_pcm_bytes, skip_done_emitted

            if full_commit_mode and skip_remaining_bytes > 0:
                take = min(skip_remaining_bytes, len(pcm))
                if take > 0:
                    skip_remaining_bytes -= take
                    skipped_pcm_bytes += take
                    pcm = pcm[take:]

            if full_commit_mode and not skip_done_emitted and skip_remaining_bytes <= 0:
                skip_done_emitted = True
                yield await emit(
                    {
                        "type": "prefix_cache_generation_skip_done",
                        "req_id": req_id,
                        "session_id": rec.session_id,
                        "commit_seq": commit_item["seq"],
                        "cache_key": prefix_cache_entry.key if prefix_cache_entry else None,
                        "cache_key_short": (
                            prefix_cache_entry.key[:8] if prefix_cache_entry else None
                        ),
                        "skipped_pcm_bytes": skipped_pcm_bytes,
                        "skipped_ms_estimate": _pcm_ms_estimate(
                            skipped_pcm_bytes,
                            sample_rate=sample_rate,
                            channels=channels,
                            sample_width=bits_per_sample // 8,
                        ),
                    }
                )
                await asyncio.sleep(0)

            if not pcm:
                return

            pending_pcm.extend(pcm)
            async for event in flush_pending(force=False):
                yield event

        yield await emit(
            {
                "type": "commit_start",
                "session_id": rec.session_id,
                "commit_seq": commit_item["seq"],
                "reason": commit_item["reason"],
                "text_preview": commit_item["text"][:180],
                "text_len": len(commit_item["text"]),
                "effective_target_emit_bytes": effective_target_emit_bytes,
                "effective_start_buffer_ms": effective_start_buffer_ms,
                "effective_fade_in_ms": effective_fade_in_ms,
                "has_prefix_cache": has_prefix_cache,
                "full_commit_mode": full_commit_mode,
                "prefix_cache_preload_context": prefix_cache_preload_context,
                "prefix_cache_key": (
                    prefix_cache_entry.key if prefix_cache_entry else None
                ),
                "prefix_cache_key_short": (
                    prefix_cache_entry.key[:8] if prefix_cache_entry else None
                ),
                "full_generation_text_preview": upstream_text[:180],
                "full_generation_text_len": len(upstream_text),
                "prefix_cache_text": prefix_cache_text or None,
                "prefix_cache_text_len": len(prefix_cache_text),
                "generation_tail_text_preview": generation_tail_text[:180],
                "generation_tail_text_len": len(generation_tail_text),
                "planned_prefix_audio_skip_bytes": planned_prefix_audio_skip_bytes,
                "prefix_audio_skip_bytes": planned_prefix_audio_skip_bytes,
                "prefix_audio_skip_ms_estimate": prefix_audio_skip_ms_estimate,
                "prefix_audio_skip_adjust_ms": PREFIX_CACHE_SKIP_ADJUST_MS,
                "server_perf_ms": int(time.perf_counter() * 1000),
            }
        )
        await asyncio.sleep(0)

        logger.info(
            "REQ %s commit_seq=%s upstream start text_len=%s ref=%s url=%s text=%r",
            req_id,
            commit_item["seq"],
            len(upstream_text),
            payload["reference_id"],
            url,
            upstream_text[:180],
        )

        async with client.stream(
            "POST",
            url,
            json=payload,
            headers=_auth_headers(),
        ) as upstream:
            if upstream.status_code != 200:
                text_body = await upstream.aread()
                msg = text_body.decode("utf-8", errors="replace")
                raise RuntimeError(f"upstream status={upstream.status_code}: {msg}")

            async for chunk in upstream.aiter_raw(1024):
                if not chunk:
                    continue

                upstream_bytes += len(chunk)

                if not header_sent:
                    header_buffer.extend(chunk)
                    parsed = _parse_wav_header(header_buffer)

                    if parsed is None:
                        continue

                    sample_rate, channels, bits_per_sample, data_offset = parsed

                    if bits_per_sample != 16:
                        raise RuntimeError(
                            f"unsupported bits_per_sample={bits_per_sample}"
                        )

                    frame_bytes = channels * (bits_per_sample // 8)

                    fade_in_bytes = _align_down(
                        _pcm_bytes_for_duration_ms(
                            sample_rate,
                            channels,
                            bits_per_sample,
                            effective_fade_in_ms,
                        ),
                        frame_bytes,
                    )

                    fade_out_bytes = _align_down(
                        _pcm_bytes_for_duration_ms(
                            sample_rate,
                            channels,
                            bits_per_sample,
                            config.playback.fade_out_ms,
                        ),
                        frame_bytes,
                    )

                    tail_hold_bytes = (
                        fade_out_bytes
                        if config.playback.boundary_smoothing_enabled
                        else 0
                    )

                    current_audio_meta = {
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "sample_width": bits_per_sample // 8,
                    }

                    start_buffer_bytes = _pcm_bytes_for_duration_ms(
                        sample_rate,
                        channels,
                        bits_per_sample,
                        effective_start_buffer_ms,
                    )
                    start_buffer_bytes = _align_down(start_buffer_bytes, frame_bytes)

                    header_sent = True

                    if rec.audio_meta is None:
                        rec.audio_meta = current_audio_meta

                        yield await emit(
                            {
                                "type": "meta",
                                "session_id": rec.session_id,
                                "commit_seq": commit_item["seq"],
                                **rec.audio_meta,
                            }
                        )
                        await asyncio.sleep(0)

                    elif rec.audio_meta != current_audio_meta:
                        raise RuntimeError(
                            "upstream WAV format changed between commits: "
                            f"was={rec.audio_meta}, now={current_audio_meta}"
                        )

                    pcm = bytes(header_buffer[data_offset:])
                    header_buffer.clear()

                    if pcm:
                        async for event in append_pcm_after_prefix_skip(pcm):
                            yield event

                else:
                    async for event in append_pcm_after_prefix_skip(chunk):
                        yield event

        if not header_sent:
            raise RuntimeError("upstream finished before WAV data header was parsed")

        if full_commit_mode and skip_remaining_bytes > 0:
            raise RuntimeError(
                "upstream finished before prefix-cache audio skip completed: "
                f"remaining={skip_remaining_bytes} planned={planned_prefix_audio_skip_bytes}"
            )

        boundary_kind, pause_ms = _pause_ms_for_commit(
            commit_item["text"],
            commit_item["reason"],
            config.playback,
        )

        if (
            config.playback.boundary_smoothing_enabled
            and pending_pcm
            and fade_out_bytes > 0
        ):
            tail_len = min(len(pending_pcm), fade_out_bytes)
            tail_len = _align_down(tail_len, frame_bytes)

            if tail_len > 0:
                start = len(pending_pcm) - tail_len
                faded_tail = _apply_pcm16_fade(
                    bytes(pending_pcm[start:]),
                    channels=channels,
                    fade_in_frames=0,
                    fade_out_frames=int(
                        sample_rate * config.playback.fade_out_ms / 1000.0
                    ),
                )
                pending_pcm[start:] = faded_tail

        if pause_ms > 0:
            pending_pcm.extend(
                _pcm_silence_bytes(
                    sample_rate,
                    channels,
                    bits_per_sample,
                    pause_ms,
                )
            )

            yield await emit(
                {
                    "type": "pause",
                    "req_id": req_id,
                    "session_id": rec.session_id,
                    "commit_seq": commit_item["seq"],
                    "boundary": boundary_kind,
                    "pause_ms": pause_ms,
                }
            )
            await asyncio.sleep(0)

        if config.playback.stop_grace_ms > 0:
            await asyncio.sleep(config.playback.stop_grace_ms / 1000.0)

        async for event in flush_pending(force=True):
            yield event

        logger.info(
            "REQ %s commit_seq=%s upstream done upstream_bytes=%s",
            req_id,
            commit_item["seq"],
            upstream_bytes,
        )

        rec.chars_since_upstream_reset += int(
            len(upstream_text)
        )

        yield await emit(
            {
                "type": "commit_done",
                "session_id": rec.session_id,
                "commit_seq": commit_item["seq"],
                "upstream_bytes": upstream_bytes,
                "boundary": boundary_kind,
                "pause_ms": pause_ms,
                "fade_in_ms": effective_fade_in_ms,
                "fade_out_ms": config.playback.fade_out_ms,
                "has_prefix_cache": has_prefix_cache,
                "full_commit_mode": full_commit_mode,
                "skipped_prefix_pcm_bytes": skipped_pcm_bytes,
                "upstream_text_len": len(upstream_text),
            }
        )

    async def body_iter() -> AsyncGenerator[bytes, None]:
        req_id = uuid.uuid4().hex[:8]

        async with rec.stream_lock:
            rec.stream_started = True
            await _touch_session(rec)

            yield await emit(
                {
                    "type": "session_start",
                    "req_id": req_id,
                    "session_id": rec.session_id,
                    "target_emit_bytes": target_emit_bytes,
                    "first_commit_target_emit_bytes": (
                        config.playback.first_commit_target_emit_bytes
                    ),
                    "first_commit_start_buffer_ms": (
                        config.playback.first_commit_start_buffer_ms
                    ),
                    "client_start_buffer_ms": config.playback.client_start_buffer_ms,
                    "client_initial_start_delay_ms": (
                        config.playback.client_initial_start_delay_ms
                    ),
                    "prefix_cache_pause_after_ms": PREFIX_CACHE_PAUSE_AFTER_MS,
                    "prefix_cache_fade_out_ms": PREFIX_CACHE_FADE_OUT_MS,
                    "prefix_cache_generation_fade_in_ms": (
                        PREFIX_CACHE_GENERATION_FADE_IN_MS
                    ),
                    "prefix_cache_full_commit_mode": PREFIX_CACHE_FULL_COMMIT_MODE,
                    "prefix_cache_skip_adjust_ms": PREFIX_CACHE_SKIP_ADJUST_MS,
                    "prefix_cache_disable_preload_in_full_commit_mode": (
                        PREFIX_CACHE_DISABLE_PRELOAD_IN_FULL_COMMIT_MODE
                    ),
                    "synthesis_session_id": rec.synthesis_session_id,
                }
            )
            await asyncio.sleep(0)

            try:
                async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                    if config.intro_cache.enabled and config.intro_cache.text.strip():
                        try:
                            async for event in emit_intro_cache(
                                rec=rec,
                                config=config,
                                client=client,
                                req_id=req_id,
                            ):
                                yield event
                        except Exception as exc:
                            if not config.intro_cache.ignore_errors:
                                raise
                            logger.warning(
                                "intro emission failed for session %s: %s",
                                session_id[:8],
                                exc,
                            )
                            yield await emit(
                                {
                                    "type": "intro_error",
                                    "req_id": req_id,
                                    "session_id": rec.session_id,
                                    "message": f"intro emission failed: {exc}",
                                }
                            )

                    while True:
                        item = await rec.commit_queue.get()

                        if item["type"] == "commit":
                            async for event in stream_one_commit(client, req_id, item):
                                yield event

                            await _touch_session(rec)
                            continue

                        if item["type"] == "eof":
                            break

                        if item["type"] == "abort":
                            yield await emit(
                                {
                                    "type": "session_aborted",
                                    "req_id": req_id,
                                    "session_id": rec.session_id,
                                }
                            )
                            return

            except asyncio.CancelledError:
                logger.warning("session stream cancelled id=%s", rec.session_id[:8])
                raise

            except Exception as exc:
                logger.exception(
                    "session stream failed id=%s: %s",
                    rec.session_id[:8],
                    exc,
                )

                yield await emit(
                    {
                        "type": "error",
                        "req_id": req_id,
                        "session_id": rec.session_id,
                        "message": f"session stream failed: {exc}",
                    }
                )
                return

            finally:
                rec.stream_finished = True
                await _touch_session(rec)

            yield await emit(
                {
                    "type": "session_done",
                    "req_id": req_id,
                    "session_id": rec.session_id,
                    "commit_count": len(rec.commit_history),
                }
            )

            if config.session.auto_close_on_finish:
                try:
                    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                        await _close_upstream_synthesis_session(
                            client,
                            rec.synthesis_session_id,
                            reason="auto_close_on_finish",
                        )
                finally:
                    await session_store.close(rec.session_id)

    return StreamingResponse(
        body_iter(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/pcm-stream")
async def pcm_stream(text: str = Query(..., min_length=1, max_length=2000)):
    config = load_runtime_config().proxy
    req_id = uuid.uuid4().hex[:8]
    payload = build_upstream_payload(text=text, config=config, commit_seq=1)
    effective_target_emit_bytes = config.playback.first_commit_target_emit_bytes
    effective_start_buffer_ms = config.playback.first_commit_start_buffer_ms

    async def emit(obj: dict[str, Any]) -> bytes:
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    async def body_iter() -> AsyncGenerator[bytes, None]:
        header_buffer = bytearray()
        pending_pcm = bytearray()
        header_sent = False
        pcm_seq = 1
        start_buffer_bytes = 0
        first_emit_done = False

        fade_in_applied = False
        sample_rate = 0
        channels = 0
        bits_per_sample = 0
        frame_bytes = 2
        tail_hold_bytes = 0
        fade_in_bytes = 0
        fade_out_bytes = 0

        async def flush_pending(force: bool = False) -> AsyncGenerator[bytes, None]:
            nonlocal pcm_seq, first_emit_done, fade_in_applied

            while True:
                if not pending_pcm:
                    return

                if (
                    config.playback.boundary_smoothing_enabled
                    and not fade_in_applied
                    and fade_in_bytes > 0
                    and len(pending_pcm) >= fade_in_bytes
                ):
                    faded = _apply_pcm16_fade(
                        bytes(pending_pcm[:fade_in_bytes]),
                        channels=channels,
                        fade_in_frames=int(
                            sample_rate * config.playback.fade_in_ms / 1000.0
                        ),
                        fade_out_frames=0,
                    )
                    pending_pcm[:fade_in_bytes] = faded
                    fade_in_applied = True

                threshold = effective_target_emit_bytes

                if not first_emit_done:
                    threshold = max(effective_target_emit_bytes, start_buffer_bytes)

                threshold = _align_down(threshold, frame_bytes)
                if threshold <= 0:
                    threshold = frame_bytes

                protected_tail = 0 if force else tail_hold_bytes
                available = len(pending_pcm) - protected_tail

                if available <= 0:
                    return

                if not force and available < threshold:
                    return

                if force:
                    out_len = len(pending_pcm)
                else:
                    out_len = min(threshold, available)

                out_len = _align_down(out_len, frame_bytes)

                if out_len <= 0:
                    return

                out = bytes(pending_pcm[:out_len])
                del pending_pcm[:out_len]

                is_first_pcm_for_this_commit = not first_emit_done

                yield await emit(
                    {
                        "type": "pcm",
                        "req_id": req_id,
                        "commit_seq": 1,
                        "seq": pcm_seq,
                        "first_pcm_for_commit": is_first_pcm_for_this_commit,
                        "data": base64.b64encode(out).decode("ascii"),
                    }
                )

                pcm_seq += 1
                first_emit_done = True
                await asyncio.sleep(0)

        yield await emit(
            {
                "type": "proxy_start",
                "req_id": req_id,
                "mode": "stateless",
            }
        )
        yield await emit(
            {
                "type": "commit_start",
                "req_id": req_id,
                "mode": "stateless",
                "commit_seq": 1,
                "reason": "stateless",
                "text_preview": text[:180],
                "text_len": len(text),
                "effective_target_emit_bytes": effective_target_emit_bytes,
                "effective_start_buffer_ms": effective_start_buffer_ms,
                "server_perf_ms": int(time.perf_counter() * 1000),
            }
        )

        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                UPSTREAM_TTS_URL,
                json=payload,
                headers=_auth_headers(),
            ) as upstream:
                if upstream.status_code != 200:
                    msg = (await upstream.aread()).decode("utf-8", errors="replace")
                    yield await emit(
                        {
                            "type": "error",
                            "req_id": req_id,
                            "message": msg,
                        }
                    )
                    return

                async for chunk in upstream.aiter_raw(1024):
                    if not chunk:
                        continue

                    if not header_sent:
                        header_buffer.extend(chunk)
                        parsed = _parse_wav_header(header_buffer)

                        if parsed is None:
                            continue

                        sample_rate, channels, bits_per_sample, data_offset = parsed

                        if bits_per_sample != 16:
                            yield await emit(
                                {
                                    "type": "error",
                                    "req_id": req_id,
                                    "message": (
                                        f"unsupported bits_per_sample={bits_per_sample}"
                                    ),
                                }
                            )
                            return

                        header_sent = True
                        frame_bytes = channels * (bits_per_sample // 8)

                        fade_in_bytes = _align_down(
                            _pcm_bytes_for_duration_ms(
                                sample_rate,
                                channels,
                                bits_per_sample,
                                config.playback.fade_in_ms,
                            ),
                            frame_bytes,
                        )

                        fade_out_bytes = _align_down(
                            _pcm_bytes_for_duration_ms(
                                sample_rate,
                                channels,
                                bits_per_sample,
                                config.playback.fade_out_ms,
                            ),
                            frame_bytes,
                        )

                        tail_hold_bytes = (
                            fade_out_bytes
                            if config.playback.boundary_smoothing_enabled
                            else 0
                        )

                        start_buffer_bytes = _pcm_bytes_for_duration_ms(
                            sample_rate,
                            channels,
                            bits_per_sample,
                            effective_start_buffer_ms,
                        )
                        start_buffer_bytes = _align_down(start_buffer_bytes, frame_bytes)

                        yield await emit(
                            {
                                "type": "meta",
                                "req_id": req_id,
                                "sample_rate": sample_rate,
                                "channels": channels,
                                "sample_width": bits_per_sample // 8,
                            }
                        )

                        pcm = bytes(header_buffer[data_offset:])
                        header_buffer.clear()

                        if pcm:
                            pending_pcm.extend(pcm)

                            async for event in flush_pending():
                                yield event

                    else:
                        pending_pcm.extend(chunk)

                        async for event in flush_pending():
                            yield event

        if not header_sent:
            yield await emit(
                {
                    "type": "error",
                    "req_id": req_id,
                    "message": "upstream finished before WAV data header was parsed",
                }
            )
            return

        boundary_kind, pause_ms = _pause_ms_for_commit(
            text,
            "force",
            config.playback,
        )

        if (
            config.playback.boundary_smoothing_enabled
            and pending_pcm
            and fade_out_bytes > 0
        ):
            tail_len = min(len(pending_pcm), fade_out_bytes)
            tail_len = _align_down(tail_len, frame_bytes)

            if tail_len > 0:
                start = len(pending_pcm) - tail_len
                faded_tail = _apply_pcm16_fade(
                    bytes(pending_pcm[start:]),
                    channels=channels,
                    fade_in_frames=0,
                    fade_out_frames=int(
                        sample_rate * config.playback.fade_out_ms / 1000.0
                    ),
                )
                pending_pcm[start:] = faded_tail

        if pause_ms > 0:
            pending_pcm.extend(
                _pcm_silence_bytes(
                    sample_rate,
                    channels,
                    bits_per_sample,
                    pause_ms,
                )
            )

            yield await emit(
                {
                    "type": "pause",
                    "req_id": req_id,
                    "boundary": boundary_kind,
                    "pause_ms": pause_ms,
                }
            )
            await asyncio.sleep(0)

        if config.playback.stop_grace_ms > 0:
            await asyncio.sleep(config.playback.stop_grace_ms / 1000.0)

        async for event in flush_pending(force=True):
            yield event

        yield await emit(
            {
                "type": "done",
                "req_id": req_id,
                "boundary": boundary_kind,
                "pause_ms": pause_ms,
            }
        )

    return StreamingResponse(
        body_iter(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )

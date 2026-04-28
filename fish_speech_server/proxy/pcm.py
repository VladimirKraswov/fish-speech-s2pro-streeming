from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import struct
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from typing import Any, AsyncGenerator, Optional, Tuple

import httpx
from fastapi import Body, FastAPI, HTTPException, Query
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

DEFAULT_SESSION_CONFIG = _RUNTIME.proxy.model_dump(mode="python")


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


class SessionFlushRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field("manual_flush", min_length=1, max_length=200)


class SessionFinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field("input_finished", min_length=1, max_length=200)


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
    audio_meta: dict[str, Any] | None = None
    stream_started: bool = False
    stream_finished: bool = False
    closed: bool = False
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
        async with self._lock:
            expired = []
            for sid, rec in list(self._items.items()):
                if rec.expires_at <= now and not rec.stream_started:
                    expired.append(sid)
                elif rec.expires_at <= now and rec.stream_finished:
                    expired.append(sid)

            for sid in expired:
                self._items.pop(sid, None)

            if expired:
                logger.info("session cleanup removed={}", len(expired))

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
                    }
                    for rec in self._items.values()
                ],
            }


session_store = SessionStore()


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


def build_upstream_payload(text: str, config: ProxyConfig) -> dict[str, Any]:
    tts = config.tts
    reference_id = tts.reference_id.strip() or config.default_reference_id

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


def _safe_hard_cut(text: str, limit: int, min_keep: int) -> int:
    """
    Не режем слово посередине при hard_limit.

    Если до limit есть пробел после min_keep — режем по последнему пробелу.
    Если нормального пробела нет — режем ровно по limit.
    """
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

    arr = np.frombuffer(pcm, dtype="<i2").copy()

    if channels <= 0:
        return pcm

    usable_samples = (arr.size // channels) * channels
    if usable_samples <= 0:
        return pcm

    prefix = arr[:usable_samples].reshape(-1, channels)
    frame_count = prefix.shape[0]

    if fade_in_frames > 0:
        n = min(frame_count, fade_in_frames)
        if n > 0:
            gain = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float32)[:, None]
            prefix[:n] = np.clip(
                prefix[:n].astype(np.float32) * gain,
                -32768,
                32767,
            ).astype(np.int16)

    if fade_out_frames > 0:
        n = min(frame_count, fade_out_frames)
        if n > 0:
            gain = np.linspace(1.0, 0.0, n, endpoint=True, dtype=np.float32)[:, None]
            prefix[-n:] = np.clip(
                prefix[-n:].astype(np.float32) * gain,
                -32768,
                32767,
            ).astype(np.int16)

    arr[:usable_samples] = prefix.reshape(-1)
    return arr.astype("<i2", copy=False).tobytes()


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

        stage = cfg.first if (next_commit_seq + len(commits)) == 1 else cfg.next

        if force:
            piece = _right_trim_committed(text)
            if piece:
                commits.append((piece, "force"))
            text = ""
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
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    stats = await session_store.stats()
    return {"ok": True, **stats}


@app.post("/session/open")
async def session_open(req: SessionOpenRequest) -> JSONResponse:
    config = normalize_config(req.config_text)
    rec = await session_store.create(config.model_dump(mode="python"), req.config_text)

    if config.tts.stateful_synthesis:
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                open_payload = {
                    "reference_id": config.tts.reference_id or config.default_reference_id,
                    "max_history_turns": config.tts.stateful_history_turns,
                    "max_history_chars": config.tts.stateful_history_chars,
                    "max_history_code_frames": config.tts.stateful_history_code_frames,
                }

                resp = await client.post(
                    UPSTREAM_TTS_URL.replace("/v1/tts", "/v1/synthesis/sessions/open"),
                    json=open_payload,
                    headers={"Accept": "application/json"},
                )

                if resp.status_code != 200:
                    body = resp.text
                    raise RuntimeError(f"upstream status={resp.status_code}: {body}")

                data = resp.json()
                rec.synthesis_session_id = data.get("synthesis_session_id")
                if not rec.synthesis_session_id:
                    raise RuntimeError("upstream did not return synthesis_session_id")

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
                    "failed to open upstream synthesis session; falling back to stateless: %s",
                    exc,
                )
                rec.synthesis_session_id = None
            else:
                await session_store.close(rec.session_id)
                raise HTTPException(
                    502,
                    detail=f"failed to open upstream synthesis session: {exc}",
                )

    logger.info(
        "session open id=%s ref=%s emit_bytes=%s stateful=%s",
        rec.session_id[:8],
        config.tts.reference_id,
        config.playback.target_emit_bytes,
        bool(rec.synthesis_session_id),
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

    if len(req.text) + len(rec.buffer_text) > config.session.max_buffer_chars:
        raise HTTPException(400, detail="session buffer limit exceeded")

    async with rec.lock:
        if rec.input_closed:
            raise HTTPException(409, detail="session input already finished")

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
        logger.info("session {} marked auto-close-on-finish", rec.session_id[:8])

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
        try:
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                await client.post(
                    UPSTREAM_TTS_URL.replace("/v1/tts", "/v1/synthesis/sessions/close"),
                    json=rec.synthesis_session_id,
                    headers={"Accept": "application/json"},
                )
        except Exception as exc:
            logger.warning(
                "failed to close upstream synthesis session %s: %s",
                rec.synthesis_session_id,
                exc,
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
        req_id: str,
        commit_item: dict[str, Any],
    ) -> AsyncGenerator[bytes, None]:
        payload = build_upstream_payload(text=commit_item["text"], config=config)

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

        fade_in_applied = False
        sample_rate = None
        channels = None
        bits_per_sample = None
        frame_bytes = 2
        tail_hold_bytes = 0
        fade_in_bytes = 0
        fade_out_bytes = 0

        async def flush_pending(force: bool = False) -> AsyncGenerator[bytes, None]:
            nonlocal first_emit_done, fade_in_applied

            while True:
                if not pending_pcm:
                    return

                # Перед первой отдачей применить fade-in к началу commit audio.
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

                threshold = target_emit_bytes

                if not first_emit_done:
                    threshold = max(target_emit_bytes, start_buffer_bytes)

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
                first_emit_done = True

                yield await emit(
                    {
                        "type": "pcm",
                        "req_id": req_id,
                        "session_id": rec.session_id,
                        "commit_seq": commit_item["seq"],
                        "seq": pcm_seq,
                        "data": base64.b64encode(out).decode("ascii"),
                    }
                )

                await asyncio.sleep(0)

        yield await emit(
            {
                "type": "commit_start",
                "session_id": rec.session_id,
                "commit_seq": commit_item["seq"],
                "reason": commit_item["reason"],
                "text": commit_item["text"],
                "text_len": len(commit_item["text"]),
            }
        )
        await asyncio.sleep(0)

        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            logger.info(
                "REQ %s commit_seq=%s upstream start text_len=%s ref=%s url=%s text=%r",
                req_id,
                commit_item["seq"],
                len(commit_item["text"]),
                payload["reference_id"],
                url,
                commit_item["text"][:180],
            )

            async with client.stream("POST", url, json=payload) as upstream:
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

                        current_audio_meta = {
                            "sample_rate": sample_rate,
                            "channels": channels,
                            "sample_width": bits_per_sample // 8,
                        }

                        start_buffer_bytes = _pcm_bytes_for_duration_ms(
                            sample_rate,
                            channels,
                            bits_per_sample,
                            config.playback.start_buffer_ms,
                        )

                        start_buffer_bytes = _align_down(
                            start_buffer_bytes, frame_bytes
                        )

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
                            pending_pcm.extend(pcm)

                            async for event in flush_pending(force=False):
                                yield event

                    else:
                        pending_pcm.extend(chunk)

                        async for event in flush_pending(force=False):
                            yield event

        if not header_sent:
            raise RuntimeError("upstream finished before WAV data header was parsed")

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
            # fade-out только на реальном аудио, ДО добавления тишины
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

        yield await emit(
            {
                "type": "commit_done",
                "session_id": rec.session_id,
                "commit_seq": commit_item["seq"],
                "upstream_bytes": upstream_bytes,
                "boundary": boundary_kind,
                "pause_ms": pause_ms,
                "fade_in_ms": config.playback.fade_in_ms,
                "fade_out_ms": config.playback.fade_out_ms,
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
                    "synthesis_session_id": rec.synthesis_session_id,
                }
            )
            await asyncio.sleep(0)

            try:
                while True:
                    item = await rec.commit_queue.get()

                    if item["type"] == "commit":
                        async for event in stream_one_commit(req_id, item):
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
                logger.warning("session stream cancelled id={}", rec.session_id[:8])
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
    payload = build_upstream_payload(text=text, config=config)
    target_emit_bytes = config.playback.target_emit_bytes

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
        sample_rate = None
        channels = None
        bits_per_sample = None
        frame_bytes = 2
        tail_hold_bytes = 0
        fade_in_bytes = 0
        fade_out_bytes = 0

        async def flush_pending(force: bool = False) -> AsyncGenerator[bytes, None]:
            nonlocal pcm_seq, first_emit_done, fade_in_applied

            while True:
                if not pending_pcm:
                    return

                # Перед первой отдачей применить fade-in к началу audio.
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

                threshold = target_emit_bytes

                if not first_emit_done:
                    threshold = max(target_emit_bytes, start_buffer_bytes)

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

                yield await emit(
                    {
                        "type": "pcm",
                        "req_id": req_id,
                        "seq": pcm_seq,
                        "data": base64.b64encode(out).decode("ascii"),
                    }
                )

                pcm_seq += 1
                first_emit_done = True

        yield await emit(
            {
                "type": "proxy_start",
                "req_id": req_id,
                "mode": "stateless",
            }
        )

        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            async with client.stream("POST", UPSTREAM_TTS_URL, json=payload) as upstream:
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
                            config.playback.start_buffer_ms,
                        )

                        start_buffer_bytes = _align_down(
                            start_buffer_bytes, frame_bytes
                        )

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
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .buffer import BufferEmit, StreamingTextBuffer
from .schema import (
    ClientCleanup,
    ClientClear,
    ClientCloseSession,
    ClientFlush,
    ClientMessage,
    ClientPatchConfig,
    ClientPing,
    ClientStartSession,
    ClientTextDelta,
    ServerAudioChunk,
    ServerAudioMeta,
    ServerBufferCleared,
    ServerChunkQueued,
    ServerCleanupDone,
    ServerConfigPatched,
    ServerError,
    ServerEvent,
    ServerPong,
    ServerSessionClosed,
    ServerSessionStarted,
    ServerTTSFinished,
    ServerTTSStarted,
    ServerTextAccepted,
    SessionModeConfig,
    TTSChunkRequest,
    TTSAudioMeta,
    event_to_dict,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OutboundFrame:
    event: dict[str, Any] | None = None
    audio: bytes | None = None


def _now_monotonic() -> float:
    return time.monotonic()


def _count_words(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class SessionManager:
    """
    Один manager на одну realtime-сессию.

    Умеет:
    - принимать text deltas;
    - буферизовать текст;
    - резать его на подходящие chunks;
    - последовательно стримить chunks в основной TTS backend;
    - отдавать наружу JSON-события и бинарные аудио чанки;
    - делать idle cleanup backend'а.
    """

    def __init__(self, config: SessionModeConfig | None = None) -> None:
        self.config = config or SessionModeConfig()

        self._events: asyncio.Queue[OutboundFrame] = asyncio.Queue()
        self._pending_tts: asyncio.Queue[TTSChunkRequest | None] = asyncio.Queue()

        self._started = False
        self._closed = False

        self._last_activity_at = _now_monotonic()
        self._last_buffer_append_at = self._last_activity_at
        self._last_cleanup_at = 0.0
        self._cleanup_sent_for_idle_window = False

        self._active_tts_request_task: asyncio.Task[None] | None = None
        self._tts_worker_task: asyncio.Task[None] | None = None
        self._idle_watchdog_task: asyncio.Task[None] | None = None

        self._buffer = self._build_buffer()
        self._client = self._build_http_client()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._started:
            return

        self._started = True

        self._tts_worker_task = asyncio.create_task(
            self._tts_worker_loop(),
            name=f"session-tts-worker-{self.config.session_id}",
        )
        self._idle_watchdog_task = asyncio.create_task(
            self._idle_watchdog_loop(),
            name=f"session-idle-watchdog-{self.config.session_id}",
        )

        await self._emit_json(
            ServerSessionStarted(
                session_id=self.config.session_id,
                config=self.config,
            )
        )

    async def close(self, reason: str | None = None) -> None:
        if self._closed:
            return

        self._closed = True
        self._buffer.clear()

        await self._drop_pending_chunks()

        if self._active_tts_request_task and not self._active_tts_request_task.done():
            self._active_tts_request_task.cancel()

        await self._pending_tts.put(None)

        if self._tts_worker_task:
            with contextlib.suppress(asyncio.CancelledError):
                await self._tts_worker_task

        if self._idle_watchdog_task:
            self._idle_watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._idle_watchdog_task

        await self._client.aclose()

        await self._emit_json(
            ServerSessionClosed(
                session_id=self.config.session_id,
                reason=reason,
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle(self, message: ClientMessage) -> None:
        if isinstance(message, ClientStartSession):
            await self._handle_start_session(message)
            return

        if not self._started:
            await self.start()

        if isinstance(message, ClientPatchConfig):
            await self._handle_patch_config(message)
        elif isinstance(message, ClientTextDelta):
            await self._handle_text_delta(message)
        elif isinstance(message, ClientFlush):
            await self._handle_flush(message)
        elif isinstance(message, ClientClear):
            await self._handle_clear(message)
        elif isinstance(message, ClientCleanup):
            await self._handle_cleanup(message)
        elif isinstance(message, ClientCloseSession):
            await self.close(reason=message.reason)
        elif isinstance(message, ClientPing):
            await self._emit_json(
                ServerPong(
                    session_id=self.config.session_id,
                    ts_ms=message.ts_ms,
                )
            )
        else:
            await self._emit_error(
                code="unsupported_message",
                message=f"Unsupported message: {type(message).__name__}",
                fatal=False,
            )

    async def next_outbound(self) -> OutboundFrame:
        return await self._events.get()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_start_session(self, message: ClientStartSession) -> None:
        if message.config is not None:
            self.config = message.config
            self._rebuild_buffer()
            await self._reset_http_client()

        await self.start()

    async def _handle_patch_config(self, message: ClientPatchConfig) -> None:
        try:
            current = self.config.model_dump(mode="json")
            merged = _deep_merge(current, message.patch)
            self.config = SessionModeConfig.model_validate(merged)
            self._rebuild_buffer(preserve_text=True)
            await self._reset_http_client()

            await self._emit_json(
                ServerConfigPatched(
                    session_id=self.config.session_id,
                    config=self.config,
                )
            )
        except Exception as exc:
            await self._emit_error(
                code="invalid_config_patch",
                message=str(exc),
                fatal=False,
            )

    async def _handle_text_delta(self, message: ClientTextDelta) -> None:
        self._touch_activity()
        self._last_buffer_append_at = _now_monotonic()
        self._cleanup_sent_for_idle_window = False

        if (
            self.config.policy.close_tts_stream_on_new_text
            and self._active_tts_request_task
            and not self._active_tts_request_task.done()
        ):
            self._active_tts_request_task.cancel()

        emits = self._buffer.push(message.text, final=message.final)

        await self._emit_json(
            ServerTextAccepted(
                session_id=self.config.session_id,
                buffered_text_len=len(self._buffer.text),
                buffered_words=_count_words(self._buffer.text),
                trace_id=message.trace_id,
                final=message.final,
            )
        )

        await self._enqueue_emits(emits=emits, trace_id=message.trace_id)

    async def _handle_flush(self, message: ClientFlush) -> None:
        self._touch_activity()
        self._cleanup_sent_for_idle_window = False

        emits = self._buffer.flush(final=message.final)
        await self._enqueue_emits(emits=emits, trace_id=None)

    async def _handle_clear(self, message: ClientClear) -> None:
        self._touch_activity()
        self._buffer.clear()

        await self._emit_json(
            ServerBufferCleared(
                session_id=self.config.session_id,
                reason=message.reason,
            )
        )

    async def _handle_cleanup(self, message: ClientCleanup) -> None:
        self._touch_activity()
        self._buffer.clear()

        await self._drop_pending_chunks()

        if self._active_tts_request_task and not self._active_tts_request_task.done():
            self._active_tts_request_task.cancel()

        ok = await self._run_backend_cleanup(reason=message.reason or "manual_cleanup")
        if ok:
            await self._emit_json(
                ServerCleanupDone(
                    session_id=self.config.session_id,
                    reason=message.reason,
                )
            )

    # ------------------------------------------------------------------
    # Buffer / chunk queue
    # ------------------------------------------------------------------

    def _build_buffer(self) -> StreamingTextBuffer:
        return StreamingTextBuffer(
            min_words=self.config.buffer.min_words,
            soft_limit_chars=self.config.buffer.soft_limit_chars,
            hard_limit_chars=self.config.buffer.hard_limit_chars,
        )

    def _rebuild_buffer(self, preserve_text: bool = True) -> None:
        current_text = self._buffer.text if preserve_text else ""
        self._buffer = self._build_buffer()
        if current_text:
            self._buffer.replace(current_text)

    async def _enqueue_emits(
        self,
        *,
        emits: list[BufferEmit],
        trace_id: str | None,
    ) -> None:
        for emit in emits:
            chunk = TTSChunkRequest(
                session_id=self.config.session_id,
                text=emit.text,
                reason=emit.reason,
                words=emit.words,
                chars=emit.chars,
                trace_id=trace_id,
                final=(emit.reason == "final"),
            )
            await self._pending_tts.put(chunk)
            await self._emit_json(
                ServerChunkQueued(
                    session_id=self.config.session_id,
                    chunk=chunk,
                    queue_size=self._pending_tts.qsize(),
                )
            )

    async def _drop_pending_chunks(self) -> None:
        while True:
            try:
                item = self._pending_tts.get_nowait()
            except asyncio.QueueEmpty:
                break

            if item is None:
                await self._pending_tts.put(None)
                break

    # ------------------------------------------------------------------
    # TTS worker
    # ------------------------------------------------------------------

    async def _tts_worker_loop(self) -> None:
        while True:
            item = await self._pending_tts.get()
            if item is None:
                return

            chunk = item

            try:
                self._active_tts_request_task = asyncio.create_task(
                    self._stream_tts_chunk(chunk),
                    name=f"session-stream-chunk-{chunk.chunk_id}",
                )
                await self._active_tts_request_task

            except asyncio.CancelledError:
                if self._closed:
                    return
                await self._emit_error(
                    code="tts_request_cancelled",
                    message="Active TTS request was cancelled",
                    fatal=False,
                    details={"chunk_id": chunk.chunk_id},
                )

            except Exception as exc:
                logger.exception("tts worker error chunk_id=%s", chunk.chunk_id)
                await self._emit_error(
                    code="tts_worker_error",
                    message=str(exc),
                    fatal=False,
                    details={"chunk_id": chunk.chunk_id},
                )

            finally:
                self._active_tts_request_task = None
                self._touch_activity()

    async def _stream_tts_chunk(self, chunk: TTSChunkRequest) -> None:
        url = f"{self.config.tts.base_url}{self.config.tts.endpoint}"

        payload = {
            "text": chunk.text,
            "chunk_length": max(
                100,
                min(300, self.config.buffer.soft_limit_chars),
            ),
            "format": self.config.tts.format,
            "streaming": self.config.tts.streaming,
            "stream_tokens": self.config.tts.stream_tokens,
            "stream_chunk_size": self.config.tts.stream_chunk_size,
            "max_new_tokens": self.config.tts.max_new_tokens,
            "top_p": self.config.tts.top_p,
            "repetition_penalty": self.config.tts.repetition_penalty,
            "temperature": self.config.tts.temperature,
            "cleanup_mode": self.config.tts.cleanup_mode,
            "stream_empty_cache": self.config.tts.stream_empty_cache,
            "use_memory_cache": self.config.tts.use_memory_cache,
            "normalize": self.config.tts.normalize,
            "reference_id": self.config.tts.reference_id,
        }

        await self._emit_json(
            ServerTTSStarted(
                session_id=self.config.session_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                trace_id=chunk.trace_id,
            )
        )

        timeout = httpx.Timeout(
            connect=self.config.tts.connect_timeout_sec,
            read=self.config.tts.read_timeout_sec,
            write=self.config.tts.connect_timeout_sec,
            pool=self.config.tts.connect_timeout_sec,
        )

        seq = 0
        total_bytes = 0
        meta_sent = False

        async with self._client.stream(
            "POST",
            url,
            json=payload,
            timeout=timeout,
        ) as response:
            response.raise_for_status()

            async for data in response.aiter_bytes():
                if not data:
                    continue

                if not meta_sent:
                    meta_sent = True
                    await self._emit_json(
                        ServerAudioMeta(
                            session_id=self.config.session_id,
                            chunk_id=chunk.chunk_id,
                            meta=TTSAudioMeta(
                                sample_rate=self.config.sample_rate,
                                channels=self.config.channels,
                                format=self.config.tts.format,
                            ),
                        )
                    )

                seq += 1
                total_bytes += len(data)

                await self._emit_audio(
                    ServerAudioChunk(
                        session_id=self.config.session_id,
                        chunk_id=chunk.chunk_id,
                        seq=seq,
                        size_bytes=len(data),
                        trace_id=chunk.trace_id,
                    ),
                    data,
                )

        await self._emit_json(
            ServerTTSFinished(
                session_id=self.config.session_id,
                chunk_id=chunk.chunk_id,
                total_chunks=seq,
                total_bytes=total_bytes,
                trace_id=chunk.trace_id,
                final=chunk.final,
            )
        )

    # ------------------------------------------------------------------
    # Idle / cleanup
    # ------------------------------------------------------------------

    async def _idle_watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(0.10)

            if self._closed:
                return

            now = _now_monotonic()
            idle_for = now - self._last_activity_at
            buffer_idle_for = now - self._last_buffer_append_at

            if not self._buffer.empty():
                if buffer_idle_for >= self.config.policy.force_flush_after_sec:
                    emits = self._buffer.flush(final=False)
                    await self._enqueue_emits(emits=emits, trace_id=None)

            total_idle_needed = (
                self.config.policy.session_idle_timeout_sec
                + self.config.policy.cleanup_after_idle_sec
            )

            is_drained = (
                self._buffer.empty()
                and self._pending_tts.qsize() == 0
                and (
                    self._active_tts_request_task is None
                    or self._active_tts_request_task.done()
                )
            )

            if (
                self.config.tts.cleanup_mode == "session_idle"
                and is_drained
                and not self._cleanup_sent_for_idle_window
                and idle_for >= total_idle_needed
            ):
                ok = await self._run_backend_cleanup(reason="idle_cleanup")
                if ok:
                    self._cleanup_sent_for_idle_window = True
                    await self._emit_json(
                        ServerCleanupDone(
                            session_id=self.config.session_id,
                            reason="idle_cleanup",
                        )
                    )

    async def _run_backend_cleanup(self, reason: str) -> bool:
        url = f"{self.config.tts.base_url}{self.config.tts.endpoint}"
        payload = {
            "text": "",
            "format": self.config.tts.format,
            "streaming": False,
            "stream_tokens": False,
            "cleanup_mode": "request_end",
            "control": "cleanup",
        }

        timeout = httpx.Timeout(
            connect=self.config.tts.connect_timeout_sec,
            read=self.config.tts.read_timeout_sec,
            write=self.config.tts.connect_timeout_sec,
            pool=self.config.tts.connect_timeout_sec,
        )

        try:
            response = await self._client.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            self._last_cleanup_at = _now_monotonic()
            logger.info(
                "session=%s backend cleanup ok reason=%s",
                self.config.session_id,
                reason,
            )
            return True

        except Exception as exc:
            logger.exception(
                "session=%s backend cleanup failed reason=%s",
                self.config.session_id,
                reason,
            )
            await self._emit_error(
                code="backend_cleanup_failed",
                message=str(exc),
                fatal=False,
                details={"reason": reason},
            )
            return False

    # ------------------------------------------------------------------
    # IO helpers
    # ------------------------------------------------------------------

    def _build_http_client(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(
            connect=self.config.tts.connect_timeout_sec,
            read=self.config.tts.read_timeout_sec,
            write=self.config.tts.connect_timeout_sec,
            pool=self.config.tts.connect_timeout_sec,
        )
        return httpx.AsyncClient(timeout=timeout)

    async def _reset_http_client(self) -> None:
        old_client = self._client
        self._client = self._build_http_client()
        await old_client.aclose()

    async def _emit_json(self, event: ServerEvent) -> None:
        await self._events.put(
            OutboundFrame(
                event=event_to_dict(event),
                audio=None,
            )
        )

    async def _emit_audio(self, event: ServerAudioChunk, audio: bytes) -> None:
        await self._events.put(
            OutboundFrame(
                event=event_to_dict(event),
                audio=audio,
            )
        )

    async def _emit_error(
        self,
        *,
        code: str,
        message: str,
        fatal: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        await self._emit_json(
            ServerError(
                code=code,
                message=message,
                session_id=self.config.session_id,
                details=details,
                fatal=fatal,
            )
        )

    def _touch_activity(self) -> None:
        self._last_activity_at = _now_monotonic()
        self._cleanup_sent_for_idle_window = False

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        active = (
            self._active_tts_request_task is not None
            and not self._active_tts_request_task.done()
        )
        return {
            "session_id": self.config.session_id,
            "started": self._started,
            "closed": self._closed,
            "buffer_chars": len(self._buffer.text),
            "buffer_words": _count_words(self._buffer.text),
            "pending_tts": self._pending_tts.qsize(),
            "tts_active": active,
            "last_cleanup_ago_sec": (
                None
                if self._last_cleanup_at <= 0
                else round(_now_monotonic() - self._last_cleanup_at, 3)
            ),
            "config": self.config.model_dump(mode="json"),
        }

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, indent=2)
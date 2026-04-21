from __future__ import annotations

import asyncio
import contextlib
import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from session_mode.manager import OutboundFrame, SessionManager
from session_mode.schema import (
    ClientMessage,
    ServerError,
    parse_client_message,
    event_to_dict,
)


_ACTIVE_SESSIONS: dict[str, SessionManager] = {}
_ACTIVE_SESSIONS_LOCK = asyncio.Lock()


async def _register_manager(manager: SessionManager) -> None:
    async with _ACTIVE_SESSIONS_LOCK:
        _ACTIVE_SESSIONS[manager.config.session_id] = manager


async def _unregister_manager(manager: SessionManager) -> None:
    async with _ACTIVE_SESSIONS_LOCK:
        current = _ACTIVE_SESSIONS.get(manager.config.session_id)
        if current is manager:
            _ACTIVE_SESSIONS.pop(manager.config.session_id, None)
            return

        stale_keys = [key for key, value in _ACTIVE_SESSIONS.items() if value is manager]
        for key in stale_keys:
            _ACTIVE_SESSIONS.pop(key, None)


async def _sync_manager_registry(manager: SessionManager) -> None:
    async with _ACTIVE_SESSIONS_LOCK:
        stale_keys = [key for key, value in _ACTIVE_SESSIONS.items() if value is manager]
        for key in stale_keys:
            if key != manager.config.session_id:
                _ACTIVE_SESSIONS.pop(key, None)

        _ACTIVE_SESSIONS[manager.config.session_id] = manager


async def _reader_loop(websocket: WebSocket, manager: SessionManager) -> None:
    while True:
        raw = await websocket.receive()
        ws_type = raw.get("type")

        if ws_type == "websocket.disconnect":
            raise WebSocketDisconnect(raw.get("code", 1000))

        if raw.get("bytes") is not None:
            await websocket.send_json(
                event_to_dict(
                    ServerError(
                        code="binary_frame_not_supported",
                        message="Binary frames are not supported by session_mode input",
                        session_id=manager.config.session_id,
                        fatal=False,
                    )
                )
            )
            continue

        text = raw.get("text")
        if text is None:
            continue

        try:
            payload: dict[str, Any] | str
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = text

            if isinstance(payload, str):
                message: ClientMessage = parse_client_message(
                    {"type": "text_delta", "text": payload}
                )
            else:
                message = parse_client_message(payload)

            await manager.handle(message)
            await _sync_manager_registry(manager)

        except Exception as exc:
            logger.exception(
                "invalid websocket message session_id={} err={}",
                manager.config.session_id,
                exc,
            )
            await websocket.send_json(
                event_to_dict(
                    ServerError(
                        code="invalid_client_message",
                        message=str(exc),
                        session_id=manager.config.session_id,
                        fatal=False,
                    )
                )
            )


async def _writer_loop(websocket: WebSocket, manager: SessionManager) -> None:
    while True:
        frame: OutboundFrame = await manager.next_outbound()

        if frame.event is not None:
            await websocket.send_json(frame.event)

        if frame.audio is not None:
            await websocket.send_bytes(frame.audio)

        event_type = None
        if isinstance(frame.event, dict):
            event_type = frame.event.get("event") or frame.event.get("type")

        if event_type == "session_closed":
            return


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("session_mode starting")

    try:
        yield
    finally:
        logger.info("session_mode shutting down")

        async with _ACTIVE_SESSIONS_LOCK:
            managers = list(_ACTIVE_SESSIONS.values())
            _ACTIVE_SESSIONS.clear()

        for manager in managers:
            with contextlib.suppress(Exception):
                await manager.close(reason="app_shutdown")

        logger.info("session_mode stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Fish Speech Session Mode",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_timing_middleware(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Elapsed-Ms"] = f"{elapsed_ms:.1f}"
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled app exception path={} err={}", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "internal_server_error",
                "detail": str(exc),
            },
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        async with _ACTIVE_SESSIONS_LOCK:
            return {
                "ok": True,
                "service": "session_mode",
                "active_sessions": len(_ACTIVE_SESSIONS),
                "session_ids": sorted(_ACTIVE_SESSIONS.keys()),
            }

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        async with _ACTIVE_SESSIONS_LOCK:
            return {
                "ok": True,
                "service": "session_mode",
                "ready": True,
                "active_sessions": len(_ACTIVE_SESSIONS),
            }

    @app.get("/sessions")
    async def list_sessions() -> dict[str, Any]:
        async with _ACTIVE_SESSIONS_LOCK:
            return {
                "count": len(_ACTIVE_SESSIONS),
                "sessions": {
                    session_id: manager.snapshot()
                    for session_id, manager in _ACTIVE_SESSIONS.items()
                },
            }

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        async with _ACTIVE_SESSIONS_LOCK:
            manager = _ACTIVE_SESSIONS.get(session_id)

        if manager is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return manager.snapshot()

    @app.websocket("/ws")
    async def websocket_session(websocket: WebSocket) -> None:
        await websocket.accept()

        manager = SessionManager()
        await manager.start()
        await _register_manager(manager)

        logger.info("session websocket connected: {}", manager.config.session_id)

        reader_task = asyncio.create_task(
            _reader_loop(websocket, manager),
            name=f"session-reader-{manager.config.session_id}",
        )
        writer_task = asyncio.create_task(
            _writer_loop(websocket, manager),
            name=f"session-writer-{manager.config.session_id}",
        )

        try:
            done, pending = await asyncio.wait(
                {reader_task, writer_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        except WebSocketDisconnect:
            logger.info("session websocket disconnected: {}", manager.config.session_id)
        except Exception:
            logger.exception("session websocket failed: {}", manager.config.session_id)
            with contextlib.suppress(Exception):
                await websocket.close(code=1011)
        finally:
            with contextlib.suppress(Exception):
                await manager.close(reason="websocket_closed")
            await _unregister_manager(manager)

            for task in (reader_task, writer_task):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    return app


app = create_app()
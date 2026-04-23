# tools/proxy/fish_proxy_server.py
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

FISH_TTS_URL = os.environ.get("FISH_TTS_URL", "http://127.0.0.1:8080/v1/tts")
FISH_HEALTH_URL = os.environ.get("FISH_HEALTH_URL", "http://127.0.0.1:8080/v1/health")
FISH_API_KEY = os.environ.get("FISH_API_KEY", "").strip()
CONNECT_TIMEOUT = float(os.environ.get("PROXY_CONNECT_TIMEOUT", "10"))
WRITE_TIMEOUT = float(os.environ.get("PROXY_WRITE_TIMEOUT", "30"))
# read=None because upstream is a stream
REQUEST_TIMEOUT = httpx.Timeout(connect=CONNECT_TIMEOUT, read=None, write=WRITE_TIMEOUT, pool=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Fish Speech Streaming Proxy", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if FISH_API_KEY:
        headers["Authorization"] = f"Bearer {FISH_API_KEY}"
    return headers


@app.get("/")
async def root() -> PlainTextResponse:
    return PlainTextResponse(
        "Fish Speech proxy is running. Use /health or /stream?text=..."
    )


@app.get("/health")
async def health() -> JSONResponse:
    try:
        response = await app.state.client.get(FISH_HEALTH_URL, headers=_auth_headers())
        ok = response.status_code == 200
        payload = {
            "proxy": "ok",
            "upstream_ok": ok,
            "upstream_status": response.status_code,
            "upstream_url": FISH_HEALTH_URL,
        }
        return JSONResponse(payload, status_code=200 if ok else 502)
    except Exception as exc:
        return JSONResponse(
            {
                "proxy": "ok",
                "upstream_ok": False,
                "upstream_url": FISH_HEALTH_URL,
                "error": str(exc),
            },
            status_code=502,
        )


@app.get("/stream")
async def stream(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str | None = Query(None),
) -> StreamingResponse:
    payload: dict[str, object] = {
        "text": text,
        "streaming": True,
        "format": "wav",
    }
    if reference_id:
        payload["reference_id"] = reference_id

    request = app.state.client.build_request(
        "POST",
        FISH_TTS_URL,
        json=payload,
        headers=_auth_headers(),
    )
    upstream = await app.state.client.send(request, stream=True)

    if upstream.status_code != 200:
        body = await upstream.aread()
        await upstream.aclose()
        detail = body.decode("utf-8", errors="replace") or upstream.reason_phrase
        raise HTTPException(status_code=upstream.status_code, detail=detail)

    async def body_iter() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                if chunk:
                    yield chunk
        except (httpx.RemoteProtocolError, httpx.ReadError):
            # Upstream sometimes closes a chunked response without a terminating chunk.
            # Browsers already received the bytes; treat this as normal end-of-stream.
            return
        finally:
            await upstream.aclose()

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(body_iter(), media_type="audio/wav", headers=headers)


@app.get("/stream-segments")
async def stream_segments(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str | None = Query(None),
) -> StreamingResponse:
    """
    Convert the upstream chunked WAV response into NDJSON where each line contains
    one complete WAV segment as base64.
    This avoids the browser waiting on a long chunked WAV and lets JS play each
    segment immediately via AudioContext.decodeAudioData.
    """
    import base64

    payload: dict[str, object] = {
        "text": text,
        "streaming": True,
        "format": "wav",
    }
    if reference_id:
        payload["reference_id"] = reference_id

    request = app.state.client.build_request(
        "POST",
        FISH_TTS_URL,
        json=payload,
        headers=_auth_headers(),
    )
    upstream = await app.state.client.send(request, stream=True)

    if upstream.status_code != 200:
        body = await upstream.aread()
        await upstream.aclose()
        detail = body.decode("utf-8", errors="replace") or upstream.reason_phrase
        raise HTTPException(status_code=upstream.status_code, detail=detail)

    async def iter_ndjson() -> AsyncIterator[bytes]:
        buffer = bytearray()
        try:
            async for chunk in upstream.aiter_bytes():
                if not chunk:
                    continue
                buffer.extend(chunk)

                while True:
                    # Need at least RIFF header + chunk size.
                    if len(buffer) < 12:
                        break

                    # Find the next RIFF header.
                    start = buffer.find(b"RIFF")
                    if start == -1:
                        # Keep only a tiny tail in case the marker is split.
                        if len(buffer) > 8:
                            del buffer[:-8]
                        break
                    if start > 0:
                        del buffer[:start]
                        if len(buffer) < 12:
                            break

                    if buffer[8:12] != b"WAVE":
                        # False positive; skip one byte and resync.
                        del buffer[:1]
                        continue

                    total_size = int.from_bytes(buffer[4:8], "little") + 8
                    if total_size <= 0 or total_size > 100_000_000:
                        del buffer[:1]
                        continue

                    if len(buffer) < total_size:
                        break

                    wav_bytes = bytes(buffer[:total_size])
                    del buffer[:total_size]
                    line = json.dumps(
                        {
                            "type": "wav",
                            "b64": base64.b64encode(wav_bytes).decode("ascii"),
                        },
                        ensure_ascii=False,
                    ).encode("utf-8") + b"\n"
                    yield line
        except (httpx.RemoteProtocolError, httpx.ReadError):
            # Same rationale as in /stream: use everything already received.
            pass
        finally:
            if buffer:
                # Best-effort: if the tail contains a full WAV, emit it.
                while len(buffer) >= 12:
                    start = buffer.find(b"RIFF")
                    if start == -1:
                        break
                    if start > 0:
                        del buffer[:start]
                        continue
                    if buffer[8:12] != b"WAVE":
                        del buffer[:1]
                        continue
                    total_size = int.from_bytes(buffer[4:8], "little") + 8
                    if len(buffer) < total_size:
                        break
                    wav_bytes = bytes(buffer[:total_size])
                    del buffer[:total_size]
                    line = json.dumps(
                        {
                            "type": "wav",
                            "b64": base64.b64encode(wav_bytes).decode("ascii"),
                        },
                        ensure_ascii=False,
                    ).encode("utf-8") + b"\n"
                    yield line
            await upstream.aclose()
            yield b'{"type":"end"}\n'

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(iter_ndjson(), media_type="application/x-ndjson", headers=headers)
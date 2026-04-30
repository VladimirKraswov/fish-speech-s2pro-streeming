from __future__ import annotations

import base64
import json
import os
import struct
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Tuple

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
REQUEST_TIMEOUT = httpx.Timeout(
    connect=CONNECT_TIMEOUT,
    read=None,
    write=WRITE_TIMEOUT,
    pool=None,
)
DEFAULT_REFERENCE_ID = os.environ.get("DEFAULT_REFERENCE_ID", "voice")
SEGMENT_PCM_BYTES = int(os.environ.get("STREAM_SEGMENT_PCM_BYTES", "32768"))


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


def _wav_header(
    *,
    sample_rate: int,
    channels: int,
    bits_per_sample: int,
    data_size: int,
) -> bytes:
    if bits_per_sample != 16:
        raise ValueError("Only 16-bit PCM WAV is supported")
    if channels <= 0:
        raise ValueError("channels must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if data_size < 0:
        raise ValueError("data_size must be non-negative")

    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    riff_size = 36 + data_size

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )


def _parse_wav_header(buf: bytes) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse enough of a WAV header to find the PCM data offset.

    Returns (sample_rate, channels, bits_per_sample, data_offset), or None if
    more bytes are needed. This deliberately ignores the data chunk's declared
    size because the upstream streaming header may use 0xFFFFFFFF.
    """
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


def _align_down(value: int, frame_bytes: int) -> int:
    if frame_bytes <= 0:
        return value
    return value - (value % frame_bytes)


def _wav_ndjson_line(
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
    bits_per_sample: int,
) -> bytes:
    wav_bytes = _wav_header(
        sample_rate=sample_rate,
        channels=channels,
        bits_per_sample=bits_per_sample,
        data_size=len(pcm),
    ) + pcm

    return (
        json.dumps(
            {
                "type": "wav",
                "b64": base64.b64encode(wav_bytes).decode("ascii"),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )


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
    effective_reference_id = (reference_id or DEFAULT_REFERENCE_ID).strip()
    payload: dict[str, object] = {
        "text": text,
        "streaming": True,
        "format": "wav",
        "reference_id": effective_reference_id,
    }

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
    Convert the upstream streaming WAV response into NDJSON where each line
    contains one finite WAV segment as base64.

    The upstream /v1/tts streaming path sends one WAV header followed by raw PCM.
    Its RIFF/data sizes may be intentionally set to 0xFFFFFFFF, so this endpoint
    must not wait for complete upstream RIFF files. Instead it parses the first
    header once, chunks raw PCM, and wraps each PCM chunk in a fresh finite WAV.
    """
    effective_reference_id = (reference_id or DEFAULT_REFERENCE_ID).strip()
    payload: dict[str, object] = {
        "text": text,
        "streaming": True,
        "format": "wav",
        "reference_id": effective_reference_id,
    }

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
        header_buffer = bytearray()
        pending_pcm = bytearray()
        header_parsed = False
        sample_rate = 0
        channels = 0
        bits_per_sample = 0
        frame_bytes = 2
        target_pcm_bytes = max(512, SEGMENT_PCM_BYTES)

        async def flush_segments(force: bool = False) -> AsyncIterator[bytes]:
            nonlocal pending_pcm

            if not header_parsed:
                return

            threshold = _align_down(target_pcm_bytes, frame_bytes)
            if threshold <= 0:
                threshold = frame_bytes

            while pending_pcm:
                if not force and len(pending_pcm) < threshold:
                    return

                out_len = len(pending_pcm) if force else threshold
                out_len = _align_down(out_len, frame_bytes)
                if out_len <= 0:
                    return

                pcm = bytes(pending_pcm[:out_len])
                del pending_pcm[:out_len]

                if pcm:
                    yield _wav_ndjson_line(
                        pcm,
                        sample_rate=sample_rate,
                        channels=channels,
                        bits_per_sample=bits_per_sample,
                    )

        try:
            async for chunk in upstream.aiter_bytes():
                if not chunk:
                    continue

                if not header_parsed:
                    header_buffer.extend(chunk)
                    parsed = _parse_wav_header(header_buffer)

                    if parsed is None:
                        continue

                    sample_rate, channels, bits_per_sample, data_offset = parsed
                    if bits_per_sample != 16:
                        yield (
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": (
                                        f"unsupported bits_per_sample={bits_per_sample}"
                                    ),
                                },
                                ensure_ascii=False,
                            ).encode("utf-8")
                            + b"\n"
                        )
                        return

                    frame_bytes = channels * (bits_per_sample // 8)
                    header_parsed = True

                    pcm = bytes(header_buffer[data_offset:])
                    header_buffer.clear()
                    if pcm:
                        pending_pcm.extend(pcm)

                    async for line in flush_segments(force=False):
                        yield line

                else:
                    pending_pcm.extend(chunk)

                    async for line in flush_segments(force=False):
                        yield line

        except (httpx.RemoteProtocolError, httpx.ReadError):
            # Use everything already received; some upstreams end chunked WAV
            # streams without a terminating chunk.
            pass
        except Exception as exc:
            yield (
                json.dumps(
                    {
                        "type": "error",
                        "message": str(exc),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
            )
            return
        finally:
            try:
                async for line in flush_segments(force=True):
                    yield line

                if not header_parsed:
                    yield (
                        json.dumps(
                            {
                                "type": "error",
                                "message": (
                                    "upstream finished before WAV data header was parsed"
                                ),
                            },
                            ensure_ascii=False,
                        ).encode("utf-8")
                        + b"\n"
                    )

                yield b'{"type":"end"}\n'
            finally:
                await upstream.aclose()

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        iter_ndjson(),
        media_type="application/x-ndjson",
        headers=headers,
    )

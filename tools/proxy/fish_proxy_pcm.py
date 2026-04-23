# tools/proxy/fish_proxy_pcm.py
import asyncio
import base64
import json
import logging
import struct
import time
import uuid
from typing import AsyncGenerator, Optional, Tuple

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

UPSTREAM_TTS_URL = "http://127.0.0.1:8080/v1/tts"
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("fish-proxy")

app = FastAPI(title="Fish Speech PCM Proxy", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_wav_header(buf: bytes) -> Optional[Tuple[int, int, int, int]]:
    """
    Parse enough of a RIFF/WAVE header to find:
      - sample_rate
      - channels
      - bits_per_sample
      - data_offset (start of PCM payload)

    For streaming WAV, return as soon as the `data` chunk header is present.
    """
    if len(buf) < 12:
        return None

    if buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise ValueError("Upstream did not return a RIFF/WAVE stream")

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
                raise ValueError("Invalid WAV fmt chunk")

            audio_format, channels, sample_rate = struct.unpack_from(
                "<HHI", buf, chunk_data_start
            )
            bits_per_sample = struct.unpack_from("<H", buf, chunk_data_start + 14)[0]

            if audio_format != 1:
                raise ValueError(f"Only PCM WAV is supported, got format={audio_format}")

            pos = needed

        elif chunk_id == b"data":
            if channels is None or sample_rate is None or bits_per_sample is None:
                raise ValueError("WAV data chunk arrived before fmt chunk")

            return sample_rate, channels, bits_per_sample, chunk_data_start

        else:
            needed = chunk_data_start + chunk_size + (chunk_size % 2)
            if len(buf) < needed:
                return None
            pos = needed


@app.get("/")
async def root():
    return {"ok": True, "service": "fish-pcm-proxy"}


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/pcm-stream")
async def pcm_stream(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str | None = Query(default=None),
):
    req_id = uuid.uuid4().hex[:8]
    payload = {"text": text, "streaming": True}
    if reference_id:
        payload["reference_id"] = reference_id

    logger.info("REQ %s start text_len=%s reference_id=%s", req_id, len(text), reference_id)

    async def emit(obj: dict) -> bytes:
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    async def body_iter() -> AsyncGenerator[bytes, None]:
        started_at = time.perf_counter()

        # До парсинга WAV header копим тут.
        header_buffer = bytearray()

        # После парсинга header складываем PCM сюда и шлём батчами.
        pending_pcm = bytearray()

        header_sent = False
        sample_rate = None
        channels = None
        sample_width = None

        upstream_bytes = 0
        pcm_bytes_sent = 0
        pcm_chunks_sent = 0

        # 4096 байт = ~46 мс при 44.1kHz mono s16
        # Достаточно мало для низкой задержки и достаточно крупно,
        # чтобы не убивать playback сотнями микрочанков.
        TARGET_EMIT_BYTES = 4096

        async def flush_pending(force: bool = False) -> AsyncGenerator[bytes, None]:
            nonlocal pcm_chunks_sent, pcm_bytes_sent

            while len(pending_pcm) >= TARGET_EMIT_BYTES or (force and pending_pcm):
                if len(pending_pcm) >= TARGET_EMIT_BYTES:
                    out = bytes(pending_pcm[:TARGET_EMIT_BYTES])
                    del pending_pcm[:TARGET_EMIT_BYTES]
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

        # Сразу даём понять клиенту, что прокси жив.
        yield await emit({"type": "proxy_start", "req_id": req_id})
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
                        logger.error("REQ %s upstream status=%s body=%s", req_id, upstream.status_code, msg)
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
                            except Exception as exc:
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
                                logger.error("REQ %s unsupported bits_per_sample=%s", req_id, bits_per_sample)
                                yield await emit(
                                    {
                                        "type": "error",
                                        "req_id": req_id,
                                        "message": f"unsupported bits_per_sample={bits_per_sample}",
                                    }
                                )
                                return

                            sample_width = bits_per_sample // 8
                            header_sent = True

                            logger.info(
                                "REQ %s meta sr=%s ch=%s sw=%s data_offset=%s",
                                req_id,
                                sample_rate,
                                channels,
                                sample_width,
                                data_offset,
                            )

                            yield await emit(
                                {
                                    "type": "meta",
                                    "req_id": req_id,
                                    "sample_rate": sample_rate,
                                    "channels": channels,
                                    "sample_width": sample_width,
                                }
                            )
                            await asyncio.sleep(0)

                            # Всё, что уже пришло после WAV header, это PCM.
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

        except Exception as exc:
            logger.exception("REQ %s stream failed: %s", req_id, exc)
            yield await emit(
                {
                    "type": "error",
                    "req_id": req_id,
                    "message": f"stream failed: {exc}",
                }
            )
            return

        # Досылаем хвост.
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

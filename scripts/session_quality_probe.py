#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


try:
    import websockets
except ImportError as exc:  # pragma: no cover - environment hint
    raise SystemExit(
        "Missing dependency: websockets. Run inside .venv-session or install websockets."
    ) from exc


SCENARIOS = {
    "short": "Привет. Это короткая проверка качества речи.",
    "medium": (
        "Сегодня мы проверяем потоковую озвучку. Текст должен делиться только на "
        "безопасных границах, чтобы слова не рубились. Каждый аудио чанк сохраняется "
        "отдельно, а затем мы сравниваем склеенный итог."
    ),
    "long": (
        "Сначала приходит несколько коротких дельт текста, затем появляется более "
        "длинная фраза без резких пауз, и система должна спокойно дождаться хорошей "
        "границы для коммита. После этого мы добавляем ещё пару предложений, чтобы "
        "проверить очередь TTS, стабильность transport layer и отсутствие дыр между "
        "аудио фрагментами. Финальная часть нужна для проверки хвоста фразы."
    ),
}


@dataclass
class ChunkCapture:
    chunk_id: str
    text: str = ""
    reason: str = ""
    queued_at: float | None = None
    started_at: float | None = None
    first_audio_at: float | None = None
    finished_at: float | None = None
    total_bytes_reported: int = 0
    audio: bytearray = field(default_factory=bytearray)
    debug_wav_path: str | None = None


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"\S+\s*", text)


def _write_wav(path: Path, pcm: bytes, *, sample_rate: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def _gap_stats(times: list[float]) -> dict[str, float | None]:
    if len(times) < 2:
        return {"avg_ms": None, "p95_ms": None, "max_ms": None}
    gaps = [(b - a) * 1000.0 for a, b in zip(times, times[1:])]
    p95 = sorted(gaps)[int(max(0, min(len(gaps) - 1, round(len(gaps) * 0.95) - 1)))]
    return {
        "avg_ms": round(statistics.fmean(gaps), 1),
        "p95_ms": round(p95, 1),
        "max_ms": round(max(gaps), 1),
    }


async def run_scenario(
    *,
    ws_url: str,
    tts_base_url: str,
    name: str,
    text: str,
    out_dir: Path,
    sample_rate: int,
    channels: int,
    stream_chunk_size: int,
    initial_stream_chunk_size: int,
    min_words: int,
    soft_limit_chars: int,
    hard_limit_chars: int,
    timeout_sec: float,
) -> dict[str, Any]:
    scenario_dir = out_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    chunks: dict[str, ChunkCapture] = {}
    pending_audio_event: dict[str, Any] | None = None
    audio_event_times: list[float] = []
    session_id: str | None = None
    done = asyncio.Event()
    started_at = time.perf_counter()
    first_audio_at: float | None = None
    last_error: str | None = None

    async with websockets.connect(ws_url, max_size=None) as ws:
        async def reader() -> None:
            nonlocal pending_audio_event, session_id, first_audio_at, last_error

            async for raw in ws:
                now = time.perf_counter()
                if isinstance(raw, bytes):
                    if pending_audio_event is None:
                        continue
                    chunk_id = str(pending_audio_event.get("chunk_id") or "unknown")
                    cap = chunks.setdefault(chunk_id, ChunkCapture(chunk_id=chunk_id))
                    if cap.first_audio_at is None:
                        cap.first_audio_at = now
                    if first_audio_at is None:
                        first_audio_at = now
                    cap.audio.extend(raw)
                    audio_event_times.append(now)
                    pending_audio_event = None
                    continue

                msg = json.loads(raw)
                event = msg.get("event") or msg.get("type")

                if event == "session_started":
                    session_id = msg.get("session_id")
                elif event == "chunk_queued":
                    chunk = msg.get("chunk") or {}
                    chunk_id = str(chunk.get("chunk_id"))
                    chunks[chunk_id] = ChunkCapture(
                        chunk_id=chunk_id,
                        text=chunk.get("text") or "",
                        reason=chunk.get("reason") or "",
                        queued_at=now,
                    )
                elif event == "tts_started":
                    cap = chunks.setdefault(
                        str(msg.get("chunk_id")),
                        ChunkCapture(chunk_id=str(msg.get("chunk_id"))),
                    )
                    cap.started_at = now
                    cap.text = cap.text or msg.get("text") or ""
                elif event == "audio_chunk":
                    pending_audio_event = msg
                elif event == "audio_debug_saved":
                    cap = chunks.setdefault(
                        str(msg.get("chunk_id")),
                        ChunkCapture(chunk_id=str(msg.get("chunk_id"))),
                    )
                    cap.debug_wav_path = msg.get("wav_path")
                elif event == "tts_finished":
                    cap = chunks.setdefault(
                        str(msg.get("chunk_id")),
                        ChunkCapture(chunk_id=str(msg.get("chunk_id"))),
                    )
                    cap.finished_at = now
                    cap.total_bytes_reported = int(msg.get("total_bytes") or 0)
                    if msg.get("final"):
                        done.set()
                elif event == "error":
                    last_error = f"{msg.get('code')}: {msg.get('message')}"
                    done.set()

        reader_task = asyncio.create_task(reader())

        await ws.send(
            json.dumps(
                {
                    "type": "start_session",
                    "config": {
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "tts": {
                            "base_url": tts_base_url,
                            "format": "pcm",
                            "streaming": True,
                            "stream_tokens": True,
                            "stream_chunk_size": stream_chunk_size,
                            "initial_stream_chunk_size": initial_stream_chunk_size,
                            "cleanup_mode": "session_idle",
                            "use_memory_cache": "off",
                            "normalize": True,
                        },
                        "buffer": {
                            "min_words": min_words,
                            "soft_limit_chars": soft_limit_chars,
                            "hard_limit_chars": hard_limit_chars,
                        },
                        "policy": {
                            "force_flush_after_sec": 0.85,
                            "cleanup_after_idle_sec": 2.0,
                            "session_idle_timeout_sec": 15.0,
                            "close_tts_stream_on_new_text": False,
                        },
                    },
                },
                ensure_ascii=False,
            )
        )

        toks = _tokens(text)
        idx = 0
        while idx < len(toks):
            step = 2 if name == "short" else 3
            delta = "".join(toks[idx : idx + step])
            idx += step
            await ws.send(
                json.dumps(
                    {
                        "type": "text_delta",
                        "text": delta,
                        "final": idx >= len(toks),
                    },
                    ensure_ascii=False,
                )
            )
            await asyncio.sleep(0.06)

        try:
            await asyncio.wait_for(done.wait(), timeout=timeout_sec)
        finally:
            await ws.send(
                json.dumps(
                    {"type": "close_session", "reason": "session_quality_probe"},
                    ensure_ascii=False,
                )
            )
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task

    joined = bytearray()
    chunk_summaries = []
    for chunk_id, cap in chunks.items():
        if cap.audio:
            wav_path = scenario_dir / f"{chunk_id}.wav"
            _write_wav(wav_path, bytes(cap.audio), sample_rate=sample_rate, channels=channels)
            joined.extend(cap.audio)
        else:
            wav_path = None

        chunk_summaries.append(
            {
                "chunk_id": chunk_id,
                "reason": cap.reason,
                "text": cap.text,
                "bytes": len(cap.audio),
                "reported_bytes": cap.total_bytes_reported,
                "local_wav": str(wav_path) if wav_path else None,
                "server_debug_wav": cap.debug_wav_path,
                "ttfa_ms": (
                    round((cap.first_audio_at - cap.queued_at) * 1000.0, 1)
                    if cap.first_audio_at is not None and cap.queued_at is not None
                    else None
                ),
            }
        )

    joined_path = scenario_dir / "_joined.wav"
    if joined:
        _write_wav(joined_path, bytes(joined), sample_rate=sample_rate, channels=channels)

    total = time.perf_counter() - started_at
    return {
        "scenario": name,
        "session_id": session_id,
        "ok": last_error is None and bool(joined),
        "error": last_error,
        "ttfa_ms": round((first_audio_at - started_at) * 1000.0, 1)
        if first_audio_at is not None
        else None,
        "total_ms": round(total * 1000.0, 1),
        "chunks": chunk_summaries,
        "chunk_count": len(chunk_summaries),
        "audio_bytes": len(joined),
        "joined_wav": str(joined_path) if joined else None,
        "inter_audio_gap_ms": _gap_stats(audio_event_times),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Session-mode TTS quality probe")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--tts-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--scenario", choices=[*SCENARIOS.keys(), "all"], default="all")
    parser.add_argument("--out-dir", type=Path, default=Path("results/session_quality"))
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--stream-chunk-size", type=int, default=12)
    parser.add_argument("--initial-stream-chunk-size", type=int, default=16)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--soft-limit-chars", type=int, default=180)
    parser.add_argument("--hard-limit-chars", type=int, default=320)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    args = parser.parse_args()

    out_dir = args.out_dir / time.strftime("%Y%m%d_%H%M%S")
    selected = SCENARIOS.keys() if args.scenario == "all" else [args.scenario]

    results = []
    for name in selected:
        result = await run_scenario(
            ws_url=args.ws_url,
            tts_base_url=args.tts_base_url,
            name=name,
            text=SCENARIOS[name],
            out_dir=out_dir,
            sample_rate=args.sample_rate,
            channels=args.channels,
            stream_chunk_size=args.stream_chunk_size,
            initial_stream_chunk_size=args.initial_stream_chunk_size,
            min_words=args.min_words,
            soft_limit_chars=args.soft_limit_chars,
            hard_limit_chars=args.hard_limit_chars,
            timeout_sec=args.timeout_sec,
        )
        results.append(result)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Session Quality Probe",
        "",
        "| scenario | ok | ttfa_ms | total_ms | chunks | bytes | joined_wav |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        md_lines.append(
            "| {scenario} | {ok} | {ttfa_ms} | {total_ms} | {chunk_count} | {audio_bytes} | {joined_wav} |".format(
                **result
            )
        )
    report_path = out_dir / "summary.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    asyncio.run(main())

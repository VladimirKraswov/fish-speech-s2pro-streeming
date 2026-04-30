from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


FULL_TEXT = "Что такое нейросети и какие виды вы знаете?"
PREFIX_TEXT = "Что такое"
TAIL_TEXT = "нейросети и какие виды вы знаете?"

PRIMARY_METRIC = "append_sent_to_first_pcm"


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base, ensure_ascii=False))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_config(reference_id: str, *, extreme: bool = False) -> str:
    config: dict[str, Any] = {
        "tts": {
            "reference_id": reference_id,
            "stateful_synthesis": True,
            "stream_tokens": True,
        },
        "intro_cache": {
            "enabled": False,
            "warm_on_session_open": False,
        },
    }

    if extreme:
        config = deep_merge(
            config,
            {
                "tts": {
                    "initial_stream_chunk_size": 4,
                    "stream_chunk_size": 4,
                    "first_initial_stream_chunk_size": 4,
                    "first_stream_chunk_size": 4,
                },
                "commit": {
                    "first": {
                        "min_chars": 1,
                        "target_chars": 12,
                        "max_chars": 32,
                        "max_wait_ms": 1,
                        "allow_partial_after_ms": 1,
                    }
                },
                "playback": {
                    "target_emit_bytes": 1024,
                    "first_commit_target_emit_bytes": 1024,
                    "start_buffer_ms": 0,
                    "first_commit_start_buffer_ms": 0,
                    "client_start_buffer_ms": 0,
                    "client_initial_start_delay_ms": 0,
                    "stop_grace_ms": 0,
                    "boundary_smoothing_enabled": False,
                    "punctuation_pauses_enabled": False,
                    "fade_in_ms": 0,
                    "fade_out_ms": 0,
                },
            },
        )

    return json.dumps(config, ensure_ascii=False, separators=(",", ":"))


@dataclass
class RunCapture:
    timestamps: dict[str, float] = field(default_factory=dict)
    audio_meta: dict[str, Any] | None = None

    prefix_pcm: bytearray = field(default_factory=bytearray)
    generation_pcm: bytearray = field(default_factory=bytearray)
    combined_pcm: bytearray = field(default_factory=bytearray)

    total_pcm_bytes: int = 0
    prefix_cache_pcm_bytes: int = 0
    generation_pcm_bytes: int = 0

    def mark_once(self, name: str, ts: float) -> None:
        self.timestamps.setdefault(name, ts)


def ms_between(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start) * 1000.0, 2)


def write_wav(path: Path, pcm: bytes, audio_meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(int(audio_meta["channels"]))
        wav.setsampwidth(int(audio_meta["sample_width"]))
        wav.setframerate(int(audio_meta["sample_rate"]))
        wav.writeframes(pcm)


async def post_json(
    client: httpx.AsyncClient,
    path: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    resp = await client.post(path, json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if resp.status_code >= 400:
        raise RuntimeError(f"{path} failed status={resp.status_code}: {resp.text}")

    return resp.json(), elapsed_ms


async def collect_events(
    client: httpx.AsyncClient,
    session_id: str,
    events_path: Path,
    capture: RunCapture,
    session_started: asyncio.Event,
) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)

    async with client.stream("GET", f"/session/{session_id}/pcm-stream") as resp:
        if resp.status_code >= 400:
            body = await resp.aread()
            raise RuntimeError(
                f"pcm-stream failed status={resp.status_code}: "
                f"{body.decode('utf-8', errors='replace')}"
            )

        with events_path.open("w", encoding="utf-8") as f:
            async for line in resp.aiter_lines():
                if not line:
                    continue

                now = time.perf_counter()
                event = json.loads(line)
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
                f.flush()

                event_type = event.get("type")

                if event_type == "session_start":
                    capture.mark_once("session_start", now)
                    session_started.set()

                elif event_type == "meta":
                    capture.audio_meta = {
                        "sample_rate": event["sample_rate"],
                        "channels": event["channels"],
                        "sample_width": event["sample_width"],
                    }

                elif event_type == "prefix_cache_context_preloaded":
                    capture.mark_once("prefix_cache_context_preloaded", now)

                elif event_type == "prefix_cache_start":
                    capture.mark_once("prefix_cache_start", now)

                elif event_type == "prefix_cache_done":
                    capture.mark_once("prefix_cache_done", now)

                elif event_type == "commit_start" and event.get("commit_seq") == 1:
                    capture.mark_once("commit1_start", now)

                elif event_type == "commit_done" and event.get("commit_seq") == 1:
                    capture.mark_once("commit1_done", now)

                elif event_type == "session_done":
                    capture.mark_once("session_done", now)

                elif event_type in {"error", "intro_error", "upstream_reset_failed"}:
                    capture.mark_once(f"error_{event_type}", now)

                if event_type != "pcm":
                    continue

                chunk = base64.b64decode(event.get("data") or "")
                if not chunk:
                    continue

                # Главная точка: первый любой звук, который пользователь может услышать.
                capture.mark_once("first_pcm", now)

                capture.combined_pcm.extend(chunk)
                capture.total_pcm_bytes += len(chunk)

                if event.get("prefix_cache") is True:
                    capture.mark_once("first_prefix_pcm", now)
                    capture.prefix_pcm.extend(chunk)
                    capture.prefix_cache_pcm_bytes += len(chunk)
                else:
                    capture.mark_once("first_gen_pcm", now)
                    capture.generation_pcm.extend(chunk)
                    capture.generation_pcm_bytes += len(chunk)


async def run_variant(
    client: httpx.AsyncClient,
    *,
    name: str,
    config_text: str,
    append_payload: dict[str, Any],
    out_dir: Path,
    stream_ready_timeout_sec: float,
) -> dict[str, Any]:
    run_dir = out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    open_data, session_open_http = await post_json(
        client,
        "/session/open",
        {"config_text": config_text},
    )
    session_id = open_data["session_id"]

    capture = RunCapture()
    session_started = asyncio.Event()

    stream_task = asyncio.create_task(
        collect_events(
            client,
            session_id,
            run_dir / "events.ndjson",
            capture,
            session_started,
        )
    )

    await asyncio.wait_for(
        session_started.wait(),
        timeout=stream_ready_timeout_sec,
    )

    append_sent = time.perf_counter()
    capture.mark_once("append_sent", append_sent)

    append_data, append_http = await post_json(
        client,
        f"/session/{session_id}/append",
        append_payload,
    )

    finish_data, finish_http = await post_json(
        client,
        f"/session/{session_id}/finish",
        {"reason": "input_finished"},
    )

    await stream_task

    if capture.audio_meta is None:
        raise RuntimeError(f"{name}: stream did not emit audio meta")

    prefix_wav = run_dir / "prefix_cache.wav"
    generation_wav = run_dir / "generation.wav"
    combined_wav = run_dir / "combined.wav"

    # Создаём все WAV всегда, чтобы структура run была одинаковой.
    write_wav(prefix_wav, bytes(capture.prefix_pcm), capture.audio_meta)
    write_wav(generation_wav, bytes(capture.generation_pcm), capture.audio_meta)
    write_wav(combined_wav, bytes(capture.combined_pcm), capture.audio_meta)

    timestamps = capture.timestamps

    append_sent_to_first_pcm = ms_between(
        timestamps.get("append_sent"),
        timestamps.get("first_pcm"),
    )

    key_metrics = {
        # ОСНОВНАЯ МЕТРИКА ПОЛЬЗОВАТЕЛЬСКОГО TTFA.
        "append_sent_to_first_pcm": append_sent_to_first_pcm,

        "session_open_http": round(session_open_http, 2),
        "append_http": round(append_http, 2),
        "finish_http": round(finish_http, 2),

        "session_start_to_first_pcm": ms_between(
            timestamps.get("session_start"),
            timestamps.get("first_pcm"),
        ),
        "prefix_cache_start_to_first_prefix_pcm": ms_between(
            timestamps.get("prefix_cache_start"),
            timestamps.get("first_prefix_pcm"),
        ),
        "prefix_cache_done_to_commit1_start": ms_between(
            timestamps.get("prefix_cache_done"),
            timestamps.get("commit1_start"),
        ),
        "commit1_start_to_first_gen_pcm_TTFA": ms_between(
            timestamps.get("commit1_start"),
            timestamps.get("first_gen_pcm"),
        ),
        "append_sent_to_first_gen_pcm": ms_between(
            timestamps.get("append_sent"),
            timestamps.get("first_gen_pcm"),
        ),
        "append_sent_to_session_done": ms_between(
            timestamps.get("append_sent"),
            timestamps.get("session_done"),
        ),

        "total_pcm_bytes": capture.total_pcm_bytes,
        "prefix_cache_pcm_bytes": capture.prefix_cache_pcm_bytes,
        "generation_pcm_bytes": capture.generation_pcm_bytes,
    }

    metrics = {
        "run": name,
        "session_id": session_id,
        "synthesis_session_id": open_data.get("synthesis_session_id"),

        "primary_metric": {
            "name": PRIMARY_METRIC,
            "value_ms": append_sent_to_first_pcm,
            "meaning": (
                "Главный пользовательский TTFA: время от отправки /append "
                "до первого PCM-аудио любого типа, то есть до первого звука."
            ),
        },

        "key_metrics": key_metrics,

        # Дублируем наверх для удобного grep/jq.
        "append_sent_to_first_pcm": append_sent_to_first_pcm,

        "http": {
            "session_open_http": round(session_open_http, 2),
            "append_http": round(append_http, 2),
            "finish_http": round(finish_http, 2),
        },

        "payloads": {
            "append_payload": append_payload,
            "append_response": append_data,
            "finish_response": finish_data,
        },

        "timestamps_present": sorted(timestamps.keys()),

        "bytes": {
            "total_pcm_bytes": capture.total_pcm_bytes,
            "prefix_cache_pcm_bytes": capture.prefix_cache_pcm_bytes,
            "generation_pcm_bytes": capture.generation_pcm_bytes,
        },

        "audio_meta": capture.audio_meta,

        "files": {
            "events_ndjson": str(run_dir / "events.ndjson"),
            "metrics_json": str(run_dir / "metrics.json"),
            "prefix_cache_wav": str(prefix_wav),
            "generation_wav": str(generation_wav),
            "combined_wav": str(combined_wav),
        },
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return metrics


def print_primary_summary(summary: dict[str, Any]) -> None:
    runs = summary["runs"]

    print("\n" + "=" * 88)
    print("ГЛАВНАЯ МЕТРИКА ПОЛЬЗОВАТЕЛЬСКОГО TTFA")
    print("=" * 88)
    print(f"{PRIMARY_METRIC} = время от отправки /append до первого PCM-аудио")
    print("-" * 88)

    for run in runs:
        value = run.get("append_sent_to_first_pcm")
        print(f"{run['run']:28s} | {PRIMARY_METRIC:30s} | {str(value):>10s} ms")

    print("=" * 88)

    print("\nKEY METRICS")
    print("-" * 88)

    metric_order = [
        "append_sent_to_first_pcm",
        "session_open_http",
        "append_http",
        "finish_http",
        "session_start_to_first_pcm",
        "prefix_cache_start_to_first_prefix_pcm",
        "prefix_cache_done_to_commit1_start",
        "commit1_start_to_first_gen_pcm_TTFA",
        "append_sent_to_first_gen_pcm",
        "append_sent_to_session_done",
        "total_pcm_bytes",
        "prefix_cache_pcm_bytes",
        "generation_pcm_bytes",
    ]

    for run in runs:
        print(f"\n[{run['run']}]")
        km = run["key_metrics"]

        for metric in metric_order:
            value = km.get(metric)
            marker = ">>> " if metric == PRIMARY_METRIC else "    "
            suffix = "  <-- PRIMARY USER TTFA" if metric == PRIMARY_METRIC else ""
            print(f"{marker}{metric:42s}: {value}{suffix}")

        print("    files:")
        print(f"      events:       {run['files']['events_ndjson']}")
        print(f"      metrics:      {run['files']['metrics_json']}")
        print(f"      prefix_cache: {run['files']['prefix_cache_wav']}")
        print(f"      generation:   {run['files']['generation_wav']}")
        print(f"      combined:     {run['files']['combined_wav']}")


async def async_main(args: argparse.Namespace) -> int:
    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")

    normal_config_text = build_config(args.reference_id, extreme=False)
    extreme_config_text = build_config(args.reference_id, extreme=True)

    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"),
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None),
    ) as client:
        add_data, prefix_cache_add_http = await post_json(
            client,
            "/prefix-cache/add",
            {
                "config_text": normal_config_text,
                "texts": [PREFIX_TEXT],
                "clear_existing": args.clear_existing,
                "fail_fast": True,
            },
        )

        if add_data.get("failed_count"):
            raise RuntimeError(
                "prefix-cache/add failed: "
                + json.dumps(add_data, ensure_ascii=False, indent=2)
            )

        metrics: list[dict[str, Any]] = []

        metrics.append(
            await run_variant(
                client,
                name="no_cache_extreme_ttfa",
                config_text=extreme_config_text,
                append_payload={"text": FULL_TEXT},
                out_dir=out_dir,
                stream_ready_timeout_sec=args.stream_ready_timeout_sec,
            )
        )

        metrics.append(
            await run_variant(
                client,
                name="no_cache_normal_ttfa",
                config_text=normal_config_text,
                append_payload={"text": FULL_TEXT},
                out_dir=out_dir,
                stream_ready_timeout_sec=args.stream_ready_timeout_sec,
            )
        )

        metrics.append(
            await run_variant(
                client,
                name="prefix_cache",
                config_text=normal_config_text,
                append_payload={"cache": PREFIX_TEXT, "text": TAIL_TEXT},
                out_dir=out_dir,
                stream_ready_timeout_sec=args.stream_ready_timeout_sec,
            )
        )

    summary = {
        "out_dir": str(out_dir),

        "primary_metric": {
            "name": PRIMARY_METRIC,
            "meaning": (
                "Главный пользовательский TTFA: время от отправки /append "
                "до первого PCM-аудио любого типа."
            ),
        },

        "prefix_cache_add": {
            "http_ms": round(prefix_cache_add_http, 2),
            "response": add_data,
        },

        "full_text": FULL_TEXT,
        "prefix_cache": PREFIX_TEXT,
        "tail": TAIL_TEXT,

        "runs": metrics,
    }

    (out_dir / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print_primary_summary(summary)

    print("\nsummary_metrics.json:")
    print(out_dir / "summary_metrics.json")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record TTFA and WAV artifacts for explicit prefix-cache sessions."
    )

    parser.add_argument("--base-url", default="http://127.0.0.1:9000")
    parser.add_argument("--reference-id", default="voice")

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("prefix_cache_ttfa_runs"),
    )

    parser.add_argument(
        "--stream-ready-timeout-sec",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear in-memory prefix-cache before adding the test prefix.",
    )

    return parser.parse_args()


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import requests


def get_memory(base_url: str):
    try:
        r = requests.get(f"{base_url}/v1/debug/memory", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--reference-id", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--out-dir", default="logs/step1_stream_ref")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": args.text,
        "reference_id": args.reference_id,
        "references": [],
        "streaming": True,
        "stream_tokens": False,
        "format": "wav",
        "max_new_tokens": 128,
        "chunk_length": 200,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "temperature": 0.8,
        "use_memory_cache": "on",
        "seed": 1234,
    }

    def one_run(run_idx: int, counted: bool):
        before_mem = get_memory(args.base_url)
        started = time.perf_counter()

        with requests.post(
            f"{args.base_url}/v1/tts",
            json=payload,
            stream=True,
            timeout=(10, 300),
        ) as r:
            r.raise_for_status()

            out_path = out_dir / f"run_{run_idx:02d}.wav"
            total = 0
            first_chunk_ms = None
            first_audio_ms = None

            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=4096):
                    if not chunk:
                        continue

                    now_ms = (time.perf_counter() - started) * 1000.0

                    if first_chunk_ms is None:
                        first_chunk_ms = now_ms

                    prev_total = total
                    total += len(chunk)
                    f.write(chunk)

                    # WAV header usually 44 bytes; нас интересует момент прихода первого аудио байта
                    if first_audio_ms is None and total > 44:
                        if prev_total <= 44:
                            first_audio_ms = now_ms

        total_ms = (time.perf_counter() - started) * 1000.0
        after_mem = get_memory(args.base_url)

        result = {
            "counted": counted,
            "run_idx": run_idx,
            "first_chunk_ms": round(first_chunk_ms or -1, 1),
            "first_audio_ms": round(first_audio_ms or -1, 1),
            "total_ms": round(total_ms, 1),
            "bytes_written": total,
            "before_allocated_gb": before_mem.get("allocated_gb"),
            "after_allocated_gb": after_mem.get("allocated_gb"),
            "before_reserved_gb": before_mem.get("reserved_gb"),
            "after_reserved_gb": after_mem.get("reserved_gb"),
            "before_max_allocated_gb": before_mem.get("max_allocated_gb"),
            "after_max_allocated_gb": after_mem.get("max_allocated_gb"),
            "out_path": str(out_path),
        }

        print(json.dumps(result, ensure_ascii=False))
        return result

    if args.warmup:
        print("=== warmup ===")
        one_run(0, counted=False)

    print("=== measured runs ===")
    results = []
    for i in range(1, args.runs + 1):
        results.append(one_run(i, counted=True))

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
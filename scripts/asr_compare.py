#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("KMP_WARNINGS", "0")

from faster_whisper import WhisperModel
from jiwer import cer, wer


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я\s]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR compare for session-quality WAVs")
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("expected_text")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    segments, info = model.transcribe(
        str(args.wav_path),
        language=args.language,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    normalized_expected = normalize_text(args.expected_text)
    normalized_transcript = normalize_text(transcript)

    result = {
        "wav_path": str(args.wav_path),
        "expected_text": args.expected_text,
        "transcript": transcript,
        "normalized_expected": normalized_expected,
        "normalized_transcript": normalized_transcript,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "wer": None,
        "cer": None,
    }

    if normalized_expected:
        result["wer"] = wer(normalized_expected, normalized_transcript)
        result["cer"] = cer(normalized_expected, normalized_transcript)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from fish_speech.codec.codes import save_codes_pt, validate_codes_for_decoder
from fish_speech.driver.api import FishSpeechDriver
from fish_speech.driver.config import load_runtime_config
from fish_speech.driver.types import DriverGenerationOptions, DriverSynthesisRequest
from fish_speech.generation.prompt_builder import GenerateResponse

ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
ARTIFACT_NAMES = ("audio.wav", "audio.pcm", "codes.pt", "meta.json")


@dataclass(frozen=True)
class IntroItem:
    id: str
    text: str


@dataclass(frozen=True)
class BuildSettings:
    llama_checkpoint_path: str
    decoder_checkpoint_path: str
    decoder_config_name: str
    device: str
    precision: torch.dtype
    compile: bool
    reference_id: str | None
    seed: int
    max_new_tokens: int
    chunk_length: int
    top_p: float
    temperature: float
    repetition_penalty: float
    initial_stream_chunk_size: int
    stream_chunk_size: int
    pcm_format: str
    audio_subtype: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached intro artifacts.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file: a list of intros or {'items': [...]}",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for cached intro artifacts.",
    )
    parser.add_argument(
        "--reference-id",
        default=None,
        help="Reference voice ID. Defaults to runtime warmup.reference_id if present.",
    )
    parser.add_argument("--llama-checkpoint-path", default=None)
    parser.add_argument("--decoder-checkpoint-path", default=None)
    parser.add_argument("--decoder-config-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--precision",
        choices=["float16", "bfloat16", "float32"],
        default=None,
    )
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        default=None,
        help="Enable torch.compile for generation.",
    )

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--chunk-length", type=int, default=160)
    parser.add_argument("--top-p", type=float, default=0.82)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--initial-stream-chunk-size", type=int, default=8)
    parser.add_argument("--stream-chunk-size", type=int, default=8)

    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing intro directories after successful generation.",
    )
    write_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip intros whose artifact directory already exists and is complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and resolved settings without generating artifacts.",
    )
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--pcm-format", choices=["pcm16le"], default="pcm16le")
    parser.add_argument("--audio-subtype", default="PCM_16")

    return parser.parse_args(argv)


def _load_raw_items(input_path: str | Path) -> list[Any]:
    with Path(input_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "items" in payload:
        payload = payload["items"]

    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list or an object with an 'items' list")

    return payload


def load_and_validate_items(input_path: str | Path) -> list[IntroItem]:
    raw_items = _load_raw_items(input_path)
    items: list[IntroItem] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()

    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"Item #{index} must be an object")

        intro_id = str(raw.get("id", "")).strip()
        text = str(raw.get("text", "")).strip()

        if not intro_id:
            raise ValueError(f"Item #{index} has empty id")
        if not ID_RE.match(intro_id):
            raise ValueError(
                f"Invalid intro id {intro_id!r}; use only [a-zA-Z0-9_-]"
            )
        if intro_id in seen_ids:
            raise ValueError(f"Duplicate intro id: {intro_id}")
        seen_ids.add(intro_id)

        if not text:
            raise ValueError(f"Item {intro_id!r} has empty text")
        if text in seen_texts:
            logger.warning("Duplicate intro text: {!r}", text[:180])
        seen_texts.add(text)

        items.append(IntroItem(id=intro_id, text=text))

    return items


def _resolve_settings(args: argparse.Namespace) -> BuildSettings:
    runtime = load_runtime_config()
    model_cfg = runtime.model
    paths_cfg = runtime.paths
    warmup_cfg = runtime.warmup

    precision_name = args.precision or model_cfg.precision
    precision = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[precision_name]

    compile_value = model_cfg.compile if args.compile is None else bool(args.compile)

    return BuildSettings(
        llama_checkpoint_path=(
            args.llama_checkpoint_path or paths_cfg.llama_checkpoint_path
        ),
        decoder_checkpoint_path=(
            args.decoder_checkpoint_path or paths_cfg.decoder_checkpoint_path
        ),
        decoder_config_name=args.decoder_config_name or paths_cfg.decoder_config_name,
        device=args.device or model_cfg.device,
        precision=precision,
        compile=compile_value,
        reference_id=args.reference_id or warmup_cfg.reference_id,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        chunk_length=args.chunk_length,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        initial_stream_chunk_size=args.initial_stream_chunk_size,
        stream_chunk_size=args.stream_chunk_size,
        pcm_format=args.pcm_format,
        audio_subtype=args.audio_subtype,
    )


def _intro_complete(intro_dir: Path) -> bool:
    return intro_dir.is_dir() and all((intro_dir / name).is_file() for name in ARTIFACT_NAMES)


def _manifest_entry_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": meta["id"],
        "text": meta["text"],
        "path": meta["id"],
        "code_frames": meta["code_frames"],
        "duration_ms": meta["duration_ms"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_manifest(
    output_dir: Path,
    *,
    manifest_name: str,
    sample_rate: int | None,
    pcm_format: str,
    entries: list[dict[str, Any]],
) -> None:
    manifest = {
        "version": 1,
        "sample_rate": sample_rate,
        "format": pcm_format,
        "items": entries,
    }
    manifest_path = output_dir / manifest_name
    tmp_path = output_dir / f".{manifest_name}.tmp"
    _write_json(tmp_path, manifest)
    tmp_path.replace(manifest_path)


def _load_existing_manifest_entry(intro_dir: Path) -> dict[str, Any] | None:
    meta_path = intro_dir / "meta.json"
    if not meta_path.is_file():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return _manifest_entry_from_meta(meta)


def _pcm16le(audio: np.ndarray) -> bytes:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    return pcm16.tobytes()


def _validate_artifact_files(intro_dir: Path) -> None:
    missing = [name for name in ARTIFACT_NAMES if not (intro_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing cached intro artifacts in {intro_dir}: {missing}")


def _build_driver(settings: BuildSettings) -> FishSpeechDriver:
    return FishSpeechDriver.from_model_paths(
        llama_checkpoint_path=settings.llama_checkpoint_path,
        decoder_checkpoint_path=settings.decoder_checkpoint_path,
        decoder_config_name=settings.decoder_config_name,
        device=settings.device,
        precision=settings.precision,
        compile=settings.compile,
    )


def _generate_intro(
    *,
    driver: FishSpeechDriver,
    item: IntroItem,
    intro_dir: Path,
    settings: BuildSettings,
    overwrite: bool,
    skip_existing: bool,
) -> tuple[str, dict[str, Any] | None]:
    if intro_dir.exists():
        if skip_existing and _intro_complete(intro_dir):
            logger.info("Skipping existing intro: {}", item.id)
            return "skipped", _load_existing_manifest_entry(intro_dir)
        if not overwrite:
            raise FileExistsError(
                f"Intro {item.id!r} already exists; use --overwrite or --skip-existing"
            )

    tmp_dir = intro_dir.parent / f".{item.id}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    try:
        request = DriverSynthesisRequest(
            text=item.text,
            segments=[item.text],
            reference_id=settings.reference_id,
            seed=settings.seed,
            use_memory_cache="on",
            normalize=True,
            stream_audio=False,
            generation=DriverGenerationOptions(
                chunk_length=settings.chunk_length,
                max_new_tokens=settings.max_new_tokens,
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
                temperature=settings.temperature,
                stream_tokens=True,
                initial_stream_chunk_size=settings.initial_stream_chunk_size,
                stream_chunk_size=settings.stream_chunk_size,
            ),
        )

        result = driver.synthesize_with_codes(request)
        codes = result["codes"]
        audio = result["audio"]
        sample_rate = int(result["sample_rate"])

        if codes is None:
            raise RuntimeError("No VQ/DAC codes generated")

        codes = validate_codes_for_decoder(
            codes,
            driver.engine.decoder_model,
            name=f"intro {item.id}",
        )

        if audio is None:
            logger.info("Decoding cached intro audio from generated codes")
            audio = driver.engine.get_audio_segment(
                GenerateResponse(action="sample", codes=codes)
            )

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size <= 0:
            raise RuntimeError("No audio samples generated")

        save_codes_pt(codes, tmp_dir / "codes.pt", name=f"intro {item.id}")
        sf.write(
            tmp_dir / "audio.wav",
            audio,
            sample_rate,
            subtype=settings.audio_subtype,
        )
        (tmp_dir / "audio.pcm").write_bytes(_pcm16le(audio))

        meta = {
            "id": item.id,
            "text": item.text,
            "sample_rate": sample_rate,
            "channels": 1,
            "sample_width": 2,
            "format": settings.pcm_format,
            "code_frames": int(codes.shape[1]),
            "num_codebooks": int(codes.shape[0]),
            "audio_samples": int(audio.size),
            "duration_ms": int(audio.size / sample_rate * 1000),
            "reference_id": settings.reference_id,
            "seed": settings.seed,
            "max_new_tokens": settings.max_new_tokens,
            "chunk_length": settings.chunk_length,
            "top_p": settings.top_p,
            "temperature": settings.temperature,
            "repetition_penalty": settings.repetition_penalty,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(tmp_dir / "meta.json", meta)
        _validate_artifact_files(tmp_dir)

        if intro_dir.exists():
            shutil.rmtree(intro_dir)
        tmp_dir.rename(intro_dir)
        _validate_artifact_files(intro_dir)
        return "generated", _manifest_entry_from_meta(meta)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items = load_and_validate_items(args.input)
    settings = _resolve_settings(args)

    if settings.reference_id is None and not args.dry_run:
        raise SystemExit(
            "--reference-id is required unless runtime warmup.reference_id is set"
        )

    logger.info("Validated {} cached intro item(s)", len(items))
    if args.dry_run:
        logger.info("Dry run complete; no artifacts were written")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    driver = _build_driver(settings)
    entries: list[dict[str, Any]] = []
    generated = 0
    skipped = 0
    failed = 0
    sample_rate: int | None = None

    try:
        for item in items:
            intro_dir = output_dir / item.id
            logger.info("Processing cached intro {} text={!r}", item.id, item.text[:180])
            try:
                status, entry = _generate_intro(
                    driver=driver,
                    item=item,
                    intro_dir=intro_dir,
                    settings=settings,
                    overwrite=args.overwrite,
                    skip_existing=args.skip_existing,
                )
                if status == "generated":
                    generated += 1
                elif status == "skipped":
                    skipped += 1
                if entry is not None:
                    entries.append(entry)
                    meta_path = intro_dir / "meta.json"
                    if meta_path.is_file():
                        with meta_path.open("r", encoding="utf-8") as f:
                            sample_rate = int(json.load(f)["sample_rate"])
            except Exception as exc:
                failed += 1
                logger.exception("Failed to build cached intro {}: {}", item.id, exc)

        _write_manifest(
            output_dir,
            manifest_name=args.manifest_name,
            sample_rate=sample_rate,
            pcm_format=settings.pcm_format,
            entries=entries,
        )

        logger.info(
            "Build finished: generated={} skipped={} failed={} output={}",
            generated,
            skipped,
            failed,
            output_dir,
        )
        return 1 if failed else 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
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

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
MAX_PREFIX_WORDS = 5
ARTIFACT_DIRS = ("pcm", "wav", "codes", "meta")


@dataclass(frozen=True)
class PrefixCacheItem:
    cache_id: str
    voice_id: str
    text: str
    normalized_text: str
    word_count: int


@dataclass(frozen=True)
class BuildSettings:
    llama_checkpoint_path: str
    decoder_checkpoint_path: str
    decoder_config_name: str
    device: str
    precision: torch.dtype
    compile: bool
    default_voice_id: str | None
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


@dataclass(frozen=True)
class ArtifactPaths:
    pcm: Path
    wav: Path
    codes: Path
    meta: Path


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def normalize_prefix_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[“”«»\"'`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip(" \t\r\n,.!?;:")


def params_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prefix-cache TTS artifacts.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file: a list of prefixes or {'items': [...]}",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for prefix-cache artifacts.",
    )
    parser.add_argument(
        "--voice-id",
        default=None,
        help="Default voice/reference ID for all items.",
    )
    parser.add_argument(
        "--reference-id",
        default=None,
        help="Alias for --voice-id.",
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
    parser.add_argument("--cache-version", default="v1")

    write_group = parser.add_mutually_exclusive_group()
    write_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifacts after successful generation.",
    )
    write_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip items whose artifact files already exist.",
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


def _validate_cache_id(cache_id: str, *, item_label: str) -> str:
    cache_id = cache_id.strip()
    if not cache_id:
        raise ValueError(f"{item_label} has empty cache_id")
    if cache_id in {".", ".."} or "/" in cache_id or "\\" in cache_id:
        raise ValueError(
            f"Invalid cache_id {cache_id!r}; path separators are not allowed"
        )
    return cache_id


def _slugify(value: str) -> str:
    value = normalize_prefix_text(value)
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value)
    return value.strip("_").lower()


def _default_cache_id(
    *,
    voice_id: str,
    normalized_text: str,
    cache_version: str,
) -> str:
    voice_slug = _slugify(voice_id) or "voice"
    text_slug = _slugify(normalized_text)
    if not text_slug:
        text_slug = hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()[:8]
    version_slug = _slugify(cache_version) or "v1"
    return f"{voice_slug}_{text_slug}_{version_slug}"


def load_and_validate_items(
    input_path: str | Path,
    *,
    default_voice_id: str | None = None,
    cache_version: str = "v1",
) -> list[PrefixCacheItem]:
    raw_items = _load_raw_items(input_path)
    items: list[PrefixCacheItem] = []
    seen_ids: set[str] = set()
    seen_lookup_keys: set[tuple[str, str]] = set()

    for index, raw in enumerate(raw_items):
        item_label = f"Item #{index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{item_label} must be an object")

        text = str(raw.get("text", "")).strip()
        if not text:
            raise ValueError(f"{item_label} has empty text")

        word_count = count_words(text)
        if word_count <= 0:
            raise ValueError(f"Prefix {text!r} has no words")
        if word_count > MAX_PREFIX_WORDS:
            raise ValueError(
                f"Prefix {text!r} has {word_count} words; max is {MAX_PREFIX_WORDS}"
            )

        voice_id = str(
            raw.get("voice_id") or raw.get("reference_id") or default_voice_id or ""
        ).strip()
        if not voice_id:
            raise ValueError(
                f"{item_label} has no voice_id; pass --voice-id or set item.voice_id"
            )

        normalized_text = normalize_prefix_text(text)
        raw_cache_id = raw.get("cache_id", raw.get("id"))
        if raw_cache_id is None:
            raw_cache_id = _default_cache_id(
                voice_id=voice_id,
                normalized_text=normalized_text,
                cache_version=cache_version,
            )
        cache_id = _validate_cache_id(str(raw_cache_id), item_label=item_label)

        if cache_id in seen_ids:
            raise ValueError(f"Duplicate cache_id: {cache_id}")
        seen_ids.add(cache_id)

        lookup_key = (voice_id, normalized_text)
        if lookup_key in seen_lookup_keys:
            logger.warning(
                "Duplicate prefix lookup key voice_id={} normalized_text={!r}",
                voice_id,
                normalized_text,
            )
        seen_lookup_keys.add(lookup_key)

        items.append(
            PrefixCacheItem(
                cache_id=cache_id,
                voice_id=voice_id,
                text=text,
                normalized_text=normalized_text,
                word_count=word_count,
            )
        )

    return items


def _resolve_settings(args: argparse.Namespace) -> BuildSettings:
    if args.voice_id and args.reference_id and args.voice_id != args.reference_id:
        raise ValueError("--voice-id and --reference-id must match when both are set")

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
        default_voice_id=args.voice_id or args.reference_id or warmup_cfg.reference_id,
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


def _artifact_paths(root: Path, cache_id: str) -> ArtifactPaths:
    return ArtifactPaths(
        pcm=root / "pcm" / f"{cache_id}.pcm",
        wav=root / "wav" / f"{cache_id}.wav",
        codes=root / "codes" / f"{cache_id}.pt",
        meta=root / "meta" / f"{cache_id}.json",
    )


def _relative_artifact_paths(cache_id: str) -> dict[str, str]:
    return {
        "pcm_path": f"pcm/{cache_id}.pcm",
        "wav_path": f"wav/{cache_id}.wav",
        "codes_path": f"codes/{cache_id}.pt",
        "meta_path": f"meta/{cache_id}.json",
    }


def _artifact_complete(output_dir: Path, cache_id: str) -> bool:
    paths = _artifact_paths(output_dir, cache_id)
    return all(
        path.is_file() for path in (paths.pcm, paths.wav, paths.codes, paths.meta)
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_manifest(
    output_dir: Path,
    *,
    manifest_name: str,
    entries: list[dict[str, Any]],
) -> None:
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "prefix-cache",
        "max_words": MAX_PREFIX_WORDS,
        "items": entries,
    }
    manifest_path = output_dir / manifest_name
    tmp_path = output_dir / f".{manifest_name}.tmp"
    _write_json(tmp_path, manifest)
    tmp_path.replace(manifest_path)


def _load_existing_manifest_entry(output_dir: Path, cache_id: str) -> dict[str, Any]:
    meta_path = _artifact_paths(output_dir, cache_id).meta
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["manifest_entry"]


def _pcm16le(audio: np.ndarray) -> bytes:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    return pcm16.tobytes()


def _params_payload(settings: BuildSettings, *, voice_id: str) -> dict[str, Any]:
    return {
        "voice_id": voice_id,
        "reference_id": voice_id,
        "seed": settings.seed,
        "top_p": settings.top_p,
        "temperature": settings.temperature,
        "repetition_penalty": settings.repetition_penalty,
        "chunk_length": settings.chunk_length,
        "max_new_tokens": settings.max_new_tokens,
        "initial_stream_chunk_size": settings.initial_stream_chunk_size,
        "stream_chunk_size": settings.stream_chunk_size,
        "model": {
            "llama_checkpoint_path": settings.llama_checkpoint_path,
            "decoder_checkpoint_path": settings.decoder_checkpoint_path,
            "decoder_config_name": settings.decoder_config_name,
        },
    }


def _build_driver(settings: BuildSettings) -> FishSpeechDriver:
    return FishSpeechDriver.from_model_paths(
        llama_checkpoint_path=settings.llama_checkpoint_path,
        decoder_checkpoint_path=settings.decoder_checkpoint_path,
        decoder_config_name=settings.decoder_config_name,
        device=settings.device,
        precision=settings.precision,
        compile=settings.compile,
    )


def _ensure_artifact_dirs(root: Path) -> None:
    for name in ARTIFACT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def _move_artifacts(tmp_paths: ArtifactPaths, final_paths: ArtifactPaths) -> None:
    for final_path in (
        final_paths.pcm,
        final_paths.wav,
        final_paths.codes,
        final_paths.meta,
    ):
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            final_path.unlink()

    shutil.move(str(tmp_paths.pcm), str(final_paths.pcm))
    shutil.move(str(tmp_paths.wav), str(final_paths.wav))
    shutil.move(str(tmp_paths.codes), str(final_paths.codes))
    shutil.move(str(tmp_paths.meta), str(final_paths.meta))


def _generate_prefix(
    *,
    driver: FishSpeechDriver,
    item: PrefixCacheItem,
    output_dir: Path,
    settings: BuildSettings,
    overwrite: bool,
    skip_existing: bool,
) -> tuple[str, dict[str, Any] | None]:
    final_paths = _artifact_paths(output_dir, item.cache_id)
    if _artifact_complete(output_dir, item.cache_id):
        if skip_existing:
            logger.info("Skipping existing prefix cache item: {}", item.cache_id)
            return "skipped", _load_existing_manifest_entry(output_dir, item.cache_id)
        if not overwrite:
            raise FileExistsError(
                f"Prefix cache item {item.cache_id!r} already exists; "
                "use --overwrite or --skip-existing"
            )
    elif any(
        path.exists()
        for path in (
            final_paths.pcm,
            final_paths.wav,
            final_paths.codes,
            final_paths.meta,
        )
    ):
        if not overwrite:
            raise FileExistsError(
                f"Prefix cache item {item.cache_id!r} has partial artifacts; "
                "use --overwrite to replace them"
            )

    tmp_dir = output_dir / f".{item.cache_id}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    _ensure_artifact_dirs(tmp_dir)
    tmp_paths = _artifact_paths(tmp_dir, item.cache_id)

    try:
        request = DriverSynthesisRequest(
            text=item.text,
            segments=[item.text],
            reference_id=item.voice_id,
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
            name=f"prefix cache {item.cache_id}",
        )

        if audio is None:
            logger.info("Decoding prefix cache audio from generated codes")
            audio = driver.engine.get_audio_segment(
                GenerateResponse(action="sample", codes=codes)
            )

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size <= 0:
            raise RuntimeError("No audio samples generated")

        normalized_codes = save_codes_pt(
            codes,
            tmp_paths.codes,
            name=f"prefix cache {item.cache_id}",
        )
        sf.write(
            tmp_paths.wav,
            audio,
            sample_rate,
            subtype=settings.audio_subtype,
        )
        tmp_paths.pcm.write_bytes(_pcm16le(audio))

        audio_meta = {
            "sample_rate": sample_rate,
            "channels": 1,
            "sample_width": 2,
        }
        param_payload = _params_payload(settings, voice_id=item.voice_id)
        rel_paths = _relative_artifact_paths(item.cache_id)
        manifest_entry = {
            "cache_id": item.cache_id,
            "voice_id": item.voice_id,
            "text": item.text,
            "normalized_text": item.normalized_text,
            "word_count": item.word_count,
            "pcm_path": rel_paths["pcm_path"],
            "codes_path": rel_paths["codes_path"],
            "code_frames": int(normalized_codes.shape[-1]),
            "audio_meta": audio_meta,
            "params_hash": params_hash(param_payload),
            "wav_path": rel_paths["wav_path"],
            "meta_path": rel_paths["meta_path"],
        }
        meta = {
            "manifest_entry": manifest_entry,
            "cache_id": item.cache_id,
            "voice_id": item.voice_id,
            "reference_id": item.voice_id,
            "text": item.text,
            "normalized_text": item.normalized_text,
            "word_count": item.word_count,
            "audio_meta": audio_meta,
            "format": settings.pcm_format,
            "num_codebooks": int(normalized_codes.shape[0]),
            "code_frames": int(normalized_codes.shape[-1]),
            "audio_samples": int(audio.size),
            "duration_ms": int(audio.size / sample_rate * 1000),
            "params_hash": manifest_entry["params_hash"],
            "params": param_payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(tmp_paths.meta, meta)

        _move_artifacts(tmp_paths, final_paths)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if not _artifact_complete(output_dir, item.cache_id):
            raise RuntimeError(f"Incomplete artifacts for {item.cache_id}")
        return "generated", manifest_entry
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = _resolve_settings(args)
    items = load_and_validate_items(
        args.input,
        default_voice_id=settings.default_voice_id,
        cache_version=args.cache_version,
    )

    logger.info("Validated {} prefix cache item(s)", len(items))
    if args.dry_run:
        logger.info("Dry run complete; no artifacts were written")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_artifact_dirs(output_dir)

    driver = _build_driver(settings)
    entries: list[dict[str, Any]] = []
    generated = 0
    skipped = 0
    failed = 0

    try:
        for item in items:
            logger.info(
                "Processing prefix cache {} voice_id={} text={!r}",
                item.cache_id,
                item.voice_id,
                item.text[:180],
            )
            try:
                status, entry = _generate_prefix(
                    driver=driver,
                    item=item,
                    output_dir=output_dir,
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
            except Exception as exc:
                failed += 1
                logger.exception(
                    "Failed to build prefix cache item {}: {}",
                    item.cache_id,
                    exc,
                )

        _write_manifest(
            output_dir,
            manifest_name=args.manifest_name,
            entries=entries,
        )

        logger.info(
            "Prefix cache build finished: generated={} skipped={} failed={} output={}",
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

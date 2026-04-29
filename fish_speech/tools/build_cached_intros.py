import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from fish_speech.codec.codes import save_codes_pt
from fish_speech.driver.api import FishSpeechDriver
from fish_speech.driver.types import DriverGenerationOptions, DriverSynthesisRequest
from fish_speech.generation.prompt_builder import GenerateResponse


def parse_args():
    parser = argparse.ArgumentParser(description="Build cached intro artifacts.")
    parser.add_argument(
        "--input", type=str, required=True, help="Input JSON file with intros."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for cached intros.",
    )
    parser.add_argument(
        "--reference-id", type=str, required=True, help="Reference voice ID."
    )
    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        required=True,
        help="Path to LLaMA checkpoint.",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        required=True,
        help="Path to decoder checkpoint.",
    )
    parser.add_argument(
        "--decoder-config-name",
        type=str,
        default="modded_dac_vq",
        help="Decoder config name.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Precision to use.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=96, help="Maximum new tokens."
    )
    parser.add_argument(
        "--chunk-length", type=int, default=160, help="Chunk length for synthesis."
    )
    parser.add_argument("--top-p", type=float, default=0.82, help="Top-p sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature."
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.03,
        help="Repetition penalty.",
    )
    parser.add_argument(
        "--initial-stream-chunk-size",
        type=int,
        default=8,
        help="Initial stream chunk size.",
    )
    parser.add_argument(
        "--stream-chunk-size", type=int, default=8, help="Stream chunk size."
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip already existing intros."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup precision
    precision = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.precision]

    # Initialize driver
    logger.info("Initializing FishSpeechDriver...")
    driver = FishSpeechDriver.from_model_paths(
        llama_checkpoint_path=args.llama_checkpoint_path,
        decoder_checkpoint_path=args.decoder_checkpoint_path,
        decoder_config_name=args.decoder_config_name,
        device=args.device,
        precision=precision,
        compile=args.compile,
    )

    # Load input
    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    if isinstance(input_data, dict) and "items" in input_data:
        items = input_data["items"]
    elif isinstance(input_data, list):
        items = input_data
    else:
        raise ValueError("Invalid input JSON format. Expected list or {'items': [...]}")

    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    generated_count = 0
    failed_count = 0
    skipped_count = 0

    for item in items:
        intro_id = item.get("id")
        text = item.get("text")

        if not intro_id or not text:
            logger.warning("Skipping item with missing id or text: {}", item)
            continue

        # Validate ID
        if not re.match(r"^[a-zA-Z0-9\-_]+$", intro_id):
            logger.error("Invalid intro id: {}. Use only alphanumeric, - and _", intro_id)
            failed_count += 1
            continue

        intro_dir = output_base_dir / intro_id
        if args.skip_existing and intro_dir.exists():
            logger.info("Skipping existing intro: {}", intro_id)
            skipped_count += 1
            continue

        logger.info("Processing intro: {} ('{}')", intro_id, text)

        try:
            request = DriverSynthesisRequest(
                text=text,
                segments=[text],
                reference_id=args.reference_id,
                seed=args.seed,
                use_memory_cache="on",
                normalize=True,
                stream_audio=False,
                generation=DriverGenerationOptions(
                    chunk_length=args.chunk_length,
                    max_new_tokens=args.max_new_tokens,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    temperature=args.temperature,
                    stream_tokens=True,
                    initial_stream_chunk_size=args.initial_stream_chunk_size,
                    stream_chunk_size=args.stream_chunk_size,
                ),
            )

            result = driver.synthesize_collect(request)
            audio = result["audio"]
            codes = result["codes"]
            sample_rate = result["sample_rate"]

            if codes is None or codes.shape[1] == 0:
                raise RuntimeError("No codes generated")

            # If audio is missing (e.g. stream_audio=False and not auto-decoded), decode it
            if audio is None:
                logger.info("Decoding audio from codes...")
                audio = driver.engine.get_audio_segment(
                    GenerateResponse(action="sample", codes=codes)
                )

            # Ensure intro directory exists
            intro_dir.mkdir(parents=True, exist_ok=True)

            # Save codes
            save_codes_pt(codes, intro_dir / "codes.pt", name=f"intro {intro_id}")

            # Save WAV
            sf.write(intro_dir / "audio.wav", audio, sample_rate)

            # Save PCM16LE
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
            (intro_dir / "audio.pcm").write_bytes(pcm16.tobytes())

            # Save meta.json
            meta = {
                "id": intro_id,
                "text": text,
                "reference_id": args.reference_id,
                "sample_rate": sample_rate,
                "channels": 1,
                "sample_width": 2,
                "format": "pcm16le",
                "code_frames": codes.shape[1],
                "num_codebooks": codes.shape[0],
                "audio_samples": len(audio),
                "duration_ms": int(len(audio) / sample_rate * 1000),
                "seed": args.seed,
                "max_new_tokens": args.max_new_tokens,
                "chunk_length": args.chunk_length,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "repetition_penalty": args.repetition_penalty,
                "created_at": datetime.now().isoformat(),
            }
            with open(intro_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info("Successfully generated intro: {}", intro_id)
            generated_count += 1

        except Exception as e:
            logger.exception("Failed to generate intro {}: {}", intro_id, e)
            failed_count += 1

    logger.info("Build finished:")
    logger.info("  Generated: {}", generated_count)
    logger.info("  Failed:    {}", failed_count)
    logger.info("  Skipped:   {}", skipped_count)
    logger.info("  Output:    {}", args.output_dir)


if __name__ == "__main__":
    main()

# Cached Intro Mode in Fish Speech

This document describes the utilities and workflow for building and using cached intro artifacts in Fish Speech.

## Overview

Cached intros allow for lower Time To First Audio (TTFA) by pre-generating the beginning of common phrases.
The server/proxy can then immediately serve the cached audio while the model generates the remainder (the suffix) in the background.

## Key Concepts

1.  **Cached Intro Artifact**: A set of files including pre-generated audio (WAV/PCM), acoustic codes (VQ tokens), and metadata.
2.  **Continuation History**: Cached codes are injected into the LLaMA model's history as a `continuation_tokens` prompt, ensuring acoustic and linguistic continuity between the cached prefix and the newly generated suffix.
3.  **Not a Reference**: Cached intros are different from speaker references. They represent specific text being spoken, whereas references represent the voice style.

## Artifact Structure

Each cached intro is stored in a directory:

- `audio.wav`: Reference audio for manual inspection.
- `audio.pcm`: Raw 16-bit little-endian PCM audio at the model's sample rate (e.g., 44.1kHz). This is served to the client.
- `codes.pt`: Torch tensor of shape `[num_codebooks, T]` containing the acoustic codes.
- `meta.json`: Metadata about the intro (text, sample rate, duration, generation parameters, etc.).

## Building Cached Intros

Use the `build_cached_intros.py` tool to generate artifacts from a JSON input.

### Input JSON Format

```json
[
  {
    "id": "welcome_message",
    "text": "Welcome to our service!"
  },
  {
    "id": "how_can_i_help",
    "text": "How can I help you today?"
  }
]
```

### Running the Builder

```bash
python -m fish_speech.tools.build_cached_intros \
  --input intros.json \
  --output-dir ./cached_intros \
  --reference-id my_voice_model \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --device cuda
```

## How it works (Server-side)

When a synthesis request is received:
1. The server checks if the requested text starts with any cached intro prefix.
2. If a match is found:
   - The server immediately sends `audio.pcm` to the client.
   - The server initiates a stateful synthesis session.
   - The cached codes from `codes.pt` are passed as `continuation_tokens`.
   - The intro text is passed as `continuation_text`.
   - The model generates only the remaining suffix text.

## Codec Utilities

New utilities are available in `fish_speech.codec.codes` for managing VQ codes:
- `normalize_codes`: Standardizes codes to `[num_codebooks, T]` CPU long tensor.
- `crop_codes_tail`: Safely trims history to fit within model context limits.
- `load_codes_pt` / `save_codes_pt`: Standardized loading and saving.
- `validate_codes_for_decoder`: Ensures codes match the decoder's expected codebook count.

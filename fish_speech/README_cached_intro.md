# Cached Intro Mode in Fish Speech Core

Cached intro mode lowers perceived TTFA by letting a server/proxy play a
pre-generated prefix immediately while Fish Speech continues the phrase.

Fish Speech core does not do prefix matching, browser streaming, PCM queueing,
or protocol events. Core is responsible for two things:

- building cached intro artifacts: WAV/PCM for the client and DAC/VQ codes for
  the model;
- accepting already-spoken acoustic history through `continuation_text` and
  `continuation_tokens`.

## Prompt vs Continuation

`prompt_text` and `prompt_tokens` are speaker reference inputs. They condition
voice identity and style. Cached intros must not be passed through
`prompt_tokens`.

`continuation_text` and `continuation_tokens` are already-spoken acoustic
history. Cached intros belong here. The prompt builder inserts each continuation
turn as:

```python
Message(role="user", parts=[TextPart(text=old_text)])
Message(role="assistant", parts=[VQPart(codes=old_codes)], modality="voice")
```

That makes the model continue naturally from the intro while the server sends
only the suffix text to TTS.

## Prefix Cache Artifacts

For new prefix-cache lookup flows, use `build_prefix_cache`. It writes a flat
manifest plus inspectable per-artifact directories:

- `pcm/<cache_id>.pcm`: raw PCM16LE without a WAV header;
- `wav/<cache_id>.wav`: inspectable audio file;
- `codes/<cache_id>.pt`: normalized CPU `torch.long` DAC/VQ codes with shape
  `[num_codebooks, T]`;
- `meta/<cache_id>.json`: text, generation settings, sample rate, duration, and
  code frame metadata.

Manifest entries are keyed for server/proxy prefix lookup:

```json
{
  "cache_id": "voice_ru_chto_takoe_v1",
  "voice_id": "voice",
  "text": "Что такое",
  "normalized_text": "что такое",
  "word_count": 2,
  "pcm_path": "pcm/voice_ru_chto_takoe_v1.pcm",
  "codes_path": "codes/voice_ru_chto_takoe_v1.pt",
  "code_frames": 120,
  "audio_meta": {
    "sample_rate": 44100,
    "channels": 1,
    "sample_width": 2
  },
  "params_hash": "..."
}
```

Prefix phrases are limited to five words. Longer items are rejected before any
model work starts.

## Legacy Cached Intro Artifacts

Each cached intro directory contains:

- `audio.wav`: inspectable audio file;
- `audio.pcm`: raw PCM served immediately to the browser;
- `codes.pt`: normalized CPU `torch.long` DAC/VQ codes with shape
  `[num_codebooks, T]`;
- `meta.json`: text, generation settings, sample rate, duration, and code frame
  metadata.

The browser needs PCM/WAV. The model needs the DAC/VQ codes.

## Building Prefix Cache Artifacts

Input may be either a list:

```json
[
  {
    "cache_id": "voice_ru_chto_takoe_v1",
    "voice_id": "voice",
    "text": "Что такое"
  }
]
```

or an object with `items`:

```json
{
  "items": [
    {
      "cache_id": "voice_ru_chto_takoe_v1",
      "voice_id": "voice",
      "text": "Что такое"
    }
  ]
}
```

Run:

```bash
python -m fish_speech.tools.build_prefix_cache \
  --input prefixes.json \
  --output-dir ./prefix_cache \
  --voice-id voice \
  --skip-existing
```

Checkpoint paths, device, precision, and compile mode default to
`config/runtime.json` unless explicitly passed.

## Building Legacy Cached Intro Artifacts

Input may be either a list:

```json
[
  { "id": "what_is", "text": "Что такое" }
]
```

or an object with `items`:

```json
{
  "items": [
    { "id": "what_is", "text": "Что такое" }
  ]
}
```

Run:

```bash
python -m fish_speech.tools.build_cached_intros \
  --input intros.json \
  --output-dir ./cached_intros \
  --reference-id voice \
  --skip-existing
```

Checkpoint paths, device, precision, and compile mode default to
`config/runtime.json` unless explicitly passed.

## Runtime Workflow

Full text:

```text
Что такое квантизация модели и зачем она нужна?
```

Cached intro:

```text
Что такое
```

The server/proxy has two safe runtime strategies.

### Continuation-only

This mode sends cached PCM to the client and sends only the suffix to Fish
Speech with cached codes in `continuation_text` / `continuation_tokens`:

1. matches the cached prefix;
2. immediately sends `prefix_cache/pcm/voice_ru_chto_takoe_v1.pcm` to the client;
3. loads `prefix_cache/codes/voice_ru_chto_takoe_v1.pt`;
4. sends only the suffix to Fish Speech:

```python
from fish_speech.codec import load_codes_pt
from fish_speech.driver import DriverSynthesisRequest

intro_codes = load_codes_pt("prefix_cache/codes/voice_ru_chto_takoe_v1.pt")

request = DriverSynthesisRequest(
    text="квантизация модели и зачем она нужна?",
    segments=["квантизация модели и зачем она нужна?"],
    reference_id="voice",
    continuation_text=["Что такое"],
    continuation_tokens=[intro_codes],
)
```

The cached prefix is not sent as normal TTS text and is not mixed into Fish
Speech audio generation. Fish Speech sees it only as continuation history.

### Full-commit proxy mode

The proxy can also generate the full phrase upstream while cached PCM is already
playing. It then skips the generated prefix in PCM space. This is the default
for high-quality prefix-cache playback because the model reads the whole phrase
with natural prosody instead of starting cold at the suffix.

To avoid duplicated final phonemes, the proxy does not use a blind fixed cut:

- it buffers a short upstream window around the expected prefix duration;
- compares the cached prefix tail with candidate upstream tails;
- chooses an adaptive skip point near the acoustic match;
- holds a tiny cached tail and crossfades it into the live head.

Useful NDJSON diagnostics:

- `prefix_cache_generation_skip_done.adaptive_skip_method`
- `adaptive_skip_delta_ms`
- `adaptive_skip_score`
- `prefix_cache_crossfade_ms`

`stream_tokens` should normally stay `false` for browser streaming. The engine
still streams tokens internally for audio and can collect final codes for
stateful history without yielding every token chunk to the HTTP/UI layer.

## Codec Utilities

`fish_speech.codec.codes` provides:

- `normalize_codes`: normalize `[C, T]` or `[1, C, T]` payloads to CPU
  `torch.long`;
- `load_codes_pt` / `save_codes_pt`: stable `.pt` load/save helpers;
- `validate_codes_for_decoder`: validate codebook count and code ranges for the
  active decoder;
- `crop_codes_tail`: return a compact tail copy for history policies.

External cached intro continuation defaults to the `full` context policy, so the
intro codes are not silently cropped by long-form generated-history settings.

# Cached Intro Mode in Fish Speech Core

Cached intro mode lowers perceived TTFA by letting a server/proxy play a
pre-generated prefix immediately while Fish Speech generates only the live
suffix.

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

## Artifacts

Each cached intro directory contains:

- `audio.wav`: inspectable audio file;
- `audio.pcm`: raw PCM served immediately to the browser;
- `codes.pt`: normalized CPU `torch.long` DAC/VQ codes with shape
  `[num_codebooks, T]`;
- `meta.json`: text, generation settings, sample rate, duration, and code frame
  metadata.

The browser needs PCM/WAV. The model needs the DAC/VQ codes.

## Building Artifacts

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

The server/proxy:

1. matches the cached prefix;
2. immediately sends `cached_intros/what_is/audio.pcm` to the client;
3. loads `cached_intros/what_is/codes.pt`;
4. sends only the suffix to Fish Speech:

```python
from fish_speech.codec import load_codes_pt
from fish_speech.driver import DriverSynthesisRequest

intro_codes = load_codes_pt("cached_intros/what_is/codes.pt")

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

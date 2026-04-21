# Realtime TTS Streaming & Session Mode

This document describes how to run and use the optimized realtime TTS streaming service, specifically tuned for low-latency applications like live interviews.

## Core Features

- **Minimal TTFA (Time To First Audible):** Aggressive first-chunk emission starts audio playback as soon as the first word is generated.
- **Inference Pipelining:** Overlaps LLM token generation with DAC audio decoding to eliminate gaps between segments.
- **Session-Aware Lifecycle:** Keeps the model "hot" during active sessions and performs light/heavy cleanup during idle periods.
- **Protocols:** Supports raw PCM and WAV streaming via WebSocket and HTTP.

## Running the Service

### 1. Start the Base TTS Backend

The base backend must be running first. It handles the actual model inference.

```bash
python -m tools.server.api \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --device cuda \
    --compile
```

Key flags:
- `--compile`: Enables Torch compilation for faster inference (recommended for RTX 5090).
- `--half`: Uses half-precision (FP16/BF16) to save VRAM.

### 2. Start the Session Mode Manager

The Session Mode Manager provides a high-level WebSocket API that handles text buffering, chunking, and session state.

```bash
python -m session_mode.app
```

The manager defaults to port `8765`.

## API Endpoints

### WebSocket API (`ws://localhost:8765/ws`)

This is the primary endpoint for realtime streaming.

#### Client -> Server Messages

- **`start_session`**: Initializes a new session.
  ```json
  { "type": "start_session", "config": { ... } }
  ```
- **`text_delta`**: Sends a chunk of text from the LLM.
  ```json
  { "type": "text_delta", "text": "Hello world", "final": false }
  ```
- **`flush`**: Forces the current buffer to be synthesized.
- **`cleanup`**: Triggers a heavy GPU memory cleanup.

#### Server -> Client Events

- **`session_started`**: Session is ready.
- **`chunk_queued`**: Text chunk has been cut and added to the TTS queue.
- **`audio_meta`**: Metadata for the upcoming audio (sample rate, format).
- **`audio_chunk`**: Binary audio data follows this JSON message (or is sent as a binary frame).
- **`tts_finished`**: All audio for a specific chunk has been sent.

### HTTP TTS Endpoint (`POST /v1/tts`)

Standard Fish Speech API, enhanced with realtime optimizations.

**Request Body:**
```json
{
    "text": "Your text here",
    "streaming": true,
    "format": "pcm",
    "stream_tokens": true,
    "stream_chunk_size": 8,
    "cleanup_mode": "session_idle"
}
```

- `format`: `wav` (default) or `pcm` (raw 16-bit LE).
- `cleanup_mode`:
    - `request_end`: Heavy cleanup after every request.
    - `session_idle`: Keep model warm, light cleanup only.
    - `none`: No automatic cleanup.

## Performance Tuning (RTX 5090 / 32GB)

For optimal performance on high-end hardware:

1. **Use `compile=True`**: Reduces kernel launch overhead.
2. **Streaming Tokens:** Set `stream_tokens=True` and `stream_chunk_size=8` (or lower) for the fastest start.
3. **Aggressive Buffer:** In `session_mode`, set `buffer.min_words` to `3` for natural speech, but rely on the `_is_first_chunk` optimization for immediate start.
4. **Environment Variables:**
   - `FISH_CACHE_MAX_SEQ_LEN`: Set to `512` or `1024` for short realtime turns.
   - `FISH_STREAM_EMPTY_CACHE`: Set to `0` during active sessions to avoid latency spikes from `cudaEmptyCache`.

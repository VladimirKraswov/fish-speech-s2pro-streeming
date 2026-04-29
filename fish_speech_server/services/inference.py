from http import HTTPStatus
from typing import Any

import numpy as np
from kui.asgi import HTTPException

from fish_speech import (
    DriverAudioChunkEvent,
    DriverErrorEvent,
    DriverFinalAudioEvent,
    DriverSynthesisRequest,
    DriverTokenChunkEvent,
    FishSpeechDriver,
)
from fish_speech_server.services.adapter import api_tts_to_driver_request
from fish_speech_server.services.audio import wav_chunk_header
from fish_speech_server.schema import ServeTTSRequest


def float_audio_to_pcm16_bytes(audio: np.ndarray | bytes | bytearray) -> bytes:
    """
    Convert float audio in [-1.0, 1.0] to little-endian PCM16 safely.

    Important: multiplying by 32768 and casting directly can overflow 1.0 to
    -32768. We clip and use 32767.0 instead.
    """
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)

    arr = np.asarray(audio, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


def inference_wrapper(
    req: ServeTTSRequest | DriverSynthesisRequest | Any,
    driver: FishSpeechDriver,
):
    """
    Wrapper for the inference function used in the API server.

    Streaming mode emits:
    - one WAV header;
    - raw PCM16 chunks;
    - token events internally when stateful mode asks for them.

    Non-streaming mode emits a single PCM16 payload.
    """
    audio_chunk_count = 0

    if isinstance(req, DriverSynthesisRequest):
        driver_req = req
        streaming = bool(getattr(driver_req, "stream_audio", False))
    else:
        driver_req = api_tts_to_driver_request(req)
        streaming = bool(getattr(req, "streaming", False))

    if streaming:
        yield wav_chunk_header(sample_rate=driver.sample_rate)

    for event in driver.synthesize(driver_req):
        match event:
            case DriverErrorEvent():
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content=str(event.error),
                )

            case DriverAudioChunkEvent():
                audio_chunk_count += 1
                yield float_audio_to_pcm16_bytes(event.audio)

            case DriverTokenChunkEvent():
                # Propagate codes for stateful session tracking.
                yield event

            case DriverFinalAudioEvent():
                # In streaming mode chunks have already been sent. Do not send
                # the final full track again, but keep consuming the generator so
                # the driver can finish normally instead of treating it as abort.
                if streaming:
                    continue

                audio_chunk_count += 1
                yield float_audio_to_pcm16_bytes(event.audio)
                return None

    if audio_chunk_count == 0:
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content="No audio generated, please check the input text.",
        )
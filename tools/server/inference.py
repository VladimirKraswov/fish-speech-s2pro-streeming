from http import HTTPStatus

import numpy as np
from kui.asgi import HTTPException
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest

# float аудио в диапазоне [-1.0, 1.0] -> int16 PCM
AMPLITUDE = 32768


def float_audio_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio)
    audio = np.clip(audio, -1.0, 1.0 - (1.0 / AMPLITUDE))
    return (audio * AMPLITUDE).astype(np.int16, copy=False).tobytes()


def inference_wrapper(req: ServeTTSRequest, engine: TTSInferenceEngine):
    """
    Wrapper for streaming inference in API server.

    Если стрим уже начался, HTTPException бросать нельзя:
    клиент получит incomplete chunked read.
    """
    count = 0
    header_sent = False

    for result in engine.inference(req):
        match result.code:
            case "header":
                if isinstance(result.audio, tuple):
                    header = result.audio[1]
                    if isinstance(header, np.ndarray):
                        header_sent = True
                        yield header.tobytes()
                    elif isinstance(header, (bytes, bytearray)):
                        header_sent = True
                        yield bytes(header)

            case "error":
                err = result.error or RuntimeError("Unknown TTS inference error")

                if req.streaming and (header_sent or count > 0):
                    logger.error(
                        "Streaming TTS aborted after response start: {}",
                        err,
                    )
                    return None

                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content=str(err),
                )

            case "segment":
                count += 1
                if isinstance(result.audio, tuple):
                    audio = result.audio[1]
                    if isinstance(audio, np.ndarray):
                        yield float_audio_to_pcm16_bytes(audio)
                    elif isinstance(audio, (bytes, bytearray)):
                        yield bytes(audio)

            case "final":
                # В streaming-режиме final не должен отправляться повторно.
                if req.streaming:
                    return None

                count += 1
                if isinstance(result.audio, tuple):
                    final = result.audio[1]
                    if isinstance(final, np.ndarray):
                        yield float_audio_to_pcm16_bytes(final)
                    elif isinstance(final, (bytes, bytearray)):
                        yield bytes(final)
                return None

    if count == 0:
        if req.streaming and header_sent:
            logger.warning("Streaming TTS finished without audio body after WAV header")
            return None

        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content="No audio generated, please check the input text.",
        )

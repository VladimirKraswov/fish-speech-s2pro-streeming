# tools/server/inference.py
from http import HTTPStatus

import numpy as np
from kui.asgi import HTTPException

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest

# float аудио в диапазоне [-1.0, 1.0] -> int16 PCM
AMPLITUDE = 32768


def inference_wrapper(req: ServeTTSRequest, engine: TTSInferenceEngine):
    """
    Wrapper for the inference function.
    Used in the API server.
    """
    count = 0

    for result in engine.inference(req):
        match result.code:
            case "header":
                if isinstance(result.audio, tuple):
                    header = result.audio[1]
                    if isinstance(header, np.ndarray):
                        yield header.tobytes()
                    elif isinstance(header, (bytes, bytearray)):
                        yield bytes(header)

            case "error":
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content=str(result.error),
                )

            case "segment":
                count += 1
                if isinstance(result.audio, tuple):
                    yield (result.audio[1] * AMPLITUDE).astype(np.int16).tobytes()

            case "final":
                # В streaming-режиме final уже содержит весь звук целиком,
                # а segment уже были отправлены ранее.
                # Если отдать final ещё раз, пользователь услышит дубль.
                if req.streaming:
                    return None

                count += 1
                if isinstance(result.audio, tuple):
                    final = result.audio[1]
                    if isinstance(final, np.ndarray):
                        yield (final * AMPLITUDE).astype(np.int16).tobytes()
                    elif isinstance(final, (bytes, bytearray)):
                        yield bytes(final)
                return None

    if count == 0:
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content="No audio generated, please check the input text.",
        )
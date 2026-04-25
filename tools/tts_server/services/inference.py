from http import HTTPStatus

import numpy as np
from kui.asgi import HTTPException

from fish_speech import (
    DriverAudioChunkEvent,
    DriverErrorEvent,
    DriverFinalAudioEvent,
    FishSpeechDriver,
)
from tools.tts_server.services.adapter import api_tts_to_driver_request
from tools.tts_server.services.audio import wav_chunk_header
from tools.tts_server.schema import ServeTTSRequest

# float аудио в диапазоне [-1.0, 1.0] -> int16 PCM
AMPLITUDE = 32768


def inference_wrapper(req: ServeTTSRequest, driver: FishSpeechDriver):
    """
    Wrapper for the inference function.
    Used in the API server.
    """
    count = 0
    driver_req = api_tts_to_driver_request(req)

    if req.streaming:
        yield wav_chunk_header(sample_rate=driver.sample_rate)

    for event in driver.synthesize(driver_req):
        match event:
            case DriverErrorEvent():
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content=str(event.error),
                )

            case DriverAudioChunkEvent():
                count += 1
                yield (event.audio * AMPLITUDE).astype(np.int16).tobytes()

            case DriverFinalAudioEvent():
                # В streaming-режиме final содержит полную собранную дорожку.
                # Сегменты уже были отправлены ранее, поэтому финал нельзя
                # отдавать повторно. Но и завершать генератор здесь тоже нельзя:
                # иначе driver inference не успевает выставить finished_normally,
                # и finally-ветка ошибочно запускает cleanup_on_abort.
                if req.streaming:
                    continue

                count += 1
                final = event.audio
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

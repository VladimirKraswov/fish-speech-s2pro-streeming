import numpy as np
import pytest
import torch

from fish_speech.driver.api import FishSpeechDriver
from fish_speech.driver.types import DriverGenerationOptions, DriverSynthesisRequest
from fish_speech.inference_engine.utils import InferenceResult


class FakeDecoder:
    sample_rate = 44100


class FakeEngine:
    def __init__(self, events):
        self.decoder_model = FakeDecoder()
        self.events = list(events)

    def inference(self, req):
        yield from self.events

    def list_reference_ids(self):
        return []


def test_synthesize_with_codes_collects_audio_and_token_chunks():
    events = [
        InferenceResult(
            code="tokens",
            audio=None,
            error=None,
            tokens=torch.ones((9, 3), dtype=torch.int32),
        ),
        InferenceResult(
            code="tokens",
            audio=None,
            error=None,
            tokens=torch.ones((9, 2), dtype=torch.int32) * 2,
        ),
        InferenceResult(
            code="final",
            audio=(44100, np.ones(8, dtype=np.float32)),
            error=None,
        ),
    ]
    driver = FishSpeechDriver(FakeEngine(events))

    result = driver.synthesize_with_codes(
        DriverSynthesisRequest(
            text="cached intro",
            generation=DriverGenerationOptions(stream_tokens=True),
        )
    )

    assert result["text"] == "cached intro"
    assert result["sample_rate"] == 44100
    assert result["audio"].shape == (8,)
    assert result["codes"].shape == (9, 5)
    assert result["codes"].dtype == torch.long
    assert len(result["code_chunks"]) == 2


def test_synthesize_collect_is_backward_compatible_alias():
    events = [
        InferenceResult(
            code="tokens",
            audio=None,
            error=None,
            tokens=torch.zeros((9, 1), dtype=torch.long),
        )
    ]
    driver = FishSpeechDriver(FakeEngine(events))

    result = driver.synthesize_collect(DriverSynthesisRequest(text="hello"))

    assert result["codes"].shape == (9, 1)


def test_synthesize_with_codes_raises_on_driver_error_event():
    err = ValueError("boom")
    driver = FishSpeechDriver(
        FakeEngine([InferenceResult(code="error", audio=None, error=err)])
    )

    with pytest.raises(RuntimeError, match="Synthesis error: boom"):
        driver.synthesize_with_codes(DriverSynthesisRequest(text="hello"))

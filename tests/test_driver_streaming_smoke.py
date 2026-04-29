import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from fish_speech.driver.api import FishSpeechDriver
from fish_speech.driver.types import (
    DriverSynthesisRequest,
    DriverGenerationOptions,
    DriverAudioChunkEvent,
    DriverTokenChunkEvent,
    DriverFinalAudioEvent,
    DriverErrorEvent
)
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.driver.warmup import run_driver_warmup

class FakeEngine:
    def __init__(self):
        self.decoder_model = MagicMock()
        self.decoder_model.sample_rate = 44100
        self.ref_by_id = {}
        self.references_dir = MagicMock()

    def list_reference_ids(self):
        return []

    def inference(self, req: DriverSynthesisRequest):
        # Mock inference behavior based on request
        if req.stream_tokens:
            yield InferenceResult(code="tokens", audio=None, error=None, tokens=torch.zeros((9, 10)))

        if req.stream_audio:
            yield InferenceResult(code="segment", audio=(44100, np.zeros(1024)), error=None)

        if not req.stream_audio:
            yield InferenceResult(code="final", audio=(44100, np.zeros(2048)), error=None)

@pytest.fixture
def fake_driver():
    engine = FakeEngine()
    return FishSpeechDriver(engine=engine)

def test_non_stream_audio(fake_driver):
    request = DriverSynthesisRequest(
        text="hello",
        stream_audio=False,
        generation=DriverGenerationOptions(stream_tokens=False)
    )
    events = list(fake_driver.synthesize(request))

    assert any(isinstance(e, DriverFinalAudioEvent) for e in events)
    assert not any(isinstance(e, DriverAudioChunkEvent) for e in events)
    assert not any(isinstance(e, DriverTokenChunkEvent) for e in events)

def test_token_stream_only(fake_driver):
    request = DriverSynthesisRequest(
        text="hello",
        stream_audio=False,
        generation=DriverGenerationOptions(stream_tokens=True)
    )
    events = list(fake_driver.synthesize(request))

    assert any(isinstance(e, DriverTokenChunkEvent) for e in events)
    assert any(isinstance(e, DriverFinalAudioEvent) for e in events)
    assert not any(isinstance(e, DriverAudioChunkEvent) for e in events)

def test_audio_stream_only(fake_driver):
    request = DriverSynthesisRequest(
        text="hello",
        stream_audio=True,
        generation=DriverGenerationOptions(stream_tokens=False)
    )
    events = list(fake_driver.synthesize(request))

    assert any(isinstance(e, DriverAudioChunkEvent) for e in events)
    assert not any(isinstance(e, DriverTokenChunkEvent) for e in events)

def test_audio_token_stream(fake_driver):
    request = DriverSynthesisRequest(
        text="hello",
        stream_audio=True,
        generation=DriverGenerationOptions(stream_tokens=True)
    )
    events = list(fake_driver.synthesize(request))

    assert any(isinstance(e, DriverTokenChunkEvent) for e in events)
    assert any(isinstance(e, DriverAudioChunkEvent) for e in events)

def test_warmup_error_handling():
    fake_driver = MagicMock()
    fake_driver.config.warmup.enabled = True
    fake_driver.config.warmup.text = "warmup"

    # Mock synthesize to return an error event
    def mock_synthesize(req):
        yield DriverErrorEvent(error=ValueError("boom"))

    fake_driver.synthesize = mock_synthesize

    with pytest.raises(RuntimeError, match="Warmup synthesis failed: boom"):
        run_driver_warmup(fake_driver)

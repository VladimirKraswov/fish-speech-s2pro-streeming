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

def test_driver_stats_counts_submitted_segments_and_audio_events(fake_driver):
    list(
        fake_driver.synthesize(
            DriverSynthesisRequest(
                text="hello",
                stream_audio=False,
                generation=DriverGenerationOptions(stream_tokens=False),
            )
        )
    )
    list(
        fake_driver.synthesize(
            DriverSynthesisRequest(
                text="stream me",
                stream_audio=True,
                generation=DriverGenerationOptions(stream_tokens=False),
            )
        )
    )

    stats = fake_driver.stats()
    assert stats.sessions_opened == 2
    assert stats.generated_segments == 2
    assert stats.audio_events == 2


def test_rename_reference_validates_paths_and_clears_cache(tmp_path):
    engine = FakeEngine()
    engine.references_dir = tmp_path
    engine.ref_by_id = {"old": "cached-old", "new": "cached-new"}
    (tmp_path / "old").mkdir()

    driver = FishSpeechDriver(engine=engine)
    driver.rename_reference("old", "new")

    assert not (tmp_path / "old").exists()
    assert (tmp_path / "new").is_dir()
    assert "old" not in engine.ref_by_id
    assert "new" not in engine.ref_by_id

    with pytest.raises(ValueError, match="invalid characters"):
        driver.rename_reference("new", "bad/name")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        driver.rename_reference("missing", "another")

    (tmp_path / "existing").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        driver.rename_reference("new", "existing")

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

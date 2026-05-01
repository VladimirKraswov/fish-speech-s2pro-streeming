from types import SimpleNamespace

import numpy as np
import torch

from fish_speech.driver.types import DriverGenerationOptions, DriverSynthesisRequest
from fish_speech.generation.prompt_builder import GenerateResponse
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.inference_engine.utils import InferenceResult


class _Wrapped:
    def __init__(self, response):
        self.status = "success"
        self.response = response


class _FakeDecoder:
    sample_rate = 44100
    frame_length = 1
    quantizer = None


class _ProbeEngine(TTSInferenceEngine):
    def __init__(self, segment):
        self.runtime = SimpleNamespace(
            model=SimpleNamespace(profile_inference=False),
        )
        self.decoder_model = _FakeDecoder()
        self.llama_device = torch.device("cpu")
        self.compile = False
        self.cleanup_on_abort = False
        self.empty_cache_per_segment = False
        self.seen_left_context = []
        self.sent_continuation_tokens = None
        self.segment = np.asarray(segment, dtype=np.float32)

    def load_by_id(self, ref_id, use_cache):
        return [], []

    def send_Llama_request(self, req, **kwargs):
        import queue

        self.sent_continuation_tokens = kwargs["continuation_tokens"]
        q = queue.Queue()
        q.put(
            _Wrapped(
                GenerateResponse(
                    action="sample",
                    codes=torch.ones((2, 4), dtype=torch.long),
                )
            )
        )
        q.put(_Wrapped(GenerateResponse(action="next")))
        return q

    def _decode_stream_codes_with_context(
        self,
        *,
        new_codes,
        left_context_codes,
        max_context_frames,
    ):
        self.seen_left_context.append(
            None if left_context_codes is None else left_context_codes.clone()
        )
        return self.segment.copy(), new_codes.detach().cpu(), int(self.segment.size)

    def _maybe_cleanup_after_success(self):
        return None

    def _cuda_cleanup(self, *, reason):
        return None


def _collect_audio_events(engine, request):
    return [
        event
        for event in engine.inference(request)
        if isinstance(event, InferenceResult) and event.code == "segment"
    ]


def _collect_events(engine, request):
    return [
        event
        for event in engine.inference(request)
        if isinstance(event, InferenceResult)
    ]


def test_stream_audio_initializes_left_context_from_last_continuation_turn():
    segment = np.arange(6, dtype=np.float32)
    engine = _ProbeEngine(segment)
    first_history = torch.zeros((2, 12), dtype=torch.long)
    last_history = torch.arange(80, dtype=torch.long).reshape(2, 40)

    events = _collect_audio_events(
        engine,
        DriverSynthesisRequest(
            text="suffix",
            segments=["suffix"],
            reference_id="voice",
            continuation_text=["old", "cached prefix"],
            continuation_tokens=[first_history, last_history],
            stream_audio=True,
            generation=DriverGenerationOptions(stream_tokens=False),
        ),
    )

    assert len(engine.sent_continuation_tokens) == 2
    assert torch.equal(engine.sent_continuation_tokens[0], first_history)
    assert torch.equal(engine.sent_continuation_tokens[1], last_history)
    assert len(engine.seen_left_context) == 1
    assert torch.equal(engine.seen_left_context[0], last_history[:, -32:])
    assert len(events) == 1
    assert np.array_equal(events[0].audio[1], segment)


def test_stream_audio_without_continuation_keeps_no_left_context_and_raw_segment():
    segment = np.arange(5, dtype=np.float32)
    engine = _ProbeEngine(segment)

    events = _collect_audio_events(
        engine,
        DriverSynthesisRequest(
            text="plain",
            segments=["plain"],
            reference_id="voice",
            stream_audio=True,
            generation=DriverGenerationOptions(stream_tokens=False),
        ),
    )

    assert engine.sent_continuation_tokens == []
    assert engine.seen_left_context == [None]
    assert len(events) == 1
    assert np.array_equal(events[0].audio[1], segment)


def test_stream_audio_can_collect_history_codes_without_public_token_stream():
    segment = np.arange(5, dtype=np.float32)
    engine = _ProbeEngine(segment)

    events = _collect_events(
        engine,
        DriverSynthesisRequest(
            text="plain",
            segments=["plain"],
            reference_id="voice",
            stream_audio=True,
            generation=DriverGenerationOptions(
                stream_tokens=False,
                collect_tokens=True,
            ),
        ),
    )

    assert [event.code for event in events] == ["segment", "tokens"]
    assert events[1].tokens.shape == (2, 4)

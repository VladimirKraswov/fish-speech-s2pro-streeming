from fish_speech_server.services.adapter import stateful_tts_to_driver_request
from fish_speech_server.schema import StatefulTTSRequest
from fish_speech_server.services.synthesis_context import SynthesisContext
import time

def test_adapter_stateful():
    ctx = SynthesisContext(
        synthesis_session_id="test_session",
        reference_id="voice",
        created_at=time.time(),
        updated_at=time.time()
    )

    req = StatefulTTSRequest(
        synthesis_session_id="test_session",
        commit_seq=1,
        text="Hello world",
        streaming=True,
        stream_tokens=True
    )

    driver_req = stateful_tts_to_driver_request(req, ctx)

    print(f"Driver Request Reference ID: {driver_req.reference_id}")
    assert driver_req.reference_id == "voice"

    print(f"Driver Request Segments: {driver_req.segments}")
    assert driver_req.segments == ["Hello world"]

    print(f"Driver Request Stream Tokens: {driver_req.generation.stream_tokens}")
    assert driver_req.generation.stream_tokens == True

    # Test override
    req2 = StatefulTTSRequest(
        synthesis_session_id="test_session",
        commit_seq=2,
        text="Override ref",
        reference_id="override_voice",
        streaming=True,
        stream_tokens=True
    )
    driver_req2 = stateful_tts_to_driver_request(req2, ctx)
    print(f"Driver Request 2 Reference ID: {driver_req2.reference_id}")
    assert driver_req2.reference_id == "override_voice"

if __name__ == "__main__":
    test_adapter_stateful()
    print("Adapter stateful test passed!")

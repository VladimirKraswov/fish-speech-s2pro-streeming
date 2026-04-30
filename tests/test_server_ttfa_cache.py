import torch
import pytest
from fish_speech_server.services.stateful_inference import _concat_code_chunks
from fish_speech_server.proxy.pcm import _intro_cache_key, _normalized_reference_id
from fish_speech_server.config import load_runtime_config


def test_api_views_import():
    import fish_speech_server.api.views as views
    assert views is not None


# Unit tests for _concat_code_chunks
def test_concat_code_chunks_2d():
    a = torch.zeros((8, 10), dtype=torch.long)
    b = torch.ones((8, 15), dtype=torch.long)

    out = _concat_code_chunks([a, b])

    assert out.shape == (8, 25)
    assert torch.equal(out[:, :10], a)
    assert torch.equal(out[:, 10:], b)


def test_concat_code_chunks_3d_batch_one():
    a = torch.zeros((1, 8, 10), dtype=torch.long)
    b = torch.ones((1, 8, 15), dtype=torch.long)

    out = _concat_code_chunks([a, b])

    assert out.shape == (8, 25)


def test_concat_code_chunks_rejects_mismatched_codebooks():
    a = torch.zeros((8, 10), dtype=torch.long)
    b = torch.ones((7, 15), dtype=torch.long)

    with pytest.raises(ValueError, match="Mismatched codebook count"):
        _concat_code_chunks([a, b])


# Unit tests for intro cache key
def test_intro_cache_key_stable_for_same_config():
    config = load_runtime_config().proxy
    assert _intro_cache_key(config) == _intro_cache_key(config)


def test_intro_cache_key_changes_when_intro_text_changes():
    config = load_runtime_config().proxy

    a = config.model_copy(deep=True)
    b = config.model_copy(deep=True)

    a.intro_cache.text = "Hello."
    b.intro_cache.text = "Different hello."

    assert _intro_cache_key(a) != _intro_cache_key(b)


def test_intro_cache_key_changes_when_reference_changes():
    config = load_runtime_config().proxy

    a = config.model_copy(deep=True)
    b = config.model_copy(deep=True)

    a.tts.reference_id = "voice_a"
    b.tts.reference_id = "voice_b"

    assert _intro_cache_key(a) != _intro_cache_key(b)


def test_intro_cache_key_changes_when_proxy_version_changes():
    config = load_runtime_config().proxy

    a = config.model_copy(deep=True)
    b = config.model_copy(deep=True)

    a.version = 1
    b.version = 2

    assert _intro_cache_key(a) != _intro_cache_key(b)


def test_normalized_reference_id():
    config = load_runtime_config().proxy

    config.tts.reference_id = "  voice_a  "
    assert _normalized_reference_id(config) == "voice_a"

    config.tts.reference_id = ""
    config.default_reference_id = " default_voice "
    assert _normalized_reference_id(config) == "default_voice"

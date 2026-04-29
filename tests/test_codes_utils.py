import torch
import numpy as np
import pytest
from pathlib import Path
from fish_speech.codec.codes import (
    estimate_code_frames,
    normalize_codes,
    crop_codes_tail,
    save_codes_pt,
    load_codes_pt,
    expected_codebooks_from_decoder
)

def test_estimate_code_frames():
    assert estimate_code_frames(None) == 0
    assert estimate_code_frames(torch.zeros(9, 10)) == 10
    assert estimate_code_frames(np.zeros((9, 10))) == 10
    assert estimate_code_frames([[1, 2, 3], [4, 5, 6]]) == 3
    assert estimate_code_frames([1, 2, 3]) == 3
    assert estimate_code_frames([]) == 0

def test_normalize_codes():
    # [C, T]
    t = torch.zeros(9, 10)
    norm = normalize_codes(t)
    assert norm.shape == (9, 10)
    assert norm.dtype == torch.long
    assert not norm.is_cuda

    # [1, C, T]
    t = torch.zeros(1, 9, 10)
    norm = normalize_codes(t)
    assert norm.shape == (9, 10)

    # numpy
    n = np.zeros((9, 10))
    norm = normalize_codes(n)
    assert norm.shape == (9, 10)

    # list
    l = [[0]*10]*9
    norm = normalize_codes(l)
    assert norm.shape == (9, 10)

    # Errors
    with pytest.raises(ValueError, match="cannot be None"):
        normalize_codes(None)

    with pytest.raises(ValueError, match="Unexpected 3D tensor shape"):
        normalize_codes(torch.zeros(2, 9, 10))

    with pytest.raises(ValueError, match="Unexpected tensor ndim"):
        normalize_codes(torch.zeros(10))

    with pytest.raises(ValueError, match="is empty"):
        normalize_codes(torch.zeros(9, 0))

    with pytest.raises(ValueError, match="codebook count mismatch"):
        normalize_codes(torch.zeros(9, 10), expected_codebooks=10)

def test_crop_codes_tail():
    t = torch.zeros(9, 100)

    # Normal crop
    cropped = crop_codes_tail(t, 20)
    assert cropped.shape == (9, 20)

    # Shorter than max_frames
    cropped = crop_codes_tail(t, 200)
    assert cropped.shape == (9, 100)

    # max_frames = 0
    assert crop_codes_tail(t, 0) is None

    # None
    assert crop_codes_tail(None, 20) is None

def test_save_load_roundtrip(tmp_path):
    t = torch.randint(0, 100, (9, 50))
    p = tmp_path / "test_codes.pt"

    save_codes_pt(t, p)
    loaded = load_codes_pt(p)

    assert torch.equal(t, loaded)
    assert loaded.dtype == torch.long

def test_load_from_bytes():
    t = torch.randint(0, 100, (9, 50))
    import io
    buf = io.BytesIO()
    torch.save(t, buf)
    data = buf.getvalue()

    loaded = load_codes_pt(data)
    assert torch.equal(t, loaded)

class MockQuantizer:
    def __init__(self, n_codebooks):
        self.n_codebooks = n_codebooks

class MockDAC:
    def __init__(self, n_codebooks):
        self.quantizer = MockQuantizer(n_codebooks)

def test_expected_codebooks_from_decoder():
    # Test fallback n_codebooks
    decoder = MockDAC(8)
    assert expected_codebooks_from_decoder(decoder) == 8

    assert expected_codebooks_from_decoder(None) is None

    class Empty: pass
    assert expected_codebooks_from_decoder(Empty()) is None

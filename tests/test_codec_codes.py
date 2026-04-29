import io
from pathlib import Path

import numpy as np
import pytest
import torch

from fish_speech.codec.codes import (
    crop_codes_tail,
    estimate_code_frames,
    expected_codebooks_from_decoder,
    load_codes_pt,
    normalize_codes,
    save_codes_pt,
    validate_codes_for_decoder,
)


def test_estimate_code_frames():
    assert estimate_code_frames(None) == 0
    assert estimate_code_frames(torch.zeros(9, 10)) == 10
    assert estimate_code_frames(np.zeros((9, 10))) == 10
    assert estimate_code_frames([[1, 2, 3], [4, 5, 6]]) == 3
    assert estimate_code_frames([1, 2, 3]) == 3
    assert estimate_code_frames([]) == 0


def test_normalize_codes():
    # [C, T] -> [C, T], long, cpu
    t = torch.zeros(9, 10)
    norm = normalize_codes(t)
    assert norm.shape == (9, 10)
    assert norm.dtype == torch.long
    assert not norm.is_cuda

    # [1, C, T] -> [C, T]
    t = torch.zeros(1, 9, 10)
    norm = normalize_codes(t)
    assert norm.shape == (9, 10)

    # numpy [9, 10] -> [9, 10]
    n = np.zeros((9, 10))
    norm = normalize_codes(n)
    assert norm.shape == (9, 10)

    # nested list
    l = [[0] * 10 for _ in range(9)]
    norm = normalize_codes(l)
    assert norm.shape == (9, 10)

    # Errors
    with pytest.raises(ValueError, match="cannot be None"):
        normalize_codes(None)

    # wrong shape [2, 9, 10] -> ValueError
    with pytest.raises(ValueError, match="unexpected 3D shape"):
        normalize_codes(torch.zeros(2, 9, 10))

    with pytest.raises(ValueError, match="unexpected ndim"):
        normalize_codes(torch.zeros(10))

    # empty T [9, 0] -> ValueError
    with pytest.raises(ValueError, match="zero frames"):
        normalize_codes(torch.zeros(9, 0))

    # expected_codebooks mismatch -> ValueError
    with pytest.raises(ValueError, match="codebook count mismatch"):
        normalize_codes(torch.zeros(9, 10), expected_codebooks=10)


def test_crop_codes_tail():
    # [9, 100], max_frames=20 -> [9, 20]
    t = torch.zeros(9, 100)
    cropped = crop_codes_tail(t, 20)
    assert cropped.shape == (9, 20)
    assert cropped.data_ptr() != t.data_ptr()

    # Shorter than max_frames
    cropped = crop_codes_tail(t, 200)
    assert cropped.shape == (9, 100)

    # max_frames=0 -> None
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
    buf = io.BytesIO()
    torch.save(t, buf)
    data = buf.getvalue()

    loaded = load_codes_pt(data)
    assert torch.equal(t, loaded)


def test_load_dict_and_tuple_payloads():
    t = torch.randint(0, 100, (9, 50))

    for payload in ({"codes": t}, {"tokens": t}, {"indices": t}, (t, "meta")):
        buf = io.BytesIO()
        torch.save(payload, buf)
        loaded = load_codes_pt(buf)
        assert torch.equal(t, loaded)


def test_reject_empty_and_batch_gt_one():
    with pytest.raises(ValueError, match="is empty"):
        normalize_codes([])
    with pytest.raises(ValueError, match="unexpected 3D shape"):
        normalize_codes(torch.zeros(2, 9, 10))


class MockQuantizer:
    def __init__(self, n_codebooks):
        self.n_codebooks = n_codebooks


class MockDAC:
    def __init__(self, n_codebooks):
        self.quantizer = MockQuantizer(n_codebooks)


class DownsampleResidualVectorQuantize:
    def __init__(self, n_codebooks):
        self.quantizer = MockQuantizer(n_codebooks)


def test_expected_codebooks_from_decoder():
    # Test fallback n_codebooks
    decoder = MockDAC(8)
    assert expected_codebooks_from_decoder(decoder) == 8

    # Test structural check
    class MockDACStructural:
        def __init__(self, n_codebooks):
            self.quantizer = DownsampleResidualVectorQuantize(n_codebooks)

    decoder = MockDACStructural(8)
    # 8 + 1 = 9
    assert expected_codebooks_from_decoder(decoder) == 9

    assert expected_codebooks_from_decoder(None) is None

    class Empty:
        pass

    assert expected_codebooks_from_decoder(Empty()) is None


class MockCodebook:
    def __init__(self, codebook_size, n_codebooks=None):
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks


class MockDownsampleDecoder:
    def __init__(self):
        self.quantizer = type(
            "DownsampleResidualVectorQuantize",
            (),
            {
                "semantic_quantizer": MockCodebook(codebook_size=4096),
                "quantizer": MockCodebook(codebook_size=1024, n_codebooks=8),
            },
        )()


def test_validate_codes_for_decoder_ranges():
    decoder = MockDownsampleDecoder()
    codes = torch.zeros((9, 10), dtype=torch.long)
    codes[0, 0] = 4095
    codes[1, 0] = 1023
    out = validate_codes_for_decoder(codes, decoder, name="intro")
    assert out.shape == (9, 10)

    bad_semantic = codes.clone()
    bad_semantic[0, 0] = 4096
    with pytest.raises(ValueError, match="intro semantic codes out of range"):
        validate_codes_for_decoder(bad_semantic, decoder, name="intro")

    bad_acoustic = codes.clone()
    bad_acoustic[1, 0] = -1
    with pytest.raises(ValueError, match="intro acoustic codes out of range"):
        validate_codes_for_decoder(bad_acoustic, decoder, name="intro")

    with pytest.raises(ValueError, match="codebook count mismatch"):
        validate_codes_for_decoder(torch.zeros((8, 10)), decoder, name="intro")

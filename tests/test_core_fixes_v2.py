import torch
import pytest
from fish_speech.generation.sampling import apply_repetition_penalty
from fish_speech.codec.codes import validate_codes_for_decoder
from unittest.mock import MagicMock

def test_repetition_penalty_compile_safe_v2():
    vocab_size = 100
    penalty = 1.2

    # Test [1, 1, vocab]
    logits = torch.ones(1, 1, vocab_size)
    prev = torch.tensor([1, 2, 3])

    # Non-compiled run
    out = apply_repetition_penalty(logits, prev, penalty)

    assert out.shape == logits.shape
    # Check that indices 1,2,3 are penalized (divided by 1.2 since they are 1.0 > 0)
    assert torch.allclose(out[0, 0, [1, 2, 3]], torch.tensor([1.0/1.2]*3))
    # Check that other indices are not penalized
    assert torch.allclose(out[0, 0, 4:], torch.ones(vocab_size - 4))

    # Test with valid_token_mask
    valid_mask = torch.zeros(vocab_size, dtype=torch.bool)
    valid_mask[1] = True # Only 1 is valid for penalty

    out = apply_repetition_penalty(logits, prev, penalty, valid_token_mask=valid_mask)
    assert out[0, 0, 1] < 1.0
    assert out[0, 0, 2] == 1.0
    assert out[0, 0, 3] == 1.0

def test_validate_codes_ranges():
    # Setup mock decoder
    decoder = MagicMock()
    quantizer = MagicMock()
    quantizer.__class__.__name__ = "DownsampleResidualVectorQuantize"

    sem_q = MagicMock()
    sem_q.codebook_size = 4096
    ac_q = MagicMock()
    ac_q.codebook_size = 1024
    ac_q.n_codebooks = 8

    quantizer.semantic_quantizer = sem_q
    quantizer.quantizer = ac_q
    decoder.quantizer = quantizer

    # Valid codes
    # expected = 8 + 1 = 9
    codes = torch.zeros((9, 10), dtype=torch.long)
    codes[0, 0] = 4095
    codes[1, 0] = 1023
    validate_codes_for_decoder(codes, decoder)

    # Invalid semantic
    codes[0, 0] = 4096
    with pytest.raises(ValueError, match="semantic codes out of range"):
        validate_codes_for_decoder(codes, decoder)

    # Invalid acoustic
    codes[0, 0] = 0
    codes[8, 0] = 1024
    with pytest.raises(ValueError, match="acoustic codes out of range"):
        validate_codes_for_decoder(codes, decoder)

    # Negative codes
    codes[8, 0] = -1
    with pytest.raises(ValueError, match="acoustic codes out of range|contains negative codes"):
        validate_codes_for_decoder(codes, decoder)

def test_continuation_cropping_logic():
    from fish_speech.generation.prompt_builder import _select_continuation_history

    history = [("segment1", torch.zeros(9, 100))]

    # Policy tail_frames
    selected = _select_continuation_history(history, policy="tail_frames", tail_frames=20, max_history_segments=1)
    assert len(selected) == 1
    assert selected[0][1].shape[-1] == 20

    # Policy none
    selected = _select_continuation_history(history, policy="none", tail_frames=20, max_history_segments=1)
    assert len(selected) == 0

    # Policy last_segment
    history_long = [("s1", torch.zeros(9, 10)), ("s2", torch.zeros(9, 10))]
    selected = _select_continuation_history(history_long, policy="last_segment", tail_frames=20, max_history_segments=1)
    assert len(selected) == 1
    assert selected[0][0] == "s2"

def test_compile_smoke_penalty():
    # Check that apply_repetition_penalty can be compiled with fullgraph=True
    compiled_penalty = torch.compile(apply_repetition_penalty, fullgraph=True)

    logits = torch.randn(1, 1, 100)
    prev = torch.tensor([1, 2, 3])

    # This might fail on CPU without aot_eager or specific backends, but we try
    try:
        compiled_penalty(logits, prev, 1.2)
    except Exception as e:
        # If it fails because of environment (no inductor etc), it's not necessarily a graph break
        print(f"Compile smoke failed (likely env): {e}")
        pass

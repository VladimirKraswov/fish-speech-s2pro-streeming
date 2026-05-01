import struct

from fish_speech_server.proxy.pcm import (
    _adaptive_prefix_skip_plan,
    _pcm16_crossfade,
)


def _pcm(samples: list[int]) -> bytes:
    out = bytearray()
    for sample in samples:
        out.extend(struct.pack("<h", sample))
    return bytes(out)


def _samples(pcm: bytes) -> list[int]:
    return [
        struct.unpack_from("<h", pcm, offset)[0]
        for offset in range(0, len(pcm), 2)
    ]


def test_adaptive_prefix_skip_finds_shifted_boundary():
    sample_rate = 1000
    cached = [0] * 20 + [i * 200 for i in range(-20, 20)] + [4000 - i * 120 for i in range(40)]
    live = [3000] * 30

    cached_pcm = _pcm(cached)
    upstream_pcm = _pcm([0] * 20 + cached + live)

    plan = _adaptive_prefix_skip_plan(
        cached_pcm=cached_pcm,
        upstream_pcm=upstream_pcm,
        planned_skip_bytes=len(cached_pcm),
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        search_ms=60,
        match_ms=20,
        min_head_ms=10,
    )

    assert plan["method"] == "adaptive_waveform"
    assert abs(plan["skip_bytes"] - (len(cached_pcm) + 40)) <= 4
    assert plan["skip_delta_bytes"] > 0
    assert plan["candidate_count"] > 1


def test_adaptive_prefix_skip_falls_back_when_window_is_too_short():
    plan = _adaptive_prefix_skip_plan(
        cached_pcm=_pcm([1, 2, 3, 4]),
        upstream_pcm=_pcm([1, 2, 3]),
        planned_skip_bytes=8,
        sample_rate=1000,
        channels=1,
        sample_width=2,
        search_ms=60,
        match_ms=20,
        min_head_ms=10,
    )

    assert plan["method"] in {
        "insufficient_lookahead",
        "insufficient_match_window",
    }


def test_pcm16_crossfade_overlaps_prefix_tail_and_live_head():
    prefix_tail = _pcm([10000] * 10)
    live_head = _pcm([0] * 10 + [2000] * 5)

    mixed = _samples(_pcm16_crossfade(prefix_tail, live_head, channels=1))

    assert len(mixed) == 15
    assert mixed[0] == 10000
    assert abs(mixed[9]) <= 1
    assert mixed[10:] == [2000] * 5

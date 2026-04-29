import struct


def wav_chunk_header(
    sample_rate: int = 44100,
    bit_depth: int = 16,
    channels: int = 1,
    data_size: int = 0xFFFFFFFF,
) -> bytes:
    """
    Streaming-friendly PCM WAV header.

    We intentionally use a very large data chunk size because streaming responses
    do not know the final PCM length at header time. The proxy only needs the
    fmt/data offset, and most streaming clients tolerate this better than a WAV
    header with data_size=0.
    """
    if bit_depth != 16:
        raise ValueError("Only 16-bit PCM WAV streaming header is supported")
    if channels <= 0:
        raise ValueError("channels must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    block_align = channels * (bit_depth // 8)
    byte_rate = sample_rate * block_align

    data_size = max(0, min(int(data_size), 0xFFFFFFFF))
    riff_size = min(36 + data_size, 0xFFFFFFFF)

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bit_depth,
        b"data",
        data_size,
    )
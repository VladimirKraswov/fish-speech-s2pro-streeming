import io
import wave


def wav_chunk_header(
    sample_rate: int = 44100, bit_depth: int = 16, channels: int = 1
) -> bytes:
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()

    return wav_header_bytes

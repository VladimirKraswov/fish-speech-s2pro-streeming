import hashlib
import io
from contextlib import nullcontext
from typing import Any

import librosa
import torch
from cachetools import LRUCache, cached

CACHE_MAXSIZE = 10000
MICRO_BATCH_SIZE = 8
ASR_SAMPLE_RATE = 16000
HUGE_GAP_THRESHOLD = 4000


def _model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _autocast_for_device(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _sample_rate(model) -> int:
    if hasattr(model, "spec_transform"):
        return int(model.spec_transform.sample_rate)
    return int(model.sample_rate)


def _stable_model_id(model: Any) -> str:
    """
    Best-effort cache namespace for a decoder model.

    Avoid using only the device in the VQGAN encode cache key: two decoder
    instances/checkpoints on the same device must not share cached encodings.
    """
    for attr in (
        "checkpoint_path",
        "model_path",
        "config_name",
        "name_or_path",
        "_get_name",
    ):
        value = getattr(model, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value:
            return f"{attr}:{value}"

    return f"object:{type(model).__module__}.{type(model).__qualname__}:{id(model)}"


def _audio_hash(audio: bytes) -> str:
    return hashlib.sha256(audio).hexdigest()


def _cache_key(model, audios: list[bytes]):
    return (
        _stable_model_id(model),
        str(_model_device(model)),
        _sample_rate(model),
        tuple(_audio_hash(bytes(audio)) for audio in audios),
    )


@torch.inference_mode()
def batch_encode(model, audios_list: list[bytes]):
    sample_rate = _sample_rate(model)
    device = _model_device(model)

    audios: list[torch.Tensor] = []
    for audio in audios_list:
        if isinstance(audio, bytes):
            wav, _ = librosa.load(io.BytesIO(audio), sr=sample_rate, mono=True)
            tensor = torch.from_numpy(wav)[None]
        elif torch.is_tensor(audio):
            tensor = audio
            if tensor.ndim == 1:
                tensor = tensor[None]
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        audios.append(tensor.float())

    if not audios:
        return []

    lengths = torch.tensor([audio.shape[-1] for audio in audios], device=device)
    max_length = int(lengths.max().item())

    print(f"Encode max length: {max_length / sample_rate:.2f}s")

    padded = torch.stack(
        [
            torch.nn.functional.pad(audio, (0, int(max_length - audio.shape[-1])))
            for audio in audios
        ]
    ).to(device=device)

    with _autocast_for_device(device):
        features, feature_lengths = model.encode(padded, audio_lengths=lengths)

    features = features.detach().cpu()
    feature_lengths = feature_lengths.detach().cpu()

    return [
        feature[..., : int(length.item())].contiguous()
        for feature, length in zip(features, feature_lengths)
    ]


@cached(cache=LRUCache(maxsize=CACHE_MAXSIZE), key=_cache_key)
def cached_vqgan_batch_encode(model, audios: list[bytes]):
    return batch_encode(model, audios)


@torch.inference_mode()
def batch_vqgan_decode(model, features):
    if not features:
        return []

    device = _model_device(model)

    normalized = []
    for feature in features:
        if not torch.is_tensor(feature):
            feature = torch.tensor(feature, dtype=torch.long)
        normalized.append(feature.detach().long().cpu())

    lengths = torch.tensor([feature.shape[-1] for feature in normalized], device=device)
    max_length = int(lengths.max().item())

    padded = torch.stack(
        [
            torch.nn.functional.pad(feature, (0, int(max_length - feature.shape[-1])))
            for feature in normalized
        ]
    ).to(device=device)

    audios = []

    with _autocast_for_device(device):
        if hasattr(model, "from_indices"):
            frame_length = int(getattr(model, "frame_length", 512))

            for i in range(0, padded.shape[0], MICRO_BATCH_SIZE):
                batch = padded[i : i + MICRO_BATCH_SIZE]
                decoded = model.from_indices(batch)[0]

                decoded = decoded.detach().float().cpu()
                batch_lengths = lengths[i : i + MICRO_BATCH_SIZE].detach().cpu()

                for audio, code_len in zip(decoded, batch_lengths):
                    sample_len = int(code_len.item()) * frame_length
                    audio = audio[..., :sample_len]
                    audios.append(audio.squeeze().numpy())

        else:
            decoded_chunks = []
            audio_length_chunks = []

            for i in range(0, padded.shape[0], MICRO_BATCH_SIZE):
                audio, audio_length = model.decode(
                    padded[i : i + MICRO_BATCH_SIZE],
                    feature_lengths=lengths[i : i + MICRO_BATCH_SIZE],
                )
                decoded_chunks.append(audio)
                audio_length_chunks.append(audio_length)

            decoded = torch.cat(decoded_chunks, dim=0).detach().float().cpu()
            audio_lengths = torch.cat(audio_length_chunks, dim=0).detach().cpu()

            for audio, audio_len in zip(decoded, audio_lengths):
                audios.append(audio[..., : int(audio_len.item())].squeeze().numpy())

    return audios

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

try:
    import numpy as np
except ImportError:
    np = None


def estimate_code_frames(codes: Any | None) -> int:
    """
    Estimate the number of frames in the codes.
    - None -> 0
    - torch.Tensor / np.ndarray / object with .shape -> int(shape[-1]), if shape is not empty
    - list/tuple:
        - empty -> 0
        - nested list/tuple [[...], [...]] -> len(first row)
        - flat list -> len(list)
    - otherwise -> 0
    """

    if codes is None:
        return 0

    if hasattr(codes, "shape"):
        shape = codes.shape
        if len(shape) > 0:
            return int(shape[-1])
        return 0

    if isinstance(codes, (list, tuple)):
        if not codes:
            return 0
        if isinstance(codes[0], (list, tuple, torch.Tensor)) or (
            np is not None and isinstance(codes[0], np.ndarray)
        ):
            # Nested list/tuple/tensor/ndarray
            if hasattr(codes[0], "shape"):
                shape = codes[0].shape
                if len(shape) > 0:
                    return int(shape[-1])
                return 0
            if isinstance(codes[0], (list, tuple)):
                return len(codes[0])
            return 0  # Should not happen based on check
        # Flat list
        return len(codes)

    return 0


def normalize_codes(
    codes: Any,
    *,
    expected_codebooks: int | None = None,
    name: str = "codes",
) -> torch.Tensor:
    """
    Standardize codes to a [num_codebooks, T] CPU long tensor.
    """

    if codes is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(codes, torch.Tensor):
        if np is not None and isinstance(codes, np.ndarray):
            codes = torch.from_numpy(codes)
        else:
            try:
                codes = torch.tensor(codes)
            except Exception as e:
                raise ValueError(f"Failed to convert {name} to torch.Tensor: {e}")

    # Standardize shape
    # We expect [C, T] or [1, C, T]
    if codes.ndim == 3:
        if codes.shape[0] == 1:
            codes = codes[0]
        else:
            raise ValueError(
                f"Unexpected 3D tensor shape {codes.shape} for {name}, "
                f"expected [1, C, T] or [C, T]"
            )

    if codes.ndim != 2:
        raise ValueError(
            f"Unexpected tensor ndim {codes.ndim}, shape {codes.shape} for {name}, "
            f"expected [num_codebooks, T]"
        )

    if codes.shape[0] == 0:
        raise ValueError(f"{name} has zero codebooks (C=0), shape {codes.shape}")

    if codes.shape[1] == 0:
        raise ValueError(f"{name} is empty (T=0), shape {codes.shape}")

    if expected_codebooks is not None and codes.shape[0] != expected_codebooks:
        raise ValueError(
            f"{name} codebook count mismatch: "
            f"got {codes.shape[0]}, expected {expected_codebooks}. "
            f"Shape: {codes.shape}"
        )

    return codes.detach().cpu().long().contiguous()


def crop_codes_tail(
    codes: Any | None,
    max_frames: int,
    *,
    expected_codebooks: int | None = None,
    name: str = "codes",
) -> torch.Tensor | None:
    """
    Crop codes from the tail to at most max_frames.
    """

    if codes is None or max_frames <= 0:
        return None

    codes = normalize_codes(codes, expected_codebooks=expected_codebooks, name=name)

    if codes.shape[1] <= max_frames:
        return codes

    return codes[:, -max_frames:].contiguous().clone()


def load_codes_pt(
    path_or_bytes: str | Path | bytes | bytearray | io.BytesIO,
    *,
    expected_codebooks: int | None = None,
    name: str = "codes",
) -> torch.Tensor:
    """
    Load torch tensor from path or bytes and normalize it.
    """

    if isinstance(path_or_bytes, (bytes, bytearray)):
        path_or_bytes = io.BytesIO(path_or_bytes)

    try:
        # torch.load weights_only is available since torch 1.13
        # Use it if possible for security
        import inspect

        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters:
            loaded = torch.load(path_or_bytes, map_location="cpu", weights_only=True)
        else:
            loaded = torch.load(path_or_bytes, map_location="cpu")

        if isinstance(loaded, (tuple, list)) and len(loaded) > 0:
            # Often models save (codes, ...) or similar
            if isinstance(loaded[0], (torch.Tensor, list, tuple)) or (
                np is not None and isinstance(loaded[0], np.ndarray)
            ):
                loaded = loaded[0]

        return normalize_codes(loaded, expected_codebooks=expected_codebooks, name=name)
    except Exception as e:
        raise ValueError(f"Failed to load {name} from {path_or_bytes}: {e}") from e


def save_codes_pt(
    codes: Any,
    path: str | Path,
    *,
    expected_codebooks: int | None = None,
    name: str = "codes",
) -> torch.Tensor:
    """
    Normalize and save codes to a file.
    """

    codes = normalize_codes(codes, expected_codebooks=expected_codebooks, name=name)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(codes, path)
    return codes


def expected_codebooks_from_decoder(decoder_model: Any) -> int | None:
    """
    Try to infer the number of codebooks from the decoder model.
    """

    if decoder_model is None:
        return None

    quantizer = getattr(decoder_model, "quantizer", None)
    if quantizer is None:
        return None

    # Check for DownsampleResidualVectorQuantize (DAC)
    from fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize

    if isinstance(quantizer, DownsampleResidualVectorQuantize):
        # semantic (1) + acoustic (n_codebooks)
        return getattr(quantizer.quantizer, "n_codebooks", 0) + 1

    # Fallback to n_codebooks attribute
    return getattr(quantizer, "n_codebooks", None)


def validate_codes_for_decoder(
    codes: Any, decoder_model: Any, *, name="codes"
) -> torch.Tensor:
    """
    Normalize and validate codes for a specific decoder model.
    """
    expected = expected_codebooks_from_decoder(decoder_model)
    return normalize_codes(codes, expected_codebooks=expected, name=name)

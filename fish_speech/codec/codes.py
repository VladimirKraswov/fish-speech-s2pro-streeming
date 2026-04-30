from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a project dependency
    np = None


def _shape(codes: Any) -> tuple[int, ...] | None:
    shape = getattr(codes, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _range_text(codes: torch.Tensor) -> str:
    if codes.numel() == 0:
        return "empty"
    return f"min={int(codes.min().item())} max={int(codes.max().item())}"


def _positive_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return default
    if not isinstance(value, (int, float)):
        return default
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _is_downsample_rvq(quantizer: Any) -> bool:
    if quantizer.__class__.__name__ == "DownsampleResidualVectorQuantize":
        return True
    explicit_attrs = getattr(quantizer, "__dict__", {})
    return "semantic_quantizer" in explicit_attrs and "quantizer" in explicit_attrs


def estimate_code_frames(codes: Any | None) -> int:
    """
    Estimate the number of VQ/DAC frames without fully normalizing the payload.
    """

    if codes is None:
        return 0

    shape = _shape(codes)
    if shape is not None:
        return int(shape[-1]) if shape else 0

    if isinstance(codes, (list, tuple)):
        if not codes:
            return 0
        first = codes[0]
        first_shape = _shape(first)
        if first_shape is not None:
            return int(first_shape[-1]) if first_shape else 0
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(codes)

    return 0


def normalize_codes(
    codes: Any,
    *,
    expected_codebooks: int | None = None,
    name: str = "codes",
) -> torch.Tensor:
    """
    Standardize VQ/DAC codes to a CPU long tensor with shape ``[C, T]``.

    Accepted input shapes are ``[C, T]`` and ``[1, C, T]``. The returned tensor is
    detached, contiguous, on CPU, and safe to cache or persist.
    """

    if codes is None:
        raise ValueError(f"{name} cannot be None")

    if isinstance(codes, torch.Tensor):
        tensor = codes.detach()
    elif np is not None and isinstance(codes, np.ndarray):
        tensor = torch.from_numpy(codes)
    elif isinstance(codes, (list, tuple)):
        if not codes:
            raise ValueError(f"{name} is empty")
        try:
            tensor = torch.tensor(codes)
        except Exception as exc:
            raise ValueError(f"Failed to convert {name} to torch.Tensor: {exc}") from exc
    else:
        try:
            tensor = torch.as_tensor(codes)
        except Exception as exc:
            raise ValueError(f"Failed to convert {name} to torch.Tensor: {exc}") from exc

    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"{name} has unexpected 3D shape {tuple(tensor.shape)}; "
                "expected [1, C, T] or [C, T]"
            )
        tensor = tensor[0]

    if tensor.ndim != 2:
        raise ValueError(
            f"{name} has unexpected ndim={tensor.ndim}, shape={tuple(tensor.shape)}; "
            "expected [C, T]"
        )

    codebooks = int(tensor.shape[0])
    frames = int(tensor.shape[1])
    if codebooks <= 0:
        raise ValueError(f"{name} has zero codebooks, shape={tuple(tensor.shape)}")
    if frames <= 0:
        raise ValueError(f"{name} has zero frames, shape={tuple(tensor.shape)}")

    if expected_codebooks is not None:
        expected_codebooks = int(expected_codebooks)
        if expected_codebooks <= 0:
            raise ValueError(
                f"{name} expected_codebooks must be positive, got {expected_codebooks}"
            )
        if codebooks != expected_codebooks:
            raise ValueError(
                f"{name} codebook count mismatch: got {codebooks}, "
                f"expected {expected_codebooks}, shape={tuple(tensor.shape)}"
            )

    return tensor.to(device="cpu", dtype=torch.long).contiguous()


def crop_codes_tail(
    codes: Any | None,
    max_frames: int,
    *,
    expected_codebooks: int | None = None,
    name: str = "codes",
) -> torch.Tensor | None:
    """
    Return a normalized copy of the last ``max_frames`` frames.
    """

    if codes is None or max_frames <= 0:
        return None

    normalized = normalize_codes(
        codes,
        expected_codebooks=expected_codebooks,
        name=name,
    )

    if normalized.shape[1] <= max_frames:
        return normalized

    return normalized[:, -max_frames:].contiguous().clone()


def _extract_codes_payload(loaded: Any, *, name: str) -> Any:
    if isinstance(loaded, dict):
        for key in ("codes", "tokens", "indices"):
            if key in loaded:
                return loaded[key]
        raise ValueError(
            f"{name} payload dict does not contain one of: codes, tokens, indices"
        )

    if isinstance(loaded, (tuple, list)):
        if not loaded:
            raise ValueError(f"{name} payload list/tuple is empty")
        return loaded[0]

    return loaded


def load_codes_pt(
    path_or_bytes: str | Path | bytes | bytearray | io.BytesIO,
    *,
    expected_codebooks: int | None = None,
    name: str = "codes",
) -> torch.Tensor:
    """
    Load a ``.pt`` codes payload and normalize it to ``[C, T]`` CPU long.
    """

    source: str | Path | io.BytesIO
    if isinstance(path_or_bytes, (bytes, bytearray)):
        source = io.BytesIO(path_or_bytes)
    else:
        source = path_or_bytes

    if isinstance(source, io.BytesIO):
        source.seek(0)

    try:
        try:
            loaded = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(source, map_location="cpu")

        payload = _extract_codes_payload(loaded, name=name)
        return normalize_codes(
            payload,
            expected_codebooks=expected_codebooks,
            name=name,
        )
    except Exception as exc:
        raise ValueError(f"Failed to load {name} from {path_or_bytes}: {exc}") from exc


def save_codes_pt(
    codes: Any,
    path: str | Path,
    *,
    expected_codebooks: int | None = None,
    name: str = "codes",
) -> torch.Tensor:
    """
    Normalize and save codes as a CPU long tensor.
    """

    normalized = normalize_codes(
        codes,
        expected_codebooks=expected_codebooks,
        name=name,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(normalized, path)
    return normalized


def expected_codebooks_from_decoder(decoder_model: Any) -> int | None:
    """
    Infer decoder codebook count when the decoder exposes quantizer metadata.
    """

    if decoder_model is None:
        return None

    quantizer = getattr(decoder_model, "quantizer", None)
    if quantizer is None:
        return None

    if _is_downsample_rvq(quantizer):
        inner = getattr(quantizer, "quantizer", None)
        n_codebooks = _positive_int(getattr(inner, "n_codebooks", None))
        return n_codebooks + 1 if n_codebooks is not None else None

    return _positive_int(getattr(quantizer, "n_codebooks", None))


def _raise_range_error(
    *,
    name: str,
    row_name: str,
    codes: torch.Tensor,
    expected_range: str,
) -> None:
    row_label = f" {row_name}" if row_name else ""
    raise ValueError(
        f"{name}{row_label} codes out of range {expected_range}: "
        f"shape={tuple(codes.shape)} {_range_text(codes)}"
    )


def _validate_row_range(
    codes: torch.Tensor,
    *,
    name: str,
    row_name: str,
    min_value: int,
    max_exclusive: int,
) -> None:
    if codes.numel() == 0:
        return
    if torch.any(codes < min_value) or torch.any(codes >= max_exclusive):
        _raise_range_error(
            name=name,
            row_name=row_name,
            codes=codes,
            expected_range=f"[{min_value}, {max_exclusive})",
        )


def validate_codes_for_decoder(
    codes: Any,
    decoder_model: Any,
    *,
    name: str = "codes",
) -> torch.Tensor:
    """
    Normalize and validate codes against decoder expectations.

    For ``DownsampleResidualVectorQuantize``, row 0 is the semantic codebook and
    rows 1+ are acoustic codebooks.
    """

    expected_codebooks = expected_codebooks_from_decoder(decoder_model)
    normalized = normalize_codes(
        codes,
        expected_codebooks=expected_codebooks,
        name=name,
    )

    quantizer = getattr(decoder_model, "quantizer", None) if decoder_model else None
    if quantizer is None:
        if torch.any(normalized < 0):
            raise ValueError(
                f"{name} contains negative codes: "
                f"shape={tuple(normalized.shape)} {_range_text(normalized)}"
            )
        return normalized

    if _is_downsample_rvq(quantizer):
        sem_q = getattr(quantizer, "semantic_quantizer", None)
        ac_q = getattr(quantizer, "quantizer", None)
        semantic_size = _positive_int(getattr(sem_q, "codebook_size", None), 4096)
        acoustic_size = _positive_int(getattr(ac_q, "codebook_size", None), 1024)

        assert semantic_size is not None
        assert acoustic_size is not None

        _validate_row_range(
            normalized[0],
            name=name,
            row_name="semantic",
            min_value=0,
            max_exclusive=semantic_size,
        )
        if normalized.shape[0] > 1:
            _validate_row_range(
                normalized[1:],
                name=name,
                row_name="acoustic",
                min_value=0,
                max_exclusive=acoustic_size,
            )
        return normalized

    size = _positive_int(getattr(quantizer, "codebook_size", None))
    if size is not None:
        _validate_row_range(
            normalized,
            name=name,
            row_name="",
            min_value=0,
            max_exclusive=size,
        )
    elif torch.any(normalized < 0):
        raise ValueError(
            f"{name} contains negative codes: "
            f"shape={tuple(normalized.shape)} {_range_text(normalized)}"
        )

    return normalized


def validate_codes_for_decoder_device(
    codes: torch.Tensor,
    decoder_model: Any,
    *,
    name: str = "codes",
) -> torch.Tensor:
    """
    Validate already-generated decoder codes without forcing a CPU roundtrip.

    This is intended for hot streaming paths where codes are already tensors on
    the LLaMA/DAC device. It keeps the tensor on-device and only synchronizes if
    validation is explicitly requested by the caller.
    """

    if not isinstance(codes, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(codes).__name__}")

    if codes.ndim == 3:
        if codes.shape[0] != 1:
            raise ValueError(
                f"{name} has unexpected 3D shape {tuple(codes.shape)}; "
                "expected [1, C, T] or [C, T]"
            )
        codes = codes[0]

    if codes.ndim != 2:
        raise ValueError(
            f"{name} has unexpected ndim={codes.ndim}, shape={tuple(codes.shape)}; "
            "expected [C, T]"
        )

    codebooks = int(codes.shape[0])
    frames = int(codes.shape[1])
    if codebooks <= 0:
        raise ValueError(f"{name} has zero codebooks, shape={tuple(codes.shape)}")
    if frames <= 0:
        raise ValueError(f"{name} has zero frames, shape={tuple(codes.shape)}")

    expected_codebooks = expected_codebooks_from_decoder(decoder_model)
    if expected_codebooks is not None and codebooks != int(expected_codebooks):
        raise ValueError(
            f"{name} codebook count mismatch: got {codebooks}, "
            f"expected {expected_codebooks}, shape={tuple(codes.shape)}"
        )

    if codes.dtype != torch.long:
        codes = codes.to(dtype=torch.long)

    quantizer = getattr(decoder_model, "quantizer", None) if decoder_model else None
    if quantizer is None:
        if torch.any(codes < 0).item():
            raise ValueError(
                f"{name} contains negative codes: "
                f"shape={tuple(codes.shape)} {_range_text(codes)}"
            )
        return codes.contiguous()

    if _is_downsample_rvq(quantizer):
        sem_q = getattr(quantizer, "semantic_quantizer", None)
        ac_q = getattr(quantizer, "quantizer", None)
        semantic_size = _positive_int(getattr(sem_q, "codebook_size", None), 4096)
        acoustic_size = _positive_int(getattr(ac_q, "codebook_size", None), 1024)

        assert semantic_size is not None
        assert acoustic_size is not None

        if torch.any(codes[0] < 0).item() or torch.any(
            codes[0] >= semantic_size
        ).item():
            _raise_range_error(
                name=name,
                row_name="semantic",
                codes=codes[0],
                expected_range=f"[0, {semantic_size})",
            )
        if codes.shape[0] > 1 and (
            torch.any(codes[1:] < 0).item()
            or torch.any(codes[1:] >= acoustic_size).item()
        ):
            _raise_range_error(
                name=name,
                row_name="acoustic",
                codes=codes[1:],
                expected_range=f"[0, {acoustic_size})",
            )
        return codes.contiguous()

    size = _positive_int(getattr(quantizer, "codebook_size", None))
    if size is not None:
        if torch.any(codes < 0).item() or torch.any(codes >= size).item():
            _raise_range_error(
                name=name,
                row_name="",
                codes=codes,
                expected_range=f"[0, {size})",
            )
    elif torch.any(codes < 0).item():
        raise ValueError(
            f"{name} contains negative codes: "
            f"shape={tuple(codes.shape)} {_range_text(codes)}"
        )

    return codes.contiguous()

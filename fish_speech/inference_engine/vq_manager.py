import gc
import os
import time
from dataclasses import dataclass
from typing import Callable

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class CUDAMemorySnapshot:
    free_gb: float | None
    allocated_gb: float | None
    reserved_gb: float | None
    total_gb: float | None


@dataclass(frozen=True)
class DACDecodeStats:
    microchunk_size: int
    microchunk_count: int
    retries: int
    cleanup_count: int
    decode_ms: float
    snapshot_before: CUDAMemorySnapshot
    snapshot_after: CUDAMemorySnapshot


class VQManager:
    def __init__(self):
        self.decoder_model: DAC
        self.load_audio: Callable

    def _decoder_device(self) -> torch.device:
        device = getattr(self.decoder_model, "device", None)
        if device is not None:
            return torch.device(device)

        try:
            return next(self.decoder_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _decoder_dtype(self) -> torch.dtype:
        try:
            return next(self.decoder_model.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _cuda_index(self, device: torch.device | str | None = None) -> int | None:
        if not torch.cuda.is_available():
            return None

        if device is None:
            return torch.cuda.current_device()

        if isinstance(device, str):
            device = torch.device(device)

        if device.type != "cuda":
            return None

        return torch.cuda.current_device() if device.index is None else device.index

    def cuda_memory_snapshot(
        self, device: torch.device | str | None = None
    ) -> CUDAMemorySnapshot:
        cuda_idx = self._cuda_index(device)
        if cuda_idx is None:
            return CUDAMemorySnapshot(
                free_gb=None,
                allocated_gb=None,
                reserved_gb=None,
                total_gb=None,
            )

        free_gb = None
        total_gb = None
        if hasattr(torch.cuda, "mem_get_info"):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_idx)
                free_gb = free_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
            except Exception:
                free_gb = None
                total_gb = None

        if total_gb is None:
            props = torch.cuda.get_device_properties(cuda_idx)
            total_gb = props.total_memory / (1024**3)

        allocated_gb = torch.cuda.memory_allocated(cuda_idx) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(cuda_idx) / (1024**3)

        if free_gb is None:
            used_bytes = max(
                torch.cuda.memory_reserved(cuda_idx),
                torch.cuda.memory_allocated(cuda_idx),
            )
            free_gb = max(0, (total_gb * (1024**3)) - used_bytes) / (1024**3)

        return CUDAMemorySnapshot(
            free_gb=free_gb,
            allocated_gb=allocated_gb,
            reserved_gb=reserved_gb,
            total_gb=total_gb,
        )

    def _light_cuda_cleanup(self, device: torch.device | str | None = None) -> None:
        cuda_idx = self._cuda_index(device)
        if cuda_idx is None:
            return

        try:
            torch.cuda.synchronize(cuda_idx)
        except Exception:
            pass

        gc.collect()
        torch.cuda.empty_cache()

    def _is_oom_error(self, err: BaseException) -> bool:
        return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()

    def _decode_codes_chunk_cpu_fallback(self, codes: torch.Tensor) -> torch.Tensor:
        original_device = self._decoder_device()
        original_dtype = self._decoder_dtype()

        logger.warning(
            "DAC emergency CPU fallback chunk_len={} original_device={}",
            int(codes.shape[1]),
            original_device,
        )

        try:
            self.decoder_model.to(device="cpu", dtype=original_dtype)
            with torch.inference_mode():
                audio = self.decoder_model.from_indices(
                    codes.to(device="cpu", dtype=torch.long, non_blocking=False)[None]
                )[0].squeeze()
            return audio.detach().cpu()
        finally:
            self.decoder_model.to(device=original_device, dtype=original_dtype)
            self._light_cuda_cleanup(original_device)

    def _decode_codes_window(
        self,
        codes: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        audio = self.decoder_model.from_indices(
            codes.to(device=device, dtype=torch.long, non_blocking=True)[None]
        )[0].squeeze()
        return audio.detach().cpu()

    def decode_vq_tokens(
        self,
        codes: torch.Tensor,
        *,
        req_tag: str = "na",
        segment_idx: int | None = None,
        max_codes_per_step: int | None = None,
    ) -> tuple[torch.Tensor, DACDecodeStats]:
        if not isinstance(codes, torch.Tensor):
            raise TypeError(f"codes must be a torch.Tensor, got {type(codes)}")

        if codes.ndim != 2:
            raise ValueError(
                f"Expected codes shape [num_codebooks, time], got {codes.shape}"
            )

        if not isinstance(self.decoder_model, DAC):
            raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

        if codes.shape[1] == 0:
            empty = torch.empty(0, dtype=torch.float32)
            snapshot = self.cuda_memory_snapshot(self._decoder_device())
            return empty, DACDecodeStats(
                microchunk_size=0,
                microchunk_count=0,
                retries=0,
                cleanup_count=0,
                decode_ms=0.0,
                snapshot_before=snapshot,
                snapshot_after=snapshot,
            )

        device = self._decoder_device()
        requested_step = max(
            1,
            max_codes_per_step
            if max_codes_per_step is not None
            else _env_int("FISH_DAC_MAX_CODES_PER_STEP", 4),
        )
        current_step = requested_step
        min_free_gb = _env_float("FISH_DAC_MIN_FREE_GB", 3.0)
        resume_free_gb = _env_float("FISH_DAC_RESUME_FREE_GB", 5.0)
        critical_free_gb = _env_float("FISH_DAC_CRITICAL_FREE_GB", 1.5)
        allow_cpu_fallback = _env_flag("FISH_DAC_CPU_FALLBACK_ON_OOM", False)

        snapshot_before = self.cuda_memory_snapshot(device)
        if snapshot_before.free_gb is not None:
            if snapshot_before.free_gb < critical_free_gb:
                current_step = 1
                self._light_cuda_cleanup(device)
            elif snapshot_before.free_gb < min_free_gb:
                current_step = max(1, min(current_step, requested_step // 2))

        decode_started = time.perf_counter()
        audio_parts: list[torch.Tensor] = []
        retries = 0
        cleanup_count = 0
        microchunk_count = 0
        offset = 0

        while offset < codes.shape[1]:
            step = min(current_step, int(codes.shape[1]) - offset)
            window = codes[:, offset : offset + step]

            while True:
                try:
                    part = self._decode_codes_window(window, device=device)
                    audio_parts.append(part)
                    microchunk_count += 1
                    offset += int(window.shape[1])
                    break
                except Exception as err:
                    if not self._is_oom_error(err):
                        raise

                    retries += 1
                    cleanup_count += 1
                    self._light_cuda_cleanup(device)

                    if int(window.shape[1]) > 1:
                        next_step = max(1, int(window.shape[1]) // 2)
                        logger.warning(
                            "dac: oom req={} seg={} offset={} step={} retry_step={}",
                            req_tag,
                            segment_idx,
                            offset,
                            int(window.shape[1]),
                            next_step,
                        )
                        current_step = min(current_step, next_step)
                        step = next_step
                        window = codes[:, offset : offset + step]
                        continue

                    if allow_cpu_fallback:
                        part = self._decode_codes_chunk_cpu_fallback(window)
                        audio_parts.append(part)
                        microchunk_count += 1
                        offset += int(window.shape[1])
                        break

                    raise

            snapshot_now = self.cuda_memory_snapshot(device)
            if snapshot_now.free_gb is not None:
                if snapshot_now.free_gb < critical_free_gb:
                    current_step = 1
                elif snapshot_now.free_gb < min_free_gb:
                    current_step = max(1, min(current_step, requested_step // 2))
                elif snapshot_now.free_gb >= resume_free_gb:
                    current_step = min(requested_step, max(1, current_step * 2))

        if len(audio_parts) == 1:
            audio = audio_parts[0]
        else:
            audio = torch.cat(audio_parts, dim=-1)

        snapshot_after = self.cuda_memory_snapshot(device)
        stats = DACDecodeStats(
            microchunk_size=current_step,
            microchunk_count=microchunk_count,
            retries=retries,
            cleanup_count=cleanup_count,
            decode_ms=(time.perf_counter() - decode_started) * 1000.0,
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

        logger.info(
            "dac: decode_summary req={} seg={} frames={} microchunk={} parts={} retries={} cleanup={} decode_ms={:.1f} free_gb={} alloc_gb={} reserved_gb={}",
            req_tag,
            segment_idx,
            int(codes.shape[1]),
            stats.microchunk_size,
            stats.microchunk_count,
            stats.retries,
            stats.cleanup_count,
            stats.decode_ms,
            f"{stats.snapshot_after.free_gb:.2f}" if stats.snapshot_after.free_gb is not None else "na",
            f"{stats.snapshot_after.allocated_gb:.2f}" if stats.snapshot_after.allocated_gb is not None else "na",
            f"{stats.snapshot_after.reserved_gb:.2f}" if stats.snapshot_after.reserved_gb is not None else "na",
        )

        return audio, stats

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate

            reference_audio_content = self.load_audio(reference_audio, sample_rate)
            audios = torch.from_numpy(reference_audio_content)[None, None, :]
            audio_lengths = torch.tensor([audios.shape[2]], dtype=torch.long)
            logger.info(
                "Loaded audio with {:.2f} seconds",
                audios.shape[2] / sample_rate,
            )

            if isinstance(self.decoder_model, DAC):
                device = getattr(self.decoder_model, "device", None)
                on_cuda = device is not None and str(device).startswith("cuda")

                if on_cuda:
                    torch.cuda.empty_cache()
                    self.decoder_model.to("cpu")
                    try:
                        prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][
                            0
                        ]
                    finally:
                        self.decoder_model.to(device)
                else:
                    prompt_tokens = self.decoder_model.encode(audios, audio_lengths)[0][
                        0
                    ]

                logger.info("Encoded prompt: {}", prompt_tokens.shape)
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens

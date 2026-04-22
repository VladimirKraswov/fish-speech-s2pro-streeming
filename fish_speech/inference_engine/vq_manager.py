import gc
import os
from typing import Callable

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC


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


class VQManager:
    def __init__(self):
        # Make Pylance happy
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

    def _cuda_free_gb(self, device: torch.device | str | None = None) -> float | None:
        cuda_idx = self._cuda_index(device)
        if cuda_idx is None:
            return None
        
        if hasattr(torch.cuda, "mem_get_info"):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(cuda_idx)
                return free_bytes / (1024**3)
            except Exception:
                pass        

        props = torch.cuda.get_device_properties(cuda_idx)
        total_bytes = props.total_memory
        reserved_bytes = torch.cuda.memory_reserved(cuda_idx)
        allocated_bytes = torch.cuda.memory_allocated(cuda_idx)
        used_bytes = max(reserved_bytes, allocated_bytes)

        free_bytes = max(0, total_bytes - used_bytes)
        return free_bytes / (1024**3)

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

    def _decode_codes_chunk_cpu_fallback(self, codes: torch.Tensor) -> torch.Tensor:
        original_device = self._decoder_device()
        original_dtype = self._decoder_dtype()

        logger.warning(
            "DAC emergency CPU fallback activated chunk_len={} original_device={}",
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
            return
        gc.collect()
        torch.cuda.empty_cache()

    def _maybe_cleanup_before_decode(
        self,
        *,
        device: torch.device | str | None,
        chunk_len: int,
        reason: str,
    ) -> None:
        """
        Перед декодированием проверяем запас VRAM.
        Если запас слишком маленький — делаем лёгкую очистку.
        Это не так дорого, как полный clear_caches, но часто спасает
        от падения на мелкой дополнительной аллокации внутри DAC.
        """
        min_free_gb = _env_float("FISH_DAC_MIN_FREE_GB", 1.0)
        free_before = self._cuda_free_gb(device)
        if free_before is None or free_before >= min_free_gb:
            return

        logger.warning(
            "DAC pre-decode cleanup: low free VRAM {:.2f} GB < {:.2f} GB "
            "(chunk_len={}, reason={})",
            free_before,
            min_free_gb,
            chunk_len,
            reason,
        )

        cuda_idx = self._cuda_index(device)
        if cuda_idx is not None:
            try:
                torch.cuda.synchronize(cuda_idx)
            except Exception:
                pass

            self._light_cuda_cleanup(device)

            if chunk_len <= 1 and os.environ.get(
                "FISH_DAC_CPU_FALLBACK_ON_OOM", ""
            ).strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}:
                return self._decode_codes_chunk_cpu_fallback(codes)

        free_after = self._cuda_free_gb(device)
        if free_after is not None:
            logger.info(
                "DAC pre-decode cleanup done: free VRAM {:.2f} -> {:.2f} GB",
                free_before,
                free_after,
            )

    def _decode_codes_chunk(
        self,
        codes: torch.Tensor,
        *,
        depth: int = 0,
    ) -> torch.Tensor:
        """
        Декодирует один кусок кодов.
        Если даже этот кусок ловит OOM — режем его пополам и пытаемся
        декодировать рекурсивно. Это даёт "аварийный" режим вместо падения
        всего процесса.
        """
        if codes.ndim != 2:
            raise ValueError(f"Expected codes with shape [num_codebooks, time], got {codes.shape}")

        chunk_len = int(codes.shape[1])
        if chunk_len == 0:
            return torch.empty(0, dtype=torch.float32)

        device = getattr(self.decoder_model, "device", None)
        if device is None:
            device = codes.device if codes.is_cuda else torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        self._maybe_cleanup_before_decode(
            device=device,
            chunk_len=chunk_len,
            reason=f"decode_chunk_depth_{depth}",
        )

        try:
            # Важно: кодируем на GPU, но результат сразу выносим на CPU,
            # чтобы не держать уже готовый аудиофрагмент в VRAM.
            audio = self.decoder_model.from_indices(
                codes.to(device=device, dtype=torch.long, non_blocking=True)[None]
            )[0].squeeze()

            return audio.detach().cpu()

        except RuntimeError as e:
            oom = "out of memory" in str(e).lower()
            if not oom or chunk_len <= 1:
                raise

            # Если даже этот кусок не влез — очищаем кеш и режем пополам.
            logger.warning(
                "DAC chunk decode OOM at depth={} chunk_len={}, split and retry",
                depth,
                chunk_len,
            )
            self._light_cuda_cleanup()

            mid = chunk_len // 2
            left = self._decode_codes_chunk(codes[:, :mid], depth=depth + 1)
            right = self._decode_codes_chunk(codes[:, mid:], depth=depth + 1)
            return torch.cat([left, right], dim=-1)

    def decode_vq_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Декодирует VQ-коды в waveform.

        Ключевая идея:
        - не пихать в DAC слишком длинный кусок за один раз;
        - при OOM уметь автоматически дробить кусок ещё сильнее;
        - переносить уже готовые аудиочасти на CPU как можно раньше.
        """
        if not isinstance(codes, torch.Tensor):
            raise TypeError(f"codes must be a torch.Tensor, got {type(codes)}")

        if codes.ndim != 2:
            raise ValueError(
                f"Expected codes shape [num_codebooks, time], got {codes.shape}"
            )

        chunk_len = int(codes.shape[1])
        logger.info("VQ features: {} (stream chunk={})", codes.shape, chunk_len)

        if not isinstance(self.decoder_model, DAC):
            raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

        if chunk_len == 0:
            return torch.empty(0, dtype=torch.float32)

        # Сколько semantic-кадров DAC пытаемся декодировать за один вызов.
        # 4 — хороший старт для стабильности на почти забитой 5090.
        max_codes_per_step = max(1, _env_int("FISH_DAC_MAX_CODES_PER_STEP", 4))

        decoder_device = self._decoder_device()
        free_gb = self._cuda_free_gb(decoder_device)
        low_free_gb = _env_float("FISH_DAC_LOW_FREE_GB", 2.5)
        critical_free_gb = _env_float("FISH_DAC_CRITICAL_FREE_GB", 1.25)

        if free_gb is not None:
            if free_gb < critical_free_gb:
                max_codes_per_step = 1
            elif free_gb < low_free_gb:
                max_codes_per_step = min(max_codes_per_step, 2)

            logger.info(
                "DAC decode budget: free_vram_gb={:.2f} max_codes_per_step={}",
                free_gb,
                max_codes_per_step,
            )

        if chunk_len <= max_codes_per_step:
            return self._decode_codes_chunk(codes)

        audio_parts: list[torch.Tensor] = []
        empty_between_microchunks = os.environ.get(
            "FISH_DAC_EMPTY_CACHE_BETWEEN_MICROCHUNKS", ""
        ).strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}

        for start in range(0, chunk_len, max_codes_per_step):
            end = min(chunk_len, start + max_codes_per_step)
            sub_codes = codes[:, start:end]

            logger.info(
                "DAC micro-decode: frames {}:{} / {}",
                start,
                end,
                chunk_len,
            )

            audio_part = self._decode_codes_chunk(sub_codes)
            audio_parts.append(audio_part)

            if empty_between_microchunks and torch.cuda.is_available():
                self._light_cuda_cleanup()

        if len(audio_parts) == 1:
            return audio_parts[0]

        return torch.cat(audio_parts, dim=-1)

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            # Загружаем референс на CPU.
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate

            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            # Референс тоже держим максимально "дешёвым" по VRAM.
            audios = torch.from_numpy(reference_audio_content)[None, None, :]
            audio_lengths = torch.tensor([audios.shape[2]], dtype=torch.long)
            logger.info(
                f"Loaded audio with {audios.shape[2] / sample_rate:.2f} seconds"
            )

            if isinstance(self.decoder_model, DAC):
                device = getattr(self.decoder_model, "device", None)
                on_cuda = device is not None and str(device).startswith("cuda")

                if on_cuda:
                    # На длинном референсе энкодер может скушать очень много VRAM.
                    # Поэтому временно выносим DAC на CPU, кодируем референс там,
                    # а потом возвращаем модель обратно.
                    torch.cuda.empty_cache()
                    self.decoder_model.to("cpu")
                    try:
                        prompt_tokens = self.decoder_model.encode(
                            audios, audio_lengths
                        )[0][0]
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
# VQ encode/decode helpers used by the driver runtime.
from typing import Callable

import torch
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC


class VQManager:

    def __init__(self):
        self.decoder_model: DAC
        self.load_audio: Callable

    def _get_decoder_device(self) -> torch.device:
        device = getattr(self.decoder_model, "device", None)
        if device is not None:
            return torch.device(device)
        return next(self.decoder_model.parameters()).device

    def _get_decoder_dtype(self) -> torch.dtype:
        return next(self.decoder_model.parameters()).dtype

    def decode_vq_tokens(self, codes):
        chunk_len = codes.shape[1] if codes.dim() >= 2 else 0
        logger.info("VQ features: {} (stream chunk={})", codes.shape, chunk_len)

        if isinstance(self.decoder_model, DAC):
            return self.decoder_model.from_indices(codes[None])[0].squeeze()

        raise ValueError(f"Unknown model type: {type(self.decoder_model)}")

    def encode_reference(self, reference_audio, enable_reference_audio):
        if enable_reference_audio and reference_audio is not None:
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate

            reference_audio_content = self.load_audio(reference_audio, sample_rate)

            audios_cpu = torch.from_numpy(reference_audio_content).to(
                dtype=torch.float32
            )[None, None, :]
            audio_lengths_cpu = torch.tensor([audios_cpu.shape[2]], dtype=torch.long)
            logger.info(
                f"Loaded audio with {audios_cpu.shape[2] / sample_rate:.2f} seconds"
            )

            if isinstance(self.decoder_model, DAC):
                from contextlib import nullcontext

                lock = getattr(self, "_decoder_lock", None)
                lock_ctx = lock if lock is not None else nullcontext()

                with lock_ctx:
                    device = self._get_decoder_device()
                    dtype = self._get_decoder_dtype()
                    on_cuda = device.type == "cuda"

                    if on_cuda:
                        torch.cuda.empty_cache()
                        self.decoder_model.to(device="cpu", dtype=torch.float32)
                        try:
                            prompt_tokens = self.decoder_model.encode(
                                audios_cpu,
                                audio_lengths_cpu,
                            )[0][0]
                        finally:
                            self.decoder_model.to(device=device, dtype=dtype)
                    else:
                        audios = audios_cpu.to(device=device, dtype=dtype)
                        audio_lengths = audio_lengths_cpu.to(device=device)
                        prompt_tokens = self.decoder_model.encode(
                            audios,
                            audio_lengths,
                        )[0][0]

                prompt_tokens = prompt_tokens.detach().cpu()
                logger.info("Encoded prompt: {}", prompt_tokens.shape)
            else:
                raise ValueError(f"Unknown model type: {type(self.decoder_model)}")
        else:
            prompt_tokens = None
            logger.info("No reference audio provided")

        return prompt_tokens
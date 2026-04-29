from __future__ import annotations

from typing import Any, Iterator

import torch

try:
    import numpy as np
except ImportError:
    np = None

from fish_speech.codec.codes import normalize_codes
from fish_speech.driver.session import DriverSession, DriverSessionConfig
from fish_speech.driver.types import (
    DriverAudioChunkEvent,
    DriverErrorEvent,
    DriverFinalAudioEvent,
    DriverHealth,
    DriverReference,
    DriverStats,
    DriverSynthesisRequest,
    DriverTokenChunkEvent,
)


def _positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _acoustic_codebook_size_from_decoder(decoder_model: Any) -> int | None:
    """
    Infer the acoustic codebook size used by DAC/VQ decoder.

    Fish S2-Pro style DownsampleResidualVectorQuantize has:
    - row 0: semantic codebook, usually 4096;
    - rows 1+: acoustic codebooks, usually 1024.

    LLaMA may have a wider fast-layer vocab to represent semantic codes, so we
    pass the acoustic size into the worker and mask acoustic logits during
    generation instead of relying on decoder-side clamping or late validation.
    """
    quantizer = getattr(decoder_model, "quantizer", None)
    if quantizer is None:
        return None

    inner_acoustic = getattr(quantizer, "quantizer", None)
    if inner_acoustic is not None and hasattr(quantizer, "semantic_quantizer"):
        return _positive_int(getattr(inner_acoustic, "codebook_size", None))

    return _positive_int(getattr(quantizer, "codebook_size", None))


class FishSpeechDriver:
    def __init__(self, engine: Any, config: Any | None = None) -> None:
        self.engine = engine
        self.config = config
        self._closed = False
        self._sessions_opened = 0

    @classmethod
    def from_engine(cls, engine: Any, config: Any | None = None) -> "FishSpeechDriver":
        return cls(engine=engine, config=config)

    @classmethod
    def from_config(cls, config: Any | None = None) -> "FishSpeechDriver":
        from fish_speech.driver.config import load_driver_config
        import torch

        config = config or load_driver_config()
        model_cfg = config.model
        paths = config.paths
        precision = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[model_cfg.precision]

        device = model_cfg.device
        if torch.backends.mps.is_available():
            device = "mps"
        elif not torch.cuda.is_available():
            device = "cpu"

        return cls.from_model_paths(
            llama_checkpoint_path=paths.llama_checkpoint_path,
            decoder_checkpoint_path=paths.decoder_checkpoint_path,
            decoder_config_name=paths.decoder_config_name,
            device=device,
            precision=precision,
            compile=model_cfg.compile,
            config=config,
        )

    @classmethod
    def from_model_paths(
        cls,
        *,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str,
        decoder_config_name: str,
        device: str,
        precision: Any,
        compile: bool,
        config: Any | None = None,
        memory_info: dict | None = None,
    ) -> "FishSpeechDriver":
        from fish_speech.codec.codes import expected_codebooks_from_decoder
        from fish_speech.codec.dac import load_model as load_decoder_model
        from fish_speech.generation.worker import launch_thread_safe_queue
        from fish_speech.inference_engine import TTSInferenceEngine

        decoder_model = load_decoder_model(
            config_name=decoder_config_name,
            checkpoint_path=decoder_checkpoint_path,
            device=device,
        )
        expected_codebooks = expected_codebooks_from_decoder(decoder_model)
        acoustic_codebook_size = _acoustic_codebook_size_from_decoder(decoder_model)

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_checkpoint_path,
            device=device,
            precision=precision,
            compile=compile,
            memory_info=memory_info,
            expected_num_codebooks=expected_codebooks,
            acoustic_codebook_size=acoustic_codebook_size,
        )

        engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=compile,
            llama_device=device,
        )
        return cls(engine=engine, config=config)

    @property
    def sample_rate(self) -> int:
        decoder_model = self.engine.decoder_model
        if hasattr(decoder_model, "spec_transform"):
            return int(decoder_model.spec_transform.sample_rate)
        return int(decoder_model.sample_rate)

    def open_session(
        self,
        config: DriverSessionConfig | None = None,
        *,
        reference_id: str | None = None,
        references: list[DriverReference] | None = None,
    ) -> DriverSession:
        if self._closed:
            raise RuntimeError("fish speech driver is closed")
        if config is None:
            config = DriverSessionConfig(
                reference_id=reference_id,
                references=list(references or []),
            )
        self._sessions_opened += 1
        return DriverSession(driver=self, config=config)

    def synthesize_with_codes(
        self,
        request: DriverSynthesisRequest,
    ) -> dict[str, Any]:
        """
        Synthesize speech and collect audio plus generated VQ/DAC codes.

        This is the preferred helper for cached intro artifact generation. Use
        ``stream_audio=False`` and ``generation.stream_tokens=True`` when callers
        need final audio and complete codes for a cache entry.
        """

        audio_chunks = []
        code_chunks = []
        sample_rate = self.sample_rate
        final_audio = None

        for event in self.synthesize(request):
            if isinstance(event, DriverAudioChunkEvent):
                audio_chunks.append(event.audio)
                sample_rate = event.sample_rate
            elif isinstance(event, DriverTokenChunkEvent):
                if event.codes is not None:
                    codes = normalize_codes(event.codes, name="token chunk codes")
                    code_chunks.append(codes)
            elif isinstance(event, DriverFinalAudioEvent):
                final_audio = event.audio
                sample_rate = event.sample_rate
            elif isinstance(event, DriverErrorEvent):
                if event.error is not None:
                    raise RuntimeError(f"Synthesis error: {event.error}") from event.error
                raise RuntimeError("Synthesis error")

        audio = None
        if final_audio is not None:
            audio = final_audio
        elif audio_chunks:
            if np is not None:
                audio = np.concatenate(audio_chunks, axis=0)
            else:
                audio = audio_chunks

        codes = None
        if code_chunks:
            codes = normalize_codes(
                torch.cat(code_chunks, dim=1),
                name="collected synthesis codes",
            )

        return {
            "text": request.text,
            "sample_rate": sample_rate,
            "audio": audio,
            "codes": codes,
            "audio_chunks": audio_chunks,
            "code_chunks": code_chunks,
        }

    def synthesize_collect(
        self,
        request: DriverSynthesisRequest,
    ) -> dict[str, Any]:
        """
        Backward-compatible wrapper for ``synthesize_with_codes``.
        """

        return self.synthesize_with_codes(request)

    def synthesize(
        self,
        request: DriverSynthesisRequest,
    ) -> Iterator[
        DriverAudioChunkEvent
        | DriverFinalAudioEvent
        | DriverTokenChunkEvent
        | DriverErrorEvent
    ]:
        session = self.open_session(
            DriverSessionConfig(
                reference_id=request.reference_id,
                references=list(request.references),
                generation=request.generation,
                seed=request.seed,
                use_memory_cache=request.use_memory_cache,
                normalize=request.normalize,
                stream_audio=request.stream_audio,
            )
        )
        try:
            yield from session.synthesize_request(request)
        finally:
            session.close()

    def warmup(self) -> None:
        from fish_speech.driver.warmup import run_driver_warmup

        run_driver_warmup(self)

    def health(self) -> DriverHealth:
        return DriverHealth(ok=not self._closed)

    def stats(self) -> DriverStats:
        return DriverStats(sessions_opened=self._sessions_opened)

    def close(self) -> None:
        self._closed = True
        llama_queue = getattr(self.engine, "llama_queue", None)
        if llama_queue is not None:
            llama_queue.put(None)

    def list_reference_ids(self) -> list[str]:
        return self.engine.list_reference_ids()

    @property
    def references_dir(self):
        return self.engine.references_dir

    def add_reference(self, reference_id: str, wav_file_path: str, text: str) -> None:
        self.engine.add_reference(reference_id, wav_file_path, text)

    def add_reference_encoded(
        self,
        reference_id: str,
        codes_bytes: bytes,
        lab_text: str,
        stem: str | None = None,
    ) -> str:
        return self.engine.add_reference_encoded(
            reference_id,
            codes_bytes,
            lab_text,
            stem=stem,
        )

    def delete_reference(self, reference_id: str) -> None:
        self.engine.delete_reference(reference_id)

    def rename_reference(self, old_reference_id: str, new_reference_id: str) -> None:
        old_dir = self.references_dir / old_reference_id
        new_dir = self.references_dir / new_reference_id
        old_dir.rename(new_dir)
        if old_reference_id in self.engine.ref_by_id:
            self.engine.ref_by_id[new_reference_id] = self.engine.ref_by_id.pop(
                old_reference_id
            )
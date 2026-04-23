import os

import torch
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class ModelManager:
    def __init__(
        self,
        mode: str,
        device: str,
        half: bool,
        compile: bool,
        llama_checkpoint_path: str,
        decoder_checkpoint_path: str,
        decoder_config_name: str,
    ) -> None:

        self.mode = mode
        self.device = device
        self.half = half
        self.compile = compile

        self.precision = torch.half if half else torch.bfloat16

        # Check if MPS or CUDA is available
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("mps is available, running on mps.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            logger.info("CUDA is not available, running on CPU.")

        # Shared dict for worker to report LLM param memory (see /v1/debug/memory).
        self._worker_memory_info = {}
        # Load the TTS models
        self.load_llama_model(
            llama_checkpoint_path, self.device, self.precision, self.compile, self.mode
        )
        self.load_decoder_model(
            decoder_config_name, decoder_checkpoint_path, self.device
        )
        self.tts_inference_engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=self.compile,
        )

        # Optional: record CUDA alloc/free history for debugging (dump via GET /v1/debug/memory?dump=1).
        if torch.cuda.is_available() and os.environ.get(
            "FISH_RECORD_MEMORY_HISTORY", ""
        ).strip() in ("1", "true", "True"):
            try:
                max_entries = int(
                    os.environ.get("FISH_MEMORY_HISTORY_MAX_ENTRIES", "100000")
                )
                torch.cuda.memory._record_memory_history(max_entries=max_entries)
                logger.info(
                    "CUDA memory history recording enabled (max_entries=%s); use ?dump=1 on /v1/debug/memory to save snapshot",
                    max_entries,
                )
            except Exception as e:
                logger.warning("Could not enable CUDA memory history recording: %s", e)

        # When --compile: run warmup before accepting requests so /v1/health "ready" = compiled.
        if self.mode == "tts" and self.compile:
            logger.warning(
                "torch.compile enabled — running warmup now (Inductor compiles kernels; typically 2–10 min). "
                "Server will accept connections after this."
            )
            self.warm_up(self.tts_inference_engine, compile=True)
            logger.warning("torch.compile warmup finished — server ready.")

    def load_llama_model(
        self, checkpoint_path, device, precision, compile, mode
    ) -> None:

        if mode == "tts":
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=checkpoint_path,
                device=device,
                precision=precision,
                compile=compile,
                memory_info=self._worker_memory_info,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        logger.info("LLAMA model loaded.")

    def load_decoder_model(self, config_name, checkpoint_path, device) -> None:
        self.decoder_model = load_decoder_model(
            config_name=config_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        logger.info("Decoder model loaded.")

    def warm_up(self, tts_inference_engine, *, compile: bool = False) -> None:
        # Compilation is per input shape: short prompt (warmup) compiles one graph; first
        # request with long prompt (e.g. reference) compiles another → second delay.
        # Use small max_new_tokens on warmup to reduce peak VRAM (fits ~32 GB with compile).
        if compile:
            logger.info(
                "Warmup: first inference (triggers torch.compile/Inductor; "
                "compiling kernels — can take 2–10+ min, then requests are fast)."
            )
        request = ServeTTSRequest(
            text="Hello world.",
            references=[],
            reference_id=None,
            max_new_tokens=64,  # was 1024; lower = less VRAM during compile, still triggers graph
            chunk_length=200,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            format="wav",
        )
        list(inference(request, tts_inference_engine))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # free cached VRAM so first request with reference fits in 32 GB
        logger.info("Warmup: first inference done.")

        # Streaming hits a different hot path than the oneshot warmup above:
        # prefill may be compiled already, but iterative token decode and bounded
        # streaming pipeline can still compile lazily on the first user request.
        if compile and _env_flag("FISH_WARMUP_STREAMING", True):
            stream_chunk_size = max(1, _env_int("FISH_STREAM_CHUNK_SIZE", 8))
            initial_stream_chunk_size = _env_optional_int(
                "FISH_INITIAL_STREAM_CHUNK_SIZE"
            )
            logger.info(
                "Warmup: streaming inference (compile iterative decode path) chunk_size={} initial_chunk_size={}",
                stream_chunk_size,
                initial_stream_chunk_size,
            )
            request_stream = ServeTTSRequest(
                text=(
                    "Hello world. This is a short streaming warmup request so the "
                    "iterative decode path is compiled before the first real client."
                ),
                references=[],
                reference_id=None,
                max_new_tokens=64,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.7,
                format="wav",
                streaming=True,
                stream_chunk_size=stream_chunk_size,
                initial_stream_chunk_size=initial_stream_chunk_size,
            )
            list(inference(request_stream, tts_inference_engine))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Warmup: streaming inference done.")

        # Optional: compile the long-prompt + streaming path so first real request is fast.
        # Set FISH_WARMUP_REFERENCE_ID to a valid reference id (e.g. en) to run one
        # streaming request with that reference at startup. Use ~50 word text so the
        # compiled graph covers typical range (Hello world .. 50 words) and avoids
        # recompile on first user request.
        warmup_ref = os.environ.get("FISH_WARMUP_REFERENCE_ID", "").strip()
        if compile and warmup_ref:
            # Text ~50 words so prompt length is in "typical long" range; one compile covers short..long.
            warmup_long_text = (
                "Hello, this is a longer warmup so the compiled graph covers prompts from a few words "
                "to about fifty. We run this once at startup to avoid recompiling on the first real request."
            )
            logger.info(
                "Warmup: running streaming inference with reference_id=%s (long prompt compile, ~50 words)",
                warmup_ref,
            )
            request_long = ServeTTSRequest(
                text=warmup_long_text,
                references=[],
                reference_id=warmup_ref,
                max_new_tokens=64,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.7,
                format="wav",
                streaming=True,
                stream_chunk_size=stream_chunk_size,
                initial_stream_chunk_size=initial_stream_chunk_size,
            )
            list(inference(request_long, tts_inference_engine))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Warmup (long prompt + streaming) done.")

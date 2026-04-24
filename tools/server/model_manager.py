import torch
from loguru import logger

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.runtime_config import load_runtime_config
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference


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
        self.runtime = load_runtime_config()

        self.precision = torch.half if half else torch.bfloat16

        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("mps is available, running on mps.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            logger.info("CUDA is not available, running on CPU.")

        self._worker_memory_info = {}

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

        if torch.cuda.is_available() and self.runtime.model.record_memory_history:
            try:
                torch.cuda.memory._record_memory_history(
                    max_entries=self.runtime.model.memory_history_max_entries
                )
                logger.info(
                    "CUDA memory history recording enabled (max_entries=%s)",
                    self.runtime.model.memory_history_max_entries,
                )
            except Exception as e:
                logger.warning("Could not enable CUDA memory history recording: %s", e)

        if self.mode == "tts" and self.compile and self.runtime.warmup.enabled:
            logger.warning(
                "torch.compile enabled — running warmup now. "
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
        warmup = self.runtime.warmup
        reference_id = (
            warmup.reference_id
            or self.runtime.proxy.tts.reference_id
            or self.runtime.proxy.default_reference_id
        )

        if compile:
            logger.info("Warmup: first inference (compile path).")

        request = ServeTTSRequest(
            text=warmup.text,
            references=[],
            reference_id=reference_id if reference_id else None,
            max_new_tokens=warmup.max_new_tokens,
            chunk_length=warmup.chunk_length,
            top_p=self.runtime.proxy.tts.top_p,
            repetition_penalty=self.runtime.proxy.tts.repetition_penalty,
            temperature=self.runtime.proxy.tts.temperature,
            format="wav",
            streaming=warmup.streaming,
            stream_tokens=warmup.stream_tokens,
            initial_stream_chunk_size=warmup.initial_stream_chunk_size,
            stream_chunk_size=warmup.stream_chunk_size,
            use_memory_cache=self.runtime.proxy.tts.use_memory_cache,
            seed=self.runtime.proxy.tts.seed,
        )
        list(inference(request, tts_inference_engine))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Warmup done.")
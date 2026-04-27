import gc
import queue
import threading
import time
from typing import Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.driver import DriverSynthesisRequest
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.generation.prompt_builder import GenerateResponse
from fish_speech.generation.worker import GenerateRequest
from fish_speech.references.loader import ReferenceLoader
from fish_speech.driver.config import load_runtime_config
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.codec.vq import VQManager


class TTSInferenceEngine(ReferenceLoader, VQManager):
    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
    ) -> None:
        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile
        self.runtime = load_runtime_config()

        model_cfg = self.runtime.model
        self.cleanup_after_request = model_cfg.cleanup_after_request
        self.cleanup_on_abort = model_cfg.cleanup_on_abort
        self.empty_cache_per_segment = model_cfg.empty_cache_per_segment
        self.cleanup_every_n_requests = model_cfg.cleanup_every_n_requests
        self._success_since_cleanup = 0
        self._cleanup_lock = threading.Lock()

    def _cuda_cleanup(self, *, reason: str) -> None:
        if not torch.cuda.is_available():
            return

        with self._cleanup_lock:
            before_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2)
            before_reserved = round(torch.cuda.memory_reserved() / (1024**3), 2)
            logger.info(
                "cuda cleanup start reason={} alloc_gb={} reserved_gb={}",
                reason,
                before_alloc,
                before_reserved,
            )

            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            after_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2)
            after_reserved = round(torch.cuda.memory_reserved() / (1024**3), 2)
            logger.info(
                "cuda cleanup done reason={} alloc_gb={} reserved_gb={}",
                reason,
                after_alloc,
                after_reserved,
            )
            self._success_since_cleanup = 0

    def _maybe_cleanup_after_success(self) -> None:
        if not torch.cuda.is_available():
            return

        self._success_since_cleanup += 1
        should_cleanup = self.cleanup_after_request
        if (
            not should_cleanup
            and self.cleanup_every_n_requests > 0
            and self._success_since_cleanup >= self.cleanup_every_n_requests
        ):
            should_cleanup = True

        if should_cleanup:
            self._cuda_cleanup(reason="success")

    def inference(
        self, req: DriverSynthesisRequest
    ) -> Generator[InferenceResult, None, None]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """

        profile = self.runtime.model.profile_inference
        req_tag = hex(id(req))[-6:]
        t_start = time.perf_counter()
        t_prev = t_start
        finished_normally = False

        def _vram_gb() -> dict:
            if not torch.cuda.is_available():
                return {}
            return {
                "vram_alloc_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                "vram_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
                "vram_max_alloc_gb": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 2
                ),
            }

        def _mark(event: str, **extra) -> None:
            nonlocal t_prev
            if not profile:
                return
            now = time.perf_counter()
            delta_ms = (now - t_prev) * 1000.0
            total_ms = (now - t_start) * 1000.0
            t_prev = now
            vram = _vram_gb()
            details = " ".join(f"{k}={v}" for k, v in (*vram.items(), *extra.items()))
            logger.info(
                "inference_timing req={} event={} delta_ms={:.1f} total_ms={:.1f} {}",
                req_tag,
                event,
                delta_ms,
                total_ms,
                details,
            )

        try:
            ref_id = req.reference_id
            prompt_tokens, prompt_texts = [], []

            ref_tokens, ref_texts = [], []
            history_tokens, history_texts = [], []

            # 1. Load main reference
            if ref_id is not None:
                ref_tokens, ref_texts = self.load_by_id(ref_id, req.use_memory_cache)
                _mark("ref_loaded", mode="id")
            elif req.references:
                ref_tokens, ref_texts = self.load_by_hash(
                    req.references, req.use_memory_cache
                )
                _mark("ref_loaded", mode="hash", refs=len(req.references))

            # 2. Load continuation history
            if req.continuation_tokens and req.continuation_text:
                history_tokens = list(req.continuation_tokens)
                history_texts = list(req.continuation_text)
                logger.info(
                    "continuation: added history after reference turns={} chars={}",
                    len(history_texts),
                    sum(len(t) for t in history_texts),
                )
            elif req.prompt_tokens and req.prompt_text:
                history_tokens = list(req.prompt_tokens)
                history_texts = list(req.prompt_text)
                logger.info(
                    "continuation: added legacy history after reference turns={} chars={}",
                    len(history_texts),
                    sum(len(t) for t in history_texts),
                )

            # Keep prompt_tokens/texts as combined for backward compatibility
            prompt_tokens = list(ref_tokens) + list(history_tokens)
            prompt_texts = list(ref_texts) + list(history_texts)

            if req.seed is not None:
                set_seed(req.seed)
                logger.warning(f"set seed: {req.seed}")

            stream_tokens = bool(req.stream_tokens)
            ack_queue = queue.Queue() if stream_tokens else None
            response_queue = self.send_Llama_request(
                req,
                prompt_tokens=ref_tokens,
                prompt_text=ref_texts,
                req_tag=req_tag,
                ack_queue=ack_queue,
                continuation_tokens=history_tokens,
                continuation_text=history_texts,
            )
            _mark("llama_queued")
            if stream_tokens:
                logger.info(
                    "stream: inference started (token streaming), req={}", req_tag
                )

            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate

            segments = []
            seg_idx = 0

            while True:
                if stream_tokens:
                    logger.info(
                        "stream: waiting for next chunk from queue, req={}", req_tag
                    )
                wrapped_result = response_queue.get()
                _mark("queue_get")
                if stream_tokens:
                    action = (
                        getattr(wrapped_result.response, "action", None)
                        if hasattr(wrapped_result.response, "action")
                        else None
                    )
                    logger.info(
                        "stream: queue_get status={} action={} req={}",
                        wrapped_result.status,
                        action,
                        req_tag,
                    )

                if wrapped_result.status == "error":
                    logger.error(
                        "stream: got error from worker req={} err={}",
                        req_tag,
                        wrapped_result.response,
                    )
                    _mark("yield_error")
                    yield InferenceResult(
                        code="error",
                        audio=None,
                        error=(
                            wrapped_result.response
                            if isinstance(wrapped_result.response, Exception)
                            else Exception("Unknown error")
                        ),
                    )
                    break

                if not isinstance(wrapped_result.response, GenerateResponse):
                    raise TypeError(
                        f"Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                    )

                result = wrapped_result.response
                if result.action != "next":
                    if req.stream_tokens and result.codes is not None:
                        yield InferenceResult(
                            code="tokens",
                            audio=None,
                            error=None,
                            tokens=(
                                result.codes.cpu()
                                if getattr(result.codes, "is_cuda", False)
                                else result.codes
                            ),
                        )
                    if stream_tokens:
                        logger.info(
                            "stream: decoding segment seg_idx={} codes_shape={} req={}",
                            seg_idx + 1,
                            result.codes.shape if result.codes is not None else None,
                            req_tag,
                        )
                    _mark(
                        "decode_vq_start",
                        segment_idx=seg_idx + 1,
                        codes_frames=(
                            result.codes.shape[1] if result.codes is not None else 0
                        ),
                    )
                    try:
                        if result.codes is not None and not result.codes.is_cuda:
                            result = GenerateResponse(
                                result.action,
                                result.codes.to(self.decoder_model.device),
                                getattr(result, "text", None),
                            )
                        segment = self.get_audio_segment(result)
                    except Exception as seg_err:
                        if stream_tokens:
                            logger.exception(
                                "stream: get_audio_segment FAILED seg_idx={} codes_shape={} req={}: {}",
                                seg_idx + 1,
                                (
                                    result.codes.shape
                                    if result.codes is not None
                                    else None
                                ),
                                req_tag,
                                seg_err,
                            )
                        raise

                    seg_idx += 1
                    _mark("segment_decoded", segment_idx=seg_idx, samples=len(segment))
                    if stream_tokens:
                        logger.info(
                            "stream: segment_decoded seg_idx={} samples={} req={}",
                            seg_idx,
                            len(segment),
                            req_tag,
                        )

                    if req.stream_audio:
                        _mark("yield_segment", segment_idx=seg_idx)
                        yield InferenceResult(
                            code="segment",
                            audio=(sample_rate, segment),
                            error=None,
                        )
                    if ack_queue is not None:
                        ack_queue.put(None)
                    segments.append(segment)
                    if self.empty_cache_per_segment and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    if stream_tokens:
                        logger.info(
                            "stream: got_next (end of stream), total_segments={} req={}",
                            seg_idx,
                            req_tag,
                        )
                    if ack_queue is not None:
                        ack_queue.put(None)
                    _mark("got_next")
                    break

            if len(segments) == 0:
                _mark("yield_error_empty")
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=RuntimeError(
                        "No audio generated, please check the input text."
                    ),
                )
            else:
                audio = np.concatenate(segments, axis=0)
                _mark("yield_final", total_samples=len(audio), segments=len(segments))
                yield InferenceResult(
                    code="final",
                    audio=(sample_rate, audio),
                    error=None,
                )

            finished_normally = True
            self._maybe_cleanup_after_success()
            return None
        finally:
            if not finished_normally and self.cleanup_on_abort:
                self._cuda_cleanup(reason="abort")

    def send_Llama_request(
        self,
        req: DriverSynthesisRequest,
        prompt_tokens: list,
        prompt_texts: list,
        req_tag: str,
        ack_queue: queue.Queue | None = None,
        continuation_tokens: list | None = None,
        continuation_text: list | None = None,
    ) -> queue.Queue:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """

        stream_tokens = bool(req.stream_tokens)
        request = dict(
            device=self.decoder_model.device,
            req_tag=req_tag,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            segments=req.committed_segments(),
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
            continuation_tokens=continuation_tokens,
            continuation_text=continuation_text,
            stream_tokens=stream_tokens,
            stream_chunk_size=req.stream_chunk_size,
            initial_stream_chunk_size=req.initial_stream_chunk_size,
            ack_queue=ack_queue,
        )

        response_queue = queue.Queue()
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )
        return response_queue

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        """
        Decode the VQ tokens to audio.
        """

        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            segment = self.decode_vq_tokens(codes=result.codes)

        return segment.float().detach().cpu().numpy()

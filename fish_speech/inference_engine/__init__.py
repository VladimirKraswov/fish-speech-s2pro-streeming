import gc
import queue
import threading
import time
from typing import Any, Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.driver import DriverSynthesisRequest
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.codec.codes import (
    estimate_code_frames,
    validate_codes_for_decoder,
    validate_codes_for_decoder_device,
)
from fish_speech.generation.prompt_builder import GenerateResponse
from fish_speech.generation.worker import GenerateRequest
from fish_speech.references.loader import ReferenceLoader
from fish_speech.driver.config import load_runtime_config
from fish_speech.utils import set_seed
from fish_speech.codec.vq import VQManager


def _as_list(value: Any | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _validate_text_turns(values: list[Any], *, name: str) -> list[str]:
    out: list[str] = []
    for idx, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}[{idx}] must be a non-empty string")
        out.append(value)
    return out


class TTSInferenceEngine(ReferenceLoader, VQManager):
    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
        llama_device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile
        self.runtime = load_runtime_config()
        self.llama_device = (
            torch.device(llama_device)
            if llama_device is not None
            else self._get_decoder_device()
        )

        model_cfg = self.runtime.model
        self.cleanup_after_request = model_cfg.cleanup_after_request
        self.cleanup_on_abort = model_cfg.cleanup_on_abort
        self.empty_cache_per_segment = model_cfg.empty_cache_per_segment
        self.cleanup_every_n_requests = model_cfg.cleanup_every_n_requests
        self._success_since_cleanup = 0
        self._cleanup_lock = threading.Lock()
        self._decoder_lock = threading.Lock()

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
        first_codes_at: float | None = None
        first_decode_start_at: float | None = None
        first_decode_end_at: float | None = None
        finished_normally = False
        cancel_event: threading.Event | None = None

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
            ref_tokens, ref_texts = [], []
            history_tokens, history_texts = [], []

            request_prompt_tokens = _as_list(req.prompt_tokens)
            request_prompt_texts = _as_list(req.prompt_text)
            request_continuation_tokens = _as_list(req.continuation_tokens)
            request_continuation_texts = _as_list(req.continuation_text)

            # 1. Load main reference
            if ref_id is not None:
                ref_tokens, ref_texts = self.load_by_id(ref_id, req.use_memory_cache)
                _mark("ref_loaded", mode="id")
            elif req.references:
                ref_tokens, ref_texts = self.load_by_hash(
                    req.references, req.use_memory_cache
                )
                _mark("ref_loaded", mode="hash", refs=len(req.references))
            elif request_prompt_tokens or request_prompt_texts:
                if not (request_prompt_tokens and request_prompt_texts):
                    raise ValueError(
                        "Reference prompt is incomplete: prompt_text and prompt_tokens must be provided together"
                    )
                if len(request_prompt_tokens) != len(request_prompt_texts):
                    raise ValueError(
                        "Prompt text and tokens must have the same length: "
                        f"texts={len(request_prompt_texts)} tokens={len(request_prompt_tokens)}"
                    )

                ref_texts = _validate_text_turns(
                    request_prompt_texts,
                    name="prompt_text",
                )
                ref_tokens = [
                    validate_codes_for_decoder(
                        t, self.decoder_model, name=f"prompt[{i}]"
                    )
                    for i, t in enumerate(request_prompt_tokens)
                ]
                logger.info(
                    "reference: using request prompt turns={} chars={} frames={}",
                    len(ref_texts),
                    sum(len(t) for t in ref_texts),
                    sum(int(t.shape[-1]) for t in ref_tokens if hasattr(t, "shape")),
                )
                _mark("ref_loaded", mode="prompt", refs=len(ref_texts))

            # 2. Load continuation history
            if request_continuation_tokens or request_continuation_texts:
                if not (request_continuation_tokens and request_continuation_texts):
                    raise ValueError(
                        "Continuation is incomplete: continuation_text and continuation_tokens must be provided together"
                    )
                if len(request_continuation_tokens) != len(request_continuation_texts):
                    raise ValueError(
                        "Continuation text and tokens must have the same length: "
                        f"texts={len(request_continuation_texts)} tokens={len(request_continuation_tokens)}"
                    )

                history_texts = _validate_text_turns(
                    request_continuation_texts,
                    name="continuation_text",
                )
                history_tokens = [
                    validate_codes_for_decoder(
                        t, self.decoder_model, name=f"continuation[{i}]"
                    )
                    for i, t in enumerate(request_continuation_tokens)
                ]
                logger.info(
                    "continuation: added history after reference turns={} chars={} frames={}",
                    len(history_texts),
                    sum(len(t) for t in history_texts),
                    sum(int(t.shape[-1]) for t in history_tokens if hasattr(t, "shape")),
                )

            if req.seed is not None:
                set_seed(req.seed)
                logger.warning(f"set seed: {req.seed}")

            stream_decode = bool(req.stream_tokens or req.stream_audio)
            emit_token_events = bool(req.stream_tokens)
            collect_codes = not req.stream_audio
            ack_queue = queue.Queue() if stream_decode else None
            cancel_event = threading.Event() if stream_decode else None
            response_queue = self.send_Llama_request(
                req,
                prompt_tokens=ref_tokens,
                prompt_texts=ref_texts,
                req_tag=req_tag,
                ack_queue=ack_queue,
                cancel_event=cancel_event,
                continuation_tokens=history_tokens,
                continuation_text=history_texts,
            )
            _mark("llama_queued")
            if stream_decode and profile:
                logger.info(
                    "stream: inference started (token streaming), req={}", req_tag
                )

            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate

            segments = []
            all_codes = []
            seg_idx = 0
            stream_decode_idx = 0
            max_vq_context_frames = 8 if req.low_latency_first_audio else 32

            vq_left_context = None
            if req.stream_audio and history_tokens:
                last_history = history_tokens[-1]
                if last_history is not None and last_history.shape[-1] > 0:
                    vq_left_context = (
                        last_history[:, -max_vq_context_frames:]
                        .detach()
                        .cpu()
                        .contiguous()
                    )
                    if profile:
                        logger.info(
                            "stream: initialized decoder left context from continuation frames={}",
                            vq_left_context.shape[-1],
                        )

            while True:
                if stream_decode and profile:
                    logger.info(
                        "stream: waiting for next chunk from queue, req={}", req_tag
                    )
                wrapped_result = response_queue.get()
                _mark("queue_get")
                if stream_decode and profile:
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
                    return

                needs_ack = ack_queue is not None
                try:
                    if not isinstance(wrapped_result.response, GenerateResponse):
                        raise TypeError(
                            f"Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                        )

                    result = wrapped_result.response
                    if result.action != "next":
                        if result.codes is not None:
                            if collect_codes:
                                all_codes.append(result.codes)

                            if emit_token_events:
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

                        if req.stream_audio and profile:
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
                            if req.stream_audio and result.codes is not None:
                                new_codes = result.codes
                                if first_codes_at is None:
                                    first_codes_at = time.perf_counter()
                                    _mark(
                                        "first_token_chunk_from_worker",
                                        tt_first_codes_ms=round(
                                            (first_codes_at - t_start) * 1000.0, 1
                                        ),
                                        first_chunk_frames=int(new_codes.shape[-1]),
                                        codes_device_before_decode=str(new_codes.device),
                                    )
                                context_frames_before = (
                                    vq_left_context.shape[-1]
                                    if vq_left_context is not None
                                    else 0
                                )

                                if first_decode_start_at is None:
                                    first_decode_start_at = time.perf_counter()
                                    _mark(
                                        "first_decode_start",
                                        codes_device_before_decode=str(new_codes.device),
                                        decoder_device=str(self._get_decoder_device()),
                                    )

                                segment, vq_left_context, full_decoded_size = (
                                    self._decode_stream_codes_with_context(
                                        new_codes=new_codes,
                                        left_context_codes=vq_left_context,
                                        max_context_frames=max_vq_context_frames,
                                    )
                                )

                                if first_decode_end_at is None:
                                    first_decode_end_at = time.perf_counter()
                                    _mark(
                                        "first_decode_end",
                                        tt_first_decode_ms=round(
                                            (
                                                first_decode_end_at
                                                - (first_decode_start_at or first_decode_end_at)
                                            )
                                            * 1000.0,
                                            1,
                                        ),
                                        first_chunk_samples=len(segment),
                                    )

                                if profile and stream_decode_idx < 3:
                                    logger.info(
                                        "stream_vq_context_decode: new_frames={} context_frames={} decode_frames={} decoded_samples={} emitted_samples={}",
                                        new_codes.shape[-1],
                                        context_frames_before,
                                        new_codes.shape[-1] + context_frames_before,
                                        full_decoded_size,
                                        len(segment),
                                    )
                                    stream_decode_idx += 1
                            elif req.stream_audio:
                                segment = self.get_audio_segment(result)
                            else:
                                segment = None
                        except Exception as seg_err:
                            if stream_decode:
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

                        if segment is not None:
                            seg_idx += 1
                            _mark(
                                "segment_decoded",
                                segment_idx=seg_idx,
                                samples=len(segment),
                            )
                            if req.stream_audio and profile:
                                logger.info(
                                    "stream: segment_decoded seg_idx={} samples={} req={}",
                                    seg_idx,
                                    len(segment),
                                    req_tag,
                                )

                            if req.stream_audio:
                                if segment.size > 0:
                                    if seg_idx == 1:
                                        now = time.perf_counter()
                                        _mark(
                                            "first_audio_yield",
                                            ttfa_ms=round((now - t_start) * 1000.0, 1),
                                            tt_first_codes_ms=(
                                                round(
                                                    (first_codes_at - t_start)
                                                    * 1000.0,
                                                    1,
                                                )
                                                if first_codes_at is not None
                                                else None
                                            ),
                                            tt_first_decode_ms=(
                                                round(
                                                    (
                                                        (first_decode_end_at or now)
                                                        - (
                                                            first_decode_start_at
                                                            or now
                                                        )
                                                    )
                                                    * 1000.0,
                                                    1,
                                                )
                                                if first_decode_start_at is not None
                                                else None
                                            ),
                                            first_chunk_samples=len(segment),
                                        )
                                    else:
                                        _mark("yield_segment", segment_idx=seg_idx)
                                    yield InferenceResult(
                                        code="segment",
                                        audio=(sample_rate, segment),
                                        error=None,
                                    )
                                    segments.append(segment)
                            else:
                                segments.append(segment)

                        if self.empty_cache_per_segment and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        if stream_decode and profile:
                            logger.info(
                                "stream: got_next (end of stream), total_segments={} req={}",
                                seg_idx,
                                req_tag,
                            )

                        _mark("got_next")
                        break
                finally:
                    if needs_ack:
                        ack_queue.put(None)

            if not segments and not all_codes:
                _mark("yield_error_empty")
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=RuntimeError(
                        "No audio generated, please check the input text."
                    ),
                )
            elif req.stream_audio:
                _mark("stream_done", total_segments=len(segments))
            elif not req.stream_audio:
                if all_codes:
                    full_codes = torch.cat(all_codes, dim=1)
                    audio = self.get_audio_segment(
                        GenerateResponse(action="sample", codes=full_codes)
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
            if cancel_event is not None:
                cancel_event.set()
            if not finished_normally and self.cleanup_on_abort:
                self._cuda_cleanup(reason="abort")

    def send_Llama_request(
        self,
        req: DriverSynthesisRequest,
        prompt_tokens: list,
        prompt_texts: list,
        req_tag: str,
        ack_queue: queue.Queue | None = None,
        cancel_event: threading.Event | None = None,
        continuation_tokens: list | None = None,
        continuation_text: list | None = None,
    ) -> queue.Queue:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """

        stream_tokens = bool(req.stream_tokens or req.stream_audio)
        request = dict(
            device=self.llama_device,
            req_tag=req_tag,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            segments=req.committed_segments(),
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0 and not req.low_latency_first_audio,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
            continuation_tokens=continuation_tokens,
            continuation_text=continuation_text,
            stream_audio=req.stream_audio,
            stream_tokens=stream_tokens,
            stream_chunk_size=req.stream_chunk_size,
            initial_stream_chunk_size=req.initial_stream_chunk_size,
            low_latency_first_audio=req.low_latency_first_audio,
            ack_queue=ack_queue,
            cancel_event=cancel_event,
        )

        response_queue = queue.Queue()
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )
        return response_queue

    def _decode_stream_codes_with_context(
        self,
        *,
        new_codes: torch.Tensor,
        left_context_codes: torch.Tensor | None,
        max_context_frames: int,
    ) -> tuple[np.ndarray, torch.Tensor | None, int]:
        """
        Decodes new VQ codes by prepending a small context from previous codes
        to ensure acoustic continuity. Returns the new audio segment, updated
        context for the next chunk, and the full size of the decoded buffer.
        """
        new_codes_fast = new_codes.detach()
        if new_codes_fast.dtype != torch.long:
            new_codes_fast = new_codes_fast.to(dtype=torch.long)

        if left_context_codes is not None and left_context_codes.numel() > 0:
            if left_context_codes.device != new_codes_fast.device:
                left_context_codes = left_context_codes.to(
                    device=new_codes_fast.device,
                    non_blocking=True,
                )
            if left_context_codes.dtype != torch.long:
                left_context_codes = left_context_codes.to(dtype=torch.long)
            decode_codes = torch.cat([left_context_codes, new_codes_fast], dim=1)
            context_frames = left_context_codes.shape[-1]
        else:
            decode_codes = new_codes_fast
            context_frames = 0

        decoded = self.get_audio_segment_fast(
            GenerateResponse(action="sample", codes=decode_codes),
            validate_debug=self.runtime.model.profile_inference,
        )

        # DAC uses a specific hop length / frame length for its VQ codebooks.
        # We must align the cropping precisely to this boundary.
        frame_samples = int(getattr(self.decoder_model, "frame_length", 512))
        context_samples = context_frames * frame_samples

        if decoded.size > context_samples:
            decoded_new = decoded[context_samples:]
        else:
            # If the decoded segment is smaller than context_samples, it means
            # the decoder didn't produce enough samples or the frame_length is wrong.
            # We return empty segment to avoid glitches.
            decoded_new = np.zeros(0, dtype=np.float32)

        if max_context_frames > 0:
            # Take the end of current decoded codes as the context for the next segment.
            next_context = decode_codes[:, -max_context_frames:].detach().contiguous()
        else:
            next_context = None

        return (
            decoded_new.astype(np.float32, copy=False),
            next_context,
            decoded.size,
        )

    def get_audio_segment_fast(
        self,
        result: GenerateResponse,
        *,
        validate_debug: bool = False,
    ) -> np.ndarray:
        """
        Decode generated VQ tokens without normalizing through CPU.

        The normal get_audio_segment path is intentionally conservative for
        external/cache inputs. Streaming generated chunks are already tensors
        from the model worker, so this path keeps codes on GPU until the final
        PCM numpy conversion.
        """

        codes = result.codes
        if codes is None or estimate_code_frames(codes) == 0:
            return np.zeros(0, dtype=np.float32)
        if not isinstance(codes, torch.Tensor):
            return self.get_audio_segment(result)

        codes = codes.detach()
        if codes.ndim == 3:
            if codes.shape[0] != 1:
                raise ValueError(
                    f"generated codes have unexpected shape {tuple(codes.shape)}"
                )
            codes = codes[0]
        if codes.ndim != 2:
            raise ValueError(
                f"generated codes must be [C, T], got {tuple(codes.shape)}"
            )
        if codes.dtype != torch.long:
            codes = codes.to(dtype=torch.long)

        decoder_device = self._get_decoder_device()
        if codes.device != decoder_device:
            codes = codes.to(device=decoder_device, non_blocking=True)

        with self._decoder_lock:
            if validate_debug:
                codes = validate_codes_for_decoder_device(
                    codes,
                    self.decoder_model,
                    name="generated streaming codes",
                )

            with torch.inference_mode():
                if isinstance(self.decoder_model, DAC):
                    segment = self.decoder_model.from_indices(codes[None])[0].squeeze()
                else:
                    segment = self.decode_vq_tokens(codes=codes)

        return segment.float().detach().cpu().numpy().astype(np.float32, copy=False)

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        """
        Decode the VQ tokens to audio.
        """
        if result.codes is None or estimate_code_frames(result.codes) == 0:
            return np.zeros(0, dtype=np.float32)

        with self._decoder_lock:
            # Standardize and validate codes
            codes = validate_codes_for_decoder(
                result.codes, self.decoder_model, name="generated codes"
            )

            decoder_device = self._get_decoder_device()
            if codes.device != decoder_device:
                codes = codes.to(decoder_device)

            with torch.inference_mode():
                segment = self.decode_vq_tokens(codes=codes)

        return segment.float().detach().cpu().numpy().astype(np.float32, copy=False)

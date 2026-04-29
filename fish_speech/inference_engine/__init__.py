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
from fish_speech.utils import set_seed
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

            # 1. Load main reference
            if ref_id is not None:
                ref_tokens, ref_texts = self.load_by_id(ref_id, req.use_memory_cache)
                _mark("ref_loaded", mode="id")
            elif req.references:
                ref_tokens, ref_texts = self.load_by_hash(
                    req.references, req.use_memory_cache
                )
                _mark("ref_loaded", mode="hash", refs=len(req.references))
            elif req.prompt_tokens and req.prompt_text:
                ref_tokens = list(req.prompt_tokens)
                ref_texts = list(req.prompt_text)
                logger.info(
                    "reference: using request prompt turns={} chars={} frames={}",
                    len(ref_texts),
                    sum(len(t) for t in ref_texts),
                    sum(int(t.shape[-1]) for t in ref_tokens if hasattr(t, "shape")),
                )
                _mark("ref_loaded", mode="prompt", refs=len(ref_texts))

            # 2. Load continuation history
            if req.continuation_tokens and req.continuation_text:
                history_tokens = list(req.continuation_tokens)
                history_texts = list(req.continuation_text)
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
            if stream_decode:
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
            vq_left_context = None
            pending_pcm_tail = None
            stream_decode_idx = 0
            max_vq_context_frames = 32

            while True:
                if stream_decode:
                    logger.info(
                        "stream: waiting for next chunk from queue, req={}", req_tag
                    )
                wrapped_result = response_queue.get()
                _mark("queue_get")
                if stream_decode:
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

                        if req.stream_audio:
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
                                context_frames_before = (
                                    vq_left_context.shape[-1]
                                    if vq_left_context is not None
                                    else 0
                                )

                                segment, vq_left_context, full_decoded_size = (
                                    self._decode_stream_codes_with_context(
                                        new_codes=new_codes,
                                        left_context_codes=vq_left_context,
                                        max_context_frames=max_vq_context_frames,
                                    )
                                )

                                if stream_decode_idx < 3:
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
                            if req.stream_audio:
                                logger.info(
                                    "stream: segment_decoded seg_idx={} samples={} req={}",
                                    seg_idx,
                                    len(segment),
                                    req_tag,
                                )

                            if req.stream_audio:
                                processed_segment, pending_pcm_tail = (
                                    self._crossfade_stream_segment(
                                        segment=segment,
                                        pending_tail=pending_pcm_tail,
                                        sample_rate=sample_rate,
                                        fade_ms=8.0,
                                    )
                                )

                                if processed_segment.size > 0:
                                    _mark("yield_segment", segment_idx=seg_idx)
                                    yield InferenceResult(
                                        code="segment",
                                        audio=(sample_rate, processed_segment),
                                        error=None,
                                    )
                                    segments.append(processed_segment)
                            else:
                                segments.append(segment)

                        if self.empty_cache_per_segment and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        if stream_decode:
                            logger.info(
                                "stream: got_next (end of stream), total_segments={} req={}",
                                seg_idx,
                                req_tag,
                            )

                        if req.stream_audio and pending_pcm_tail is not None:
                            _mark("yield_final_tail")
                            tail = self._flush_stream_tail(pending_pcm_tail)
                            if tail is not None:
                                yield InferenceResult(
                                    code="segment",
                                    audio=(sample_rate, tail),
                                    error=None,
                                )
                                segments.append(tail)
                            pending_pcm_tail = None

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
        new_codes_cpu = new_codes.detach().cpu()

        if left_context_codes is not None and left_context_codes.numel() > 0:
            decode_codes_cpu = torch.cat([left_context_codes, new_codes_cpu], dim=1)
            context_frames = left_context_codes.shape[-1]
        else:
            decode_codes_cpu = new_codes_cpu
            context_frames = 0

        decoded = self.get_audio_segment(
            GenerateResponse(action="sample", codes=decode_codes_cpu)
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
            next_context = decode_codes_cpu[:, -max_context_frames:].contiguous()
        else:
            next_context = None

        return (
            decoded_new.astype(np.float32, copy=False),
            next_context,
            decoded.size,
        )

    def _flush_stream_tail(self, pending_tail: np.ndarray | None) -> np.ndarray | None:
        return pending_tail

    def _crossfade_stream_segment(
        self,
        *,
        segment: np.ndarray,
        pending_tail: np.ndarray | None,
        sample_rate: int,
        fade_ms: float = 8.0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        segment = np.asarray(segment, dtype=np.float32)
        if segment.size == 0:
            return segment, pending_tail

        fade_samples = max(1, int(sample_rate * fade_ms / 1000.0))

        if pending_tail is None:
            if segment.size <= fade_samples:
                return np.zeros(0, dtype=np.float32), segment.copy()
            return segment[:-fade_samples].copy(), segment[-fade_samples:].copy()

        pending_tail = np.asarray(pending_tail, dtype=np.float32)

        if segment.size <= fade_samples:
            combined = np.concatenate([pending_tail, segment])
            if combined.size <= fade_samples:
                return np.zeros(0, dtype=np.float32), combined.copy()
            # If still smaller than fade_samples but combined > fade_samples, we proceed
            segment = combined
            pending_tail = None

        if pending_tail is None:
            return segment[:-fade_samples].copy(), segment[-fade_samples:].copy()

        crossfade_len = min(pending_tail.size, segment.size, fade_samples)
        old_keep = pending_tail[:-crossfade_len]
        old_tail = pending_tail[-crossfade_len:]
        new_head = segment[:crossfade_len]

        fade_out = np.linspace(1.0, 0.0, crossfade_len, endpoint=False, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, crossfade_len, endpoint=False, dtype=np.float32)

        blended = old_tail * fade_out + new_head * fade_in

        body_end = max(crossfade_len, segment.size - fade_samples)
        body = segment[crossfade_len:body_end]
        new_tail = segment[-fade_samples:].copy()

        output = np.concatenate([old_keep, blended, body]).astype(np.float32, copy=False)
        output = np.clip(output, -1.0, 1.0)

        return output, new_tail

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        """
        Decode the VQ tokens to audio.
        """

        if result.codes is None:
            return np.zeros(0, dtype=np.float32)

        codes = result.codes
        decoder_device = getattr(self.decoder_model, "device", None)
        if decoder_device is None:
            try:
                decoder_device = next(self.decoder_model.parameters()).device
            except (AttributeError, StopIteration):
                decoder_device = codes.device

        if codes.device != decoder_device:
            codes = codes.to(decoder_device)

        with torch.inference_mode():
            segment = self.decode_vq_tokens(codes=codes)

        return segment.float().detach().cpu().numpy().astype(np.float32, copy=False)

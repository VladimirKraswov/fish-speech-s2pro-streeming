import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Generator, Literal

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import DACDecodeStats, VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import ServeTTSRequest

ACK_CONTINUE = "continue"
ACK_ABORT = "abort"
QUEUE_POLL_TIMEOUT_SEC = 0.1
QUEUE_BLOCK_LOG_INTERVAL_SEC = 0.5
SENDER_WAIT_LOG_INTERVAL_SEC = 1.0
THREAD_JOIN_TIMEOUT_SEC = 2.0


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


@dataclass
class PipelineConfig:
    codes_queue_maxsize: int
    audio_queue_maxsize: int
    max_ahead_chunks: int
    response_queue_maxsize: int


@dataclass
class CodeChunk:
    sequence_id: int
    codes: torch.Tensor
    text: str | None = None


@dataclass
class AudioChunk:
    sequence_id: int
    audio: np.ndarray
    text: str | None = None
    decode_ms: float = 0.0
    microchunk_size: int = 1


@dataclass
class PipelineSignal:
    kind: Literal["eos", "error", "cancel"]
    error: Exception | None = None
    reason: str | None = None


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

        optimize_for_inference = getattr(
            self.decoder_model, "optimize_for_inference", None
        )
        if callable(optimize_for_inference):
            try:
                removed = optimize_for_inference(remove_weight_norm=True)
                logger.info(
                    "DAC inference optimization enabled: removed_weight_norm_layers={}",
                    removed,
                )
            except Exception as opt_err:
                logger.warning("DAC inference optimization skipped: {}", opt_err)

    def _sample_rate(self) -> int:
        if hasattr(self.decoder_model, "spec_transform"):
            return self.decoder_model.spec_transform.sample_rate
        return self.decoder_model.sample_rate

    def _build_pipeline_config(self) -> PipelineConfig:
        codes_queue_maxsize = max(1, _env_int("FISH_CODES_QUEUE_MAXSIZE", 4))
        audio_queue_maxsize = max(1, _env_int("FISH_AUDIO_QUEUE_MAXSIZE", 4))
        max_ahead_chunks = max(1, _env_int("FISH_STREAM_MAX_AHEAD_CHUNKS", 3))
        max_ahead_chunks = min(max_ahead_chunks, codes_queue_maxsize)

        return PipelineConfig(
            codes_queue_maxsize=codes_queue_maxsize,
            audio_queue_maxsize=audio_queue_maxsize,
            max_ahead_chunks=max_ahead_chunks,
            response_queue_maxsize=max(1, min(2, max_ahead_chunks)),
        )

    def _safe_ack(
        self,
        ack_queue: queue.Queue | None,
        ack: str,
        *,
        req_tag: str,
        reason: str,
    ) -> None:
        if ack_queue is None:
            return

        try:
            ack_queue.put_nowait(ack)
        except queue.Full:
            logger.warning(
                "pipeline: req={} failed to send ack={} reason={}",
                req_tag,
                ack,
                reason,
            )

    def _put_queue_item(
        self,
        target_queue: queue.Queue,
        item,
        *,
        cancel_event: threading.Event,
        req_tag: str,
        stage: str,
        blocked_message: str,
        success_message: str,
        extra_log: str = "",
    ) -> bool:
        blocked_at: float | None = None

        while not cancel_event.is_set():
            try:
                target_queue.put(item, timeout=QUEUE_POLL_TIMEOUT_SEC)
                if blocked_at is not None:
                    logger.info(
                        "{} req={} blocked_ms={:.1f} depth={} {}",
                        success_message,
                        req_tag,
                        (time.perf_counter() - blocked_at) * 1000.0,
                        target_queue.qsize(),
                        extra_log,
                    )
                return True
            except queue.Full:
                if blocked_at is None:
                    blocked_at = time.perf_counter()
                    logger.warning(
                        "{} req={} depth={} {}",
                        blocked_message,
                        req_tag,
                        target_queue.qsize(),
                        extra_log,
                    )
                elif (
                    time.perf_counter() - blocked_at
                ) >= QUEUE_BLOCK_LOG_INTERVAL_SEC:
                    logger.info(
                        "{} req={} depth={} {}",
                        blocked_message,
                        req_tag,
                        target_queue.qsize(),
                        extra_log,
                    )
                    blocked_at = time.perf_counter()

        logger.info(
            "pipeline: req={} stage={} stop_put reason=cancelled depth={}",
            req_tag,
            stage,
            target_queue.qsize(),
        )
        return False

    def _put_terminal_signal(
        self,
        target_queue: queue.Queue,
        signal: PipelineSignal,
        *,
        req_tag: str,
        stage: str,
        timeout_sec: float = 2.0,
    ) -> bool:
        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline:
            try:
                target_queue.put(signal, timeout=QUEUE_POLL_TIMEOUT_SEC)
                return True
            except queue.Full:
                continue

        logger.warning(
            "pipeline: failed to forward terminal signal req={} stage={} kind={} depth={}",
            req_tag,
            stage,
            signal.kind,
            target_queue.qsize(),
        )
        return False

    def _wait_for_producer_window(
        self,
        *,
        codes_queue: queue.Queue,
        cancel_event: threading.Event,
        memory_pressure_event: threading.Event,
        config: PipelineConfig,
        req_tag: str,
    ) -> bool:
        blocked_at: float | None = None

        while not cancel_event.is_set():
            ahead_limit = 1 if memory_pressure_event.is_set() else config.max_ahead_chunks
            depth = codes_queue.qsize()
            if depth < ahead_limit:
                if blocked_at is not None:
                    logger.info(
                        "producer: unblocked req={} blocked_ms={:.1f} codes_depth={} ahead_limit={}",
                        req_tag,
                        (time.perf_counter() - blocked_at) * 1000.0,
                        depth,
                        ahead_limit,
                    )
                return True

            if blocked_at is None:
                blocked_at = time.perf_counter()
                logger.warning(
                    "producer: blocked req={} codes_depth={} ahead_limit={}",
                    req_tag,
                    depth,
                    ahead_limit,
                )
            elif (
                time.perf_counter() - blocked_at
            ) >= QUEUE_BLOCK_LOG_INTERVAL_SEC:
                logger.info(
                    "producer: still_blocked req={} codes_depth={} ahead_limit={}",
                    req_tag,
                    depth,
                    ahead_limit,
                )
                blocked_at = time.perf_counter()

            time.sleep(QUEUE_POLL_TIMEOUT_SEC)

        return False

    def _producer_loop(
        self,
        *,
        req_tag: str,
        response_queue: queue.Queue,
        codes_queue: queue.Queue,
        ack_queue: queue.Queue,
        cancel_event: threading.Event,
        producer_done: threading.Event,
        memory_pressure_event: threading.Event,
        config: PipelineConfig,
    ) -> None:
        next_sequence_id = 0
        sent_terminal = False

        try:
            while not cancel_event.is_set():
                try:
                    wrapped_result = response_queue.get(timeout=QUEUE_POLL_TIMEOUT_SEC)
                except queue.Empty:
                    continue

                if wrapped_result.status == "error":
                    err = (
                        wrapped_result.response
                        if isinstance(wrapped_result.response, Exception)
                        else RuntimeError("Unknown LLM worker error")
                    )
                    logger.error("producer: worker_error req={} err={}", req_tag, err)
                    self._put_queue_item(
                        codes_queue,
                        PipelineSignal(kind="error", error=err, reason="llm_worker_error"),
                        cancel_event=cancel_event,
                        req_tag=req_tag,
                        stage="producer",
                        blocked_message="producer: blocked forwarding error",
                        success_message="producer: unblocked forwarding error",
                    )
                    sent_terminal = True
                    break

                if not isinstance(wrapped_result.response, GenerateResponse):
                    err = TypeError(
                        "Expected GenerateResponse, got "
                        f"{type(wrapped_result.response).__name__}"
                    )
                    logger.error("producer: malformed response req={} err={}", req_tag, err)
                    self._safe_ack(
                        ack_queue,
                        ACK_ABORT,
                        req_tag=req_tag,
                        reason="malformed_response",
                    )
                    self._put_queue_item(
                        codes_queue,
                        PipelineSignal(kind="error", error=err, reason="malformed_response"),
                        cancel_event=cancel_event,
                        req_tag=req_tag,
                        stage="producer",
                        blocked_message="producer: blocked forwarding malformed_response",
                        success_message="producer: unblocked forwarding malformed_response",
                    )
                    sent_terminal = True
                    break

                result = wrapped_result.response

                if result.action == "next":
                    logger.info(
                        "producer: eos req={} queued_segments={} codes_depth={}",
                        req_tag,
                        next_sequence_id,
                        codes_queue.qsize(),
                    )
                    if not self._put_queue_item(
                        codes_queue,
                        PipelineSignal(kind="eos", reason="llm_eos"),
                        cancel_event=cancel_event,
                        req_tag=req_tag,
                        stage="producer",
                        blocked_message="producer: blocked forwarding eos",
                        success_message="producer: unblocked forwarding eos",
                    ):
                        self._safe_ack(
                            ack_queue,
                            ACK_ABORT,
                            req_tag=req_tag,
                            reason="cancelled_while_forwarding_eos",
                        )
                        break

                    sent_terminal = True
                    break

                codes = result.codes
                if codes is None:
                    err = RuntimeError("Worker returned sample action without codes")
                    logger.error("producer: missing_codes req={}", req_tag)
                    self._safe_ack(
                        ack_queue,
                        ACK_ABORT,
                        req_tag=req_tag,
                        reason="missing_codes",
                    )
                    self._put_queue_item(
                        codes_queue,
                        PipelineSignal(kind="error", error=err, reason="missing_codes"),
                        cancel_event=cancel_event,
                        req_tag=req_tag,
                        stage="producer",
                        blocked_message="producer: blocked forwarding missing_codes",
                        success_message="producer: unblocked forwarding missing_codes",
                    )
                    sent_terminal = True
                    break

                codes_cpu = codes.detach().to(device="cpu", dtype=torch.long).contiguous()

                if not self._wait_for_producer_window(
                    codes_queue=codes_queue,
                    cancel_event=cancel_event,
                    memory_pressure_event=memory_pressure_event,
                    config=config,
                    req_tag=req_tag,
                ):
                    self._safe_ack(
                        ack_queue,
                        ACK_ABORT,
                        req_tag=req_tag,
                        reason="cancelled_while_waiting_for_window",
                    )
                    break

                queued = self._put_queue_item(
                    codes_queue,
                    CodeChunk(
                        sequence_id=next_sequence_id,
                        codes=codes_cpu,
                        text=result.text,
                    ),
                    cancel_event=cancel_event,
                    req_tag=req_tag,
                    stage="producer",
                    blocked_message="producer: blocked on codes_queue",
                    success_message="producer: unblocked on codes_queue",
                    extra_log=f"sequence_id={next_sequence_id}",
                )
                if not queued:
                    self._safe_ack(
                        ack_queue,
                        ACK_ABORT,
                        req_tag=req_tag,
                        reason="cancelled_while_putting_codes",
                    )
                    break

                logger.info(
                    "producer: queued req={} sequence_id={} codes_depth={} frames={}",
                    req_tag,
                    next_sequence_id,
                    codes_queue.qsize(),
                    int(codes_cpu.shape[1]),
                )
                self._safe_ack(
                    ack_queue,
                    ACK_CONTINUE,
                    req_tag=req_tag,
                    reason="codes_forwarded",
                )
                next_sequence_id += 1
        except Exception as err:
            logger.exception("producer: exception req={} err={}", req_tag, err)
            self._safe_ack(
                ack_queue,
                ACK_ABORT,
                req_tag=req_tag,
                reason="producer_exception",
            )
            self._put_queue_item(
                codes_queue,
                PipelineSignal(kind="error", error=err, reason="producer_exception"),
                cancel_event=cancel_event,
                req_tag=req_tag,
                stage="producer",
                blocked_message="producer: blocked forwarding exception",
                success_message="producer: unblocked forwarding exception",
            )
            sent_terminal = True
        finally:
            if cancel_event.is_set() and not sent_terminal:
                self._safe_ack(
                    ack_queue,
                    ACK_ABORT,
                    req_tag=req_tag,
                    reason="request_cancelled",
                )

            producer_done.set()
            logger.info(
                "producer: done req={} queued_segments={} codes_depth={}",
                req_tag,
                next_sequence_id,
                codes_queue.qsize(),
            )

    def _dac_consumer_loop(
        self,
        *,
        req_tag: str,
        codes_queue: queue.Queue,
        audio_queue: queue.Queue,
        cancel_event: threading.Event,
        producer_done: threading.Event,
        dac_done: threading.Event,
        memory_pressure_event: threading.Event,
    ) -> None:
        current_microchunk = max(1, _env_int("FISH_DAC_MAX_CODES_PER_STEP", 4))
        default_microchunk = current_microchunk
        min_free_gb = _env_float("FISH_DAC_MIN_FREE_GB", 3.0)
        resume_free_gb = _env_float("FISH_DAC_RESUME_FREE_GB", 5.0)
        critical_free_gb = _env_float("FISH_DAC_CRITICAL_FREE_GB", 1.5)
        pressure_active = False
        emitted_terminal = False

        try:
            while True:
                if cancel_event.is_set():
                    break

                try:
                    item = codes_queue.get(timeout=QUEUE_POLL_TIMEOUT_SEC)
                except queue.Empty:
                    continue

                if isinstance(item, PipelineSignal):
                    if item.kind != "cancel":
                        self._put_queue_item(
                            audio_queue,
                            item,
                            cancel_event=cancel_event,
                            req_tag=req_tag,
                            stage="dac",
                            blocked_message="dac: blocked forwarding terminal",
                            success_message="dac: unblocked forwarding terminal",
                            extra_log=f"kind={item.kind}",
                        )
                        emitted_terminal = True
                    else:
                        emitted_terminal = True
                    break

                if not isinstance(item, CodeChunk):
                    err = TypeError(
                        f"Unexpected codes_queue payload: {type(item).__name__}"
                    )
                    self._put_queue_item(
                        audio_queue,
                        PipelineSignal(kind="error", error=err, reason="invalid_codes_item"),
                        cancel_event=cancel_event,
                        req_tag=req_tag,
                        stage="dac",
                        blocked_message="dac: blocked forwarding invalid item",
                        success_message="dac: unblocked forwarding invalid item",
                    )
                    emitted_terminal = True
                    break

                snapshot = self.cuda_memory_snapshot(self._decoder_device())
                free_gb = snapshot.free_gb
                if free_gb is not None and free_gb < critical_free_gb:
                    if not pressure_active:
                        logger.warning(
                            "dac: memory_pressure_on req={} free_gb={:.2f} codes_depth={}",
                            req_tag,
                            free_gb,
                            codes_queue.qsize(),
                        )
                    pressure_active = True
                    memory_pressure_event.set()
                    current_microchunk = 1
                    self._light_cuda_cleanup(self._decoder_device())
                elif free_gb is not None and free_gb < min_free_gb:
                    if not pressure_active:
                        logger.warning(
                            "dac: memory_pressure_on req={} free_gb={:.2f} codes_depth={}",
                            req_tag,
                            free_gb,
                            codes_queue.qsize(),
                        )
                    pressure_active = True
                    memory_pressure_event.set()
                    current_microchunk = max(1, min(current_microchunk, default_microchunk // 2))
                elif free_gb is not None and free_gb >= resume_free_gb:
                    if pressure_active:
                        logger.info(
                            "dac: memory_pressure_off req={} free_gb={:.2f} codes_depth={}",
                            req_tag,
                            free_gb,
                            codes_queue.qsize(),
                        )
                    pressure_active = False
                    memory_pressure_event.clear()
                    current_microchunk = min(
                        default_microchunk,
                        max(1, current_microchunk * 2),
                    )

                try:
                    segment, stats = self.get_audio_segment(
                        item.codes,
                        req_tag=req_tag,
                        segment_idx=item.sequence_id + 1,
                        max_codes_per_step=current_microchunk,
                    )
                    current_microchunk = stats.microchunk_size
                except Exception as err:
                    logger.exception(
                        "dac: decode_failed req={} sequence_id={} frames={} err={}",
                        req_tag,
                        item.sequence_id,
                        int(item.codes.shape[1]),
                        err,
                    )
                    self._put_terminal_signal(
                        audio_queue,
                        PipelineSignal(kind="error", error=err, reason="dac_decode_failed"),
                        req_tag=req_tag,
                        stage="dac",
                    )
                    cancel_event.set()
                    emitted_terminal = True
                    break

                queued = self._put_queue_item(
                    audio_queue,
                    AudioChunk(
                        sequence_id=item.sequence_id,
                        audio=segment,
                        text=item.text,
                        decode_ms=stats.decode_ms,
                        microchunk_size=stats.microchunk_size,
                    ),
                    cancel_event=cancel_event,
                    req_tag=req_tag,
                    stage="dac",
                    blocked_message="dac: blocked on audio_queue",
                    success_message="dac: unblocked on audio_queue",
                    extra_log=f"sequence_id={item.sequence_id}",
                )
                if not queued:
                    break

                logger.info(
                    "dac: queued_audio req={} sequence_id={} samples={} decode_ms={:.1f} microchunk={} codes_depth={} audio_depth={} free_gb={} alloc_gb={} reserved_gb={}",
                    req_tag,
                    item.sequence_id,
                    len(segment),
                    stats.decode_ms,
                    stats.microchunk_size,
                    codes_queue.qsize(),
                    audio_queue.qsize(),
                    f"{stats.snapshot_after.free_gb:.2f}" if stats.snapshot_after.free_gb is not None else "na",
                    f"{stats.snapshot_after.allocated_gb:.2f}" if stats.snapshot_after.allocated_gb is not None else "na",
                    f"{stats.snapshot_after.reserved_gb:.2f}" if stats.snapshot_after.reserved_gb is not None else "na",
                )
        finally:
            if cancel_event.is_set() and not emitted_terminal:
                self._put_terminal_signal(
                    audio_queue,
                    PipelineSignal(kind="cancel", reason="request_cancelled"),
                    req_tag=req_tag,
                    stage="dac",
                )
            dac_done.set()
            logger.info(
                "dac: done req={} codes_depth={} audio_depth={}",
                req_tag,
                codes_queue.qsize(),
                audio_queue.qsize(),
            )

    def _drain_pipeline_queue(self, target_queue: queue.Queue) -> None:
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return

    def inference(self, req: ServeTTSRequest) -> Generator[InferenceResult, None, None]:
        profile = _env_flag("FISH_PROFILE_INFERENCE", False)
        req_tag = hex(id(req))[-6:]
        t_start = time.perf_counter()
        ttfa_logged = False

        def _mark(event: str, **extra) -> None:
            if not profile:
                return
            now = time.perf_counter()
            elapsed_ms = (now - t_start) * 1000.0
            details = " ".join(f"{k}={v}" for k, v in extra.items())
            logger.info(
                "pipeline_timing req={} event={} elapsed_ms={:.1f} {}",
                req_tag,
                event,
                elapsed_ms,
                details,
            )

        ref_id = req.reference_id
        prompt_tokens, prompt_texts = [], []

        if ref_id is not None:
            prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)
            _mark("ref_loaded", mode="id")
        elif req.references:
            prompt_tokens, prompt_texts = self.load_by_hash(
                req.references, req.use_memory_cache
            )
            _mark("ref_loaded", mode="hash", refs=len(req.references))

        if req.seed is not None:
            set_seed(req.seed)
            logger.warning("set seed: {}", req.seed)

        if req.control == "cleanup":
            response_queue = self.send_Llama_request(
                req,
                prompt_tokens,
                prompt_texts,
                req_tag=req_tag,
                ack_queue=None,
                cancel_event=None,
                response_queue_maxsize=1,
            )
            wrapped_result = response_queue.get()
            if wrapped_result.status == "error":
                err = (
                    wrapped_result.response
                    if isinstance(wrapped_result.response, Exception)
                    else RuntimeError("Cleanup request failed")
                )
                yield InferenceResult(code="error", audio=None, error=err)
            return

        sample_rate = self._sample_rate()
        pipeline_config = self._build_pipeline_config()
        cancel_event = threading.Event()
        producer_done = threading.Event()
        dac_done = threading.Event()
        memory_pressure_event = threading.Event()
        ack_queue: queue.Queue = queue.Queue()
        codes_queue: queue.Queue = queue.Queue(maxsize=pipeline_config.codes_queue_maxsize)
        audio_queue: queue.Queue = queue.Queue(maxsize=pipeline_config.audio_queue_maxsize)

        response_queue = self.send_Llama_request(
            req,
            prompt_tokens,
            prompt_texts,
            req_tag=req_tag,
            ack_queue=ack_queue,
            cancel_event=cancel_event,
            response_queue_maxsize=pipeline_config.response_queue_maxsize,
        )
        _mark(
            "llama_queued",
            codes_queue_maxsize=pipeline_config.codes_queue_maxsize,
            audio_queue_maxsize=pipeline_config.audio_queue_maxsize,
            max_ahead_chunks=pipeline_config.max_ahead_chunks,
        )

        producer_thread = threading.Thread(
            target=self._producer_loop,
            kwargs=dict(
                req_tag=req_tag,
                response_queue=response_queue,
                codes_queue=codes_queue,
                ack_queue=ack_queue,
                cancel_event=cancel_event,
                producer_done=producer_done,
                memory_pressure_event=memory_pressure_event,
                config=pipeline_config,
            ),
            daemon=True,
        )
        dac_thread = threading.Thread(
            target=self._dac_consumer_loop,
            kwargs=dict(
                req_tag=req_tag,
                codes_queue=codes_queue,
                audio_queue=audio_queue,
                cancel_event=cancel_event,
                producer_done=producer_done,
                dac_done=dac_done,
                memory_pressure_event=memory_pressure_event,
            ),
            daemon=True,
        )
        producer_thread.start()
        dac_thread.start()

        pending_audio: dict[int, AudioChunk] = {}
        next_expected_sequence = 0
        segments: list[np.ndarray] = []
        produced_segments = 0
        terminal_signal: PipelineSignal | None = None
        sender_wait_since: float | None = None

        try:
            if req.streaming and req.format == "wav":
                yield InferenceResult(
                    code="header",
                    audio=(sample_rate, np.array(wav_chunk_header(sample_rate=sample_rate))),
                    error=None,
                )

            while True:
                try:
                    item = audio_queue.get(timeout=QUEUE_POLL_TIMEOUT_SEC)
                    sender_wait_since = None
                except queue.Empty:
                    if producer_done.is_set() and dac_done.is_set():
                        break

                    if sender_wait_since is None:
                        sender_wait_since = time.perf_counter()
                    elif (
                        time.perf_counter() - sender_wait_since
                    ) >= SENDER_WAIT_LOG_INTERVAL_SEC:
                        logger.info(
                            "sender: underrun req={} codes_depth={} audio_depth={} producer_done={} dac_done={}",
                            req_tag,
                            codes_queue.qsize(),
                            audio_queue.qsize(),
                            producer_done.is_set(),
                            dac_done.is_set(),
                        )
                        sender_wait_since = time.perf_counter()
                    continue

                if isinstance(item, PipelineSignal):
                    terminal_signal = item
                    break

                if not isinstance(item, AudioChunk):
                    terminal_signal = PipelineSignal(
                        kind="error",
                        error=TypeError(
                            f"Unexpected audio_queue payload: {type(item).__name__}"
                        ),
                        reason="invalid_audio_item",
                    )
                    break

                pending_audio[item.sequence_id] = item
                while next_expected_sequence in pending_audio:
                    ready = pending_audio.pop(next_expected_sequence)
                    if not ttfa_logged:
                        logger.info(
                            "pipeline: ttfa req={} ttfa_ms={:.1f}",
                            req_tag,
                            (time.perf_counter() - t_start) * 1000.0,
                        )
                        ttfa_logged = True

                    logger.info(
                        "sender: segment req={} sequence_id={} samples={} decode_ms={:.1f} microchunk={} codes_depth={} audio_depth={}",
                        req_tag,
                        ready.sequence_id,
                        len(ready.audio),
                        ready.decode_ms,
                        ready.microchunk_size,
                        codes_queue.qsize(),
                        audio_queue.qsize(),
                    )
                    yield InferenceResult(
                        code="segment",
                        audio=(sample_rate, ready.audio),
                        error=None,
                    )
                    produced_segments += 1
                    if not req.streaming:
                        segments.append(ready.audio)
                    next_expected_sequence += 1

            if terminal_signal is None and producer_done.is_set() and dac_done.is_set():
                terminal_signal = PipelineSignal(kind="cancel", reason="pipeline_stopped")

            if terminal_signal is not None and terminal_signal.kind == "error":
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=terminal_signal.error or RuntimeError("Unknown pipeline error"),
                )
                return

            if terminal_signal is not None and terminal_signal.kind == "cancel":
                logger.info(
                    "pipeline: cancelled req={} reason={}",
                    req_tag,
                    terminal_signal.reason or "unknown",
                )
                return

            if produced_segments == 0:
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=RuntimeError("No audio generated, please check the input text."),
                )
                return

            if req.streaming:
                yield InferenceResult(code="final", audio=None, error=None)
                return

            audio = np.concatenate(segments, axis=0)
            yield InferenceResult(
                code="final",
                audio=(sample_rate, audio),
                error=None,
            )
        except GeneratorExit:
            cancel_event.set()
            self._safe_ack(
                ack_queue,
                ACK_ABORT,
                req_tag=req_tag,
                reason="generator_closed",
            )
            logger.info("pipeline: generator_closed req={}", req_tag)
            raise
        finally:
            if terminal_signal is not None and terminal_signal.kind == "eos":
                self._safe_ack(
                    ack_queue,
                    ACK_CONTINUE,
                    req_tag=req_tag,
                    reason="inference_finally_eos",
                )
            else:
                cancel_event.set()
                self._safe_ack(
                    ack_queue,
                    ACK_ABORT,
                    req_tag=req_tag,
                    reason="inference_finally_abort",
                )

            self._drain_pipeline_queue(audio_queue)
            self._drain_pipeline_queue(codes_queue)

            producer_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            dac_thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)

            logger.info(
                "pipeline: finished req={} producer_alive={} dac_alive={} codes_depth={} audio_depth={} total_ms={:.1f}",
                req_tag,
                producer_thread.is_alive(),
                dac_thread.is_alive(),
                codes_queue.qsize(),
                audio_queue.qsize(),
                (time.perf_counter() - t_start) * 1000.0,
            )

    def send_Llama_request(
        self,
        req: ServeTTSRequest,
        prompt_tokens: list,
        prompt_texts: list,
        req_tag: str,
        ack_queue: queue.Queue | None = None,
        cancel_event: threading.Event | None = None,
        response_queue_maxsize: int = 0,
    ) -> queue.Queue:
        stream_tokens = getattr(req, "stream_tokens", False) or req.streaming

        request = dict(
            device=self.decoder_model.device,
            req_tag=req_tag,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
            stream_tokens=stream_tokens,
            stream_chunk_size=getattr(req, "stream_chunk_size", 8),
            initial_stream_chunk_size=getattr(req, "initial_stream_chunk_size", None),
            ack_queue=ack_queue,
            cancel_event=cancel_event,
            cleanup_mode=getattr(req, "cleanup_mode", "request_end"),
            stream_empty_cache=getattr(req, "stream_empty_cache", None),
            control=getattr(req, "control", None),
        )

        response_queue = queue.Queue(maxsize=max(0, response_queue_maxsize))

        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        return response_queue

    def get_audio_segment(
        self,
        codes: torch.Tensor,
        *,
        req_tag: str,
        segment_idx: int,
        max_codes_per_step: int | None = None,
    ) -> tuple[np.ndarray, DACDecodeStats]:
        with torch.inference_mode():
            decoder_device = self._decoder_device()
            with autocast_exclude_mps(
                device_type=decoder_device.type,
                dtype=self.precision,
            ):
                segment, stats = self.decode_vq_tokens(
                    codes=codes,
                    req_tag=req_tag,
                    segment_idx=segment_idx,
                    max_codes_per_step=max_codes_per_step,
                )

        return segment.float().detach().cpu().numpy(), stats

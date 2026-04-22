import os
import queue
import time
from typing import Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import ServeTTSRequest


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

    def inference(self, req: ServeTTSRequest) -> Generator[InferenceResult, None, None]:
        """
        Основной inference-пайплайн:
        - загружаем reference
        - отправляем запрос в LLM worker
        - получаем semantic/VQ chunk'и
        - декодируем их в аудио

        Важное изменение:
        здесь включён ЖЁСТКИЙ backpressure для streaming-режима.

        Раньше worker мог начать генерировать следующий chunk ещё до того,
        как текущий chunk:
        1) был декодирован DAC'ом
        2) был реально отдан наружу через HTTP/WebSocket поток

        На 32 GB VRAM это легко приводит к накоплению промежуточных буферов,
        скачкам памяти, смене поведения/голоса на длинной сессии и OOM.

        Теперь следующий chunk разрешается только после того,
        как текущий chunk уже обработан до конца.
        """

        profile = os.getenv("FISH_PROFILE_INFERENCE", "0") in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }
        req_tag = hex(id(req))[-6:]
        t_start = time.perf_counter()
        t_prev = t_start

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

        def _ack_worker_once() -> None:
            """
            Подтверждение worker'у, что текущий chunk уже можно считать
            полностью обработанным.

            Очередь подтверждений имеет maxsize=1, поэтому используем put_nowait().
            Если там уже лежит ack — значит логика где-то дала двойное подтверждение,
            но блокироваться тут нельзя.
            """
            if ack_queue is None:
                return
            try:
                ack_queue.put_nowait(None)
            except queue.Full:
                logger.warning("stream: ack queue already full req={}", req_tag)

        ref_id = req.reference_id
        prompt_tokens, prompt_texts = [], []

        if ref_id is not None:
            prompt_tokens, prompt_texts = self.load_by_id(
                ref_id, req.use_memory_cache
            )
            _mark("ref_loaded", mode="id")
        elif req.references:
            prompt_tokens, prompt_texts = self.load_by_hash(
                req.references, req.use_memory_cache
            )
            _mark("ref_loaded", mode="hash", refs=len(req.references))

        if req.seed is not None:
            set_seed(req.seed)
            logger.warning(f"set seed: {req.seed}")

        stream_tokens = getattr(req, "stream_tokens", False) or req.streaming

        # В streaming-режиме используем очередь подтверждений размером 1.
        # Это делает пайплайн строго пошаговым:
        # worker -> response_queue -> decode -> yield -> ack -> следующий chunk.
        ack_queue = queue.Queue(maxsize=1) if stream_tokens else None

        response_queue = self.send_Llama_request(
            req,
            prompt_tokens,
            prompt_texts,
            req_tag=req_tag,
            ack_queue=ack_queue,
        )
        _mark("llama_queued")

        if stream_tokens:
            logger.info(
                "stream: inference started (strict backpressure mode), req={}",
                req_tag,
            )

        if hasattr(self.decoder_model, "spec_transform"):
            sample_rate = self.decoder_model.spec_transform.sample_rate
        else:
            sample_rate = self.decoder_model.sample_rate

        if req.streaming and req.format == "wav":
            _mark("yield_header")
            yield InferenceResult(
                code="header",
                audio=(
                    sample_rate,
                    np.array(wav_chunk_header(sample_rate=sample_rate)),
                ),
                error=None,
            )

        segments: list[np.ndarray] = []
        seg_idx = 0
        produced_segments = 0
        failed = False

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
                failed = True
                logger.error(
                    "stream: got error from worker req={} err={}",
                    req_tag,
                    wrapped_result.response,
                )

                # Освобождаем worker, если он ждёт подтверждение.
                _ack_worker_once()

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
                _ack_worker_once()
                raise TypeError(
                    f"Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                )

            result = wrapped_result.response

            if result.action != "next":
                if result.codes is None:
                    failed = True
                    err = RuntimeError(
                        "Worker returned action='sample' but codes is None"
                    )
                    logger.error("stream: invalid worker result req={} err={}", req_tag, err)
                    _ack_worker_once()
                    yield InferenceResult(
                        code="error",
                        audio=None,
                        error=err,
                    )
                    break

                if stream_tokens:
                    logger.info(
                        "stream: decoding segment seg_idx={} codes_shape={} req={}",
                        seg_idx + 1,
                        result.codes.shape,
                        req_tag,
                    )

                _mark(
                    "decode_vq_start",
                    segment_idx=seg_idx + 1,
                    codes_frames=result.codes.shape[1],
                )

                # В response_queue chunk'и держим на CPU, чтобы не копить VRAM.
                # Непосредственно перед DAC decode переносим текущий chunk на GPU.
                codes = result.codes
                if not codes.is_cuda:
                    codes = codes.to(self.decoder_model.device)

                try:
                    segment = self.get_audio_segment(codes)
                except Exception as seg_err:
                    if stream_tokens:
                        logger.exception(
                            "stream: get_audio_segment FAILED seg_idx={} codes_shape={} req={}: {}",
                            seg_idx + 1,
                            result.codes.shape,
                            req_tag,
                            seg_err,
                        )
                    raise
                finally:
                    # Важно удалить временный GPU tensor как можно раньше.
                    del codes

                seg_idx += 1
                produced_segments += 1

                _mark("segment_decoded", segment_idx=seg_idx, samples=len(segment))

                if stream_tokens:
                    logger.info(
                        "stream: segment_decoded seg_idx={} samples={} req={}",
                        seg_idx,
                        len(segment),
                        req_tag,
                    )

                if req.streaming:
                    _mark("yield_segment", segment_idx=seg_idx)

                    # Критично:
                    # ack отправляем НЕ ДО decode и НЕ ДО yield,
                    # а только после того, как chunk реально отдан наружу.
                    yield InferenceResult(
                        code="segment",
                        audio=(sample_rate, segment),
                        error=None,
                    )

                    _ack_worker_once()
                else:
                    segments.append(segment)
                    _ack_worker_once()
            else:
                if stream_tokens:
                    logger.info(
                        "stream: got_next (end of stream), total_segments={} req={}",
                        seg_idx,
                        req_tag,
                    )

                _ack_worker_once()
                _mark("got_next")
                break

        if failed:
            return None

        if produced_segments == 0:
            _mark("yield_error_empty")
            yield InferenceResult(
                code="error",
                audio=None,
                error=RuntimeError(
                    "No audio generated, please check the input text."
                ),
            )
            return None

        # В streaming-режиме итоговый final не должен ещё раз возвращать всё аудио целиком,
        # иначе верхний слой может воспроизвести ответ повторно.
        if req.streaming:
            _mark("yield_final_stream")
            yield InferenceResult(
                code="final",
                audio=None,
                error=None,
            )
        else:
            audio = np.concatenate(segments, axis=0)
            _mark("yield_final", total_samples=len(audio), segments=len(segments))
            yield InferenceResult(
                code="final",
                audio=(sample_rate, audio),
                error=None,
            )

        return None

    def send_Llama_request(
        self,
        req: ServeTTSRequest,
        prompt_tokens: list,
        prompt_texts: list,
        req_tag: str,
        ack_queue: queue.Queue | None = None,
    ) -> queue.Queue:
        """
        Отправка запроса в LLM worker.

        Важное изменение:
        response_queue теперь ограниченная.
        Для streaming-режима нам не нужна длинная очередь ответов —
        нужен максимум один chunk "в полёте".
        """

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
            cleanup_mode=getattr(req, "cleanup_mode", "request_end"),
            stream_empty_cache=getattr(req, "stream_empty_cache", None),
            control=getattr(req, "control", None),
        )

        # Для stream достаточно одной ячейки.
        # Это дополнительная страховка от накопления chunk'ов.
        response_queue = queue.Queue(maxsize=1 if stream_tokens else 2)

        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        return response_queue

    def get_audio_segment(self, codes: torch.Tensor) -> np.ndarray:
        """
        Декодирование одного VQ chunk в PCM waveform.
        """
        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            segment = self.decode_vq_tokens(codes=codes)

        return segment.float().detach().cpu().numpy()
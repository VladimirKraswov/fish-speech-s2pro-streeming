from __future__ import annotations

import gc
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
from loguru import logger

from fish_speech.generation.decode import _cache_max_seq_len
from fish_speech.generation.models import init_model
from fish_speech.generation.prompt_builder import GenerateResponse, generate_committed_segments
from fish_speech.driver.config import load_runtime_config

@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[Union[GenerateResponse, Exception]] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def _model_param_memory_gb(module: torch.nn.Module) -> tuple[float, int]:
    total = 0
    count = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
        count += p.numel()
    return (round(total / (1024**3), 3), count)


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
    memory_info: dict | None = None,
):
    """
    memory_info: optional shared dict; worker will set llama_param_gb, llama_param_count after load.
    """
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        runtime = load_runtime_config()
        model_cfg = runtime.model

        profile_cadence = model_cfg.profile_inference
        cleanup_after_request = model_cfg.cleanup_after_request
        cleanup_every_n_requests = model_cfg.cleanup_every_n_requests
        cleanup_on_error = model_cfg.cleanup_on_error
        empty_cache_per_stream_chunk = model_cfg.empty_cache_per_stream_chunk
        requests_since_cleanup = 0

        model, decode_one_token = init_model(
            checkpoint_path, device, precision, compile=compile
        )
        cache_len = _cache_max_seq_len(model)
        logger.info(
            "KV cache max_seq_len={} (model max={})",
            cache_len,
            model.config.max_seq_len,
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=cache_len,
                dtype=next(model.parameters()).dtype,
            )
        if memory_info is not None:
            gb, count = _model_param_memory_gb(model)
            memory_info["llama_param_gb"] = gb
            memory_info["llama_param_count"] = count
        init_event.set()

        def maybe_cleanup(*, reason: str, force: bool = False) -> None:
            nonlocal requests_since_cleanup
            if not torch.cuda.is_available():
                return

            should_cleanup = force or cleanup_after_request
            if (
                not should_cleanup
                and cleanup_every_n_requests > 0
                and requests_since_cleanup >= cleanup_every_n_requests
            ):
                should_cleanup = True

            if not should_cleanup:
                return

            before_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2)
            before_reserved = round(torch.cuda.memory_reserved() / (1024**3), 2)
            logger.info(
                "worker: cleanup start reason={} alloc_gb={} reserved_gb={}",
                reason,
                before_alloc,
                before_reserved,
            )

            clear_caches_fn = getattr(model, "clear_caches", None)
            if clear_caches_fn is not None:
                try:
                    clear_caches_fn()
                except Exception as clear_err:
                    logger.warning("clear_caches failed: {}", clear_err)

            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

            after_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2)
            after_reserved = round(torch.cuda.memory_reserved() / (1024**3), 2)
            logger.info(
                "worker: cleanup done reason={} alloc_gb={} reserved_gb={}",
                reason,
                after_alloc,
                after_reserved,
            )
            requests_since_cleanup = 0

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                maybe_cleanup(reason="shutdown", force=True)
                break

            kwargs = item.request
            req_tag = str(kwargs.pop("req_tag", "na"))
            ack_queue = kwargs.pop("ack_queue", None)
            response_queue = item.response_queue
            t_req_start = time.perf_counter()
            t_last_put = t_req_start
            stream_tokens = kwargs.get("stream_tokens", False)
            put_count = 0
            if stream_tokens:
                logger.info(
                    "stream: worker got request req={} initial_stream_chunk_size={} stream_chunk_size={}",
                    req_tag,
                    kwargs.get("initial_stream_chunk_size"),
                    kwargs.get("stream_chunk_size"),
                )

            try:
                for chunk in generate_committed_segments(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    if stream_tokens and put_count < 5:
                        logger.info(
                            "stream: worker putting chunk put_count={} action={} req={}",
                            put_count,
                            getattr(chunk, "action", None),
                            req_tag,
                        )
                    put_count += 1
                    if profile_cadence:
                        now = time.perf_counter()
                        delta_ms = (now - t_last_put) * 1000.0
                        total_ms = (now - t_req_start) * 1000.0
                        t_last_put = now
                        action = getattr(chunk, "action", type(chunk).__name__)
                        vram_s = ""
                        if torch.cuda.is_available():
                            vram_s = " vram_alloc_gb={:.2f} vram_max_gb={:.2f}".format(
                                torch.cuda.memory_allocated() / (1024**3),
                                torch.cuda.max_memory_allocated() / (1024**3),
                            )
                        logger.info(
                            "queue_put req={} action={} delta_ms={:.1f} total_ms={:.1f}{}",
                            req_tag,
                            action,
                            delta_ms,
                            total_ms,
                            vram_s,
                        )
                    out = chunk
                    codes = getattr(chunk, "codes", None)
                    if codes is not None and codes.is_cuda:
                        out = GenerateResponse(
                            action=chunk.action,
                            codes=codes.cpu(),
                            text=getattr(chunk, "text", None),
                        )
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=out)
                    )
                    if stream_tokens and ack_queue is not None:
                        ack_queue.get()
                    if stream_tokens and empty_cache_per_stream_chunk and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                requests_since_cleanup += 1
                maybe_cleanup(reason="success", force=False)

            except Exception as e:
                logger.error(
                    "stream: worker EXCEPTION req={} put_count={}: {}",
                    req_tag,
                    put_count,
                    e,
                    exc_info=True,
                )
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))
                if cleanup_on_error:
                    maybe_cleanup(reason="error", force=True)

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue

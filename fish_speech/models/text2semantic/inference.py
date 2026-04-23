import gc
import os
import queue
import re
import threading
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal, Optional, Tuple, Union, cast

import click
import numpy as np
import torch
import torch._inductor.config
from loguru import logger
from tqdm import tqdm

from fish_speech.content_sequence import (
    TextPart,
    VQPart,
)
from fish_speech.conversation import Conversation, Message
from fish_speech.tokenizer import IM_END_TOKEN

# Явные ack-команды между pipeline producer и LLM worker.
# Теперь ack означает, что chunk безопасно принят bounded CPU pipeline,
# а не то, что DAC уже закончил decode.
ACK_CONTINUE = "continue"
ACK_ABORT = "abort"


def _to_normal_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Copy tensor so it is not an inference tensor (AOT fails on inplace into inference tensors).
    detach().clone() inside inference_mode(False) can still yield inference tensors in some PyTorch
    versions; roundtrip via CPU forces a new allocation and clears the flag.
    """
    if t is None:
        return None
    with torch.inference_mode(False):
        return t.detach().cpu().clone().to(t.device)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _normalize_cleanup_mode(cleanup_mode: str | None) -> str:
    mode = (cleanup_mode or "request_end").strip().lower()
    aliases = {
        "default": "request_end",
        "heavy": "request_end",
        "request_end": "request_end",
        "session_close": "request_end",
        "idle": "session_idle",
        "light": "session_idle",
        "session_idle": "session_idle",
        "keep_warm": "none",
        "hot": "none",
        "none": "none",
    }
    return aliases.get(mode, mode)


def _cuda_index(device: torch.device | str | None = None) -> int | None:
    if not torch.cuda.is_available():
        return None

    if device is None:
        return torch.cuda.current_device()

    if isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda":
        return None

    return torch.cuda.current_device() if device.index is None else device.index


def _cuda_free_gb(device: torch.device | str | None = None) -> float | None:
    cuda_idx = _cuda_index(device)
    if cuda_idx is None:
        return None

    if hasattr(torch.cuda, "mem_get_info"):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(cuda_idx)
            return free_bytes / (1024**3)
        except Exception:
            pass    

    props = torch.cuda.get_device_properties(cuda_idx)
    total_bytes = props.total_memory
    reserved_bytes = torch.cuda.memory_reserved(cuda_idx)
    allocated_bytes = torch.cuda.memory_allocated(cuda_idx)
    used_bytes = max(reserved_bytes, allocated_bytes)
    free_bytes = max(0, total_bytes - used_bytes)
    return free_bytes / (1024**3)


def _light_cuda_cleanup() -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def _heavy_cuda_cleanup(
    model: torch.nn.Module | None = None,
) -> tuple[float | None, float | None]:
    before_clear_gb = None
    after_clear_gb = None

    if torch.cuda.is_available():
        before_clear_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)

    clear_caches_fn = getattr(model, "clear_caches", None) if model is not None else None
    if clear_caches_fn is not None:
        try:
            clear_caches_fn()
        except Exception as clear_err:
            logger.warning("clear_caches failed: {}", clear_err)

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        after_clear_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)

    return before_clear_gb, after_clear_gb


def _run_request_cleanup(
    *,
    model: torch.nn.Module,
    cleanup_mode: str,
    req_tag: str,
    reason: str = "request_end",
) -> None:
    mode = _normalize_cleanup_mode(cleanup_mode)

    if mode == "request_end":
        logger.info(
            "worker: post-request cleanup starting req={} mode={} reason={}",
            req_tag,
            mode,
            reason,
        )
        before_clear_gb, after_clear_gb = _heavy_cuda_cleanup(model)
        if before_clear_gb is not None:
            logger.info(
                "worker: clear_caches done alloc before={} GB after={} GB req={}",
                before_clear_gb,
                after_clear_gb,
                req_tag,
            )
        return

    if mode == "session_idle":
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = None

        free_gb = _cuda_free_gb(device)
        force_heavy_below_gb = _env_float(
            "FISH_SESSION_IDLE_FORCE_HEAVY_BELOW_GB", 3.0
        )

        if free_gb is not None and free_gb < force_heavy_below_gb:
            logger.warning(
                "worker: session_idle fallback to heavy cleanup req={} free={:.2f} GB < {:.2f} GB reason={}",
                req_tag,
                free_gb,
                force_heavy_below_gb,
                reason,
            )
            before_clear_gb, after_clear_gb = _heavy_cuda_cleanup(model)
            if before_clear_gb is not None:
                logger.info(
                    "worker: clear_caches done alloc before={} GB after={} GB req={}",
                    before_clear_gb,
                    after_clear_gb,
                    req_tag,
                )
            return
        
        logger.info(
            "worker: keeping model warm req={} mode={} reason={}",
            req_tag,
            mode,
            reason,
        )
        if _env_flag("FISH_LIGHT_CUDA_CLEANUP_ON_SESSION_IDLE", True):
            _light_cuda_cleanup()
        return

    if mode == "none":
        logger.info(
            "worker: skip cleanup req={} mode={} reason={}",
            req_tag,
            mode,
            reason,
        )
        return

    logger.warning(
        "worker: unknown cleanup_mode='{}' for req={}, fallback to request_end",
        cleanup_mode,
        req_tag,
    )
    before_clear_gb, after_clear_gb = _heavy_cuda_cleanup(model)
    if before_clear_gb is not None:
        logger.info(
            "worker: clear_caches done alloc before={} GB after={} GB req={}",
            before_clear_gb,
            after_clear_gb,
            req_tag,
        )


def _maybe_force_cleanup_before_request(
    *,
    model: torch.nn.Module,
    req_tag: str,
    cleanup_mode: str,
) -> None:
    """
    Если мы живём в session_idle режиме, но VRAM уже почти не осталось,
    лучше один раз сделать heavy cleanup до старта нового запроса,
    чем уронить процесс посередине DAC decode.
    """
    if _normalize_cleanup_mode(cleanup_mode) != "session_idle":
        return

    min_free_gb = _env_float("FISH_SESSION_IDLE_MIN_FREE_GB", 4.0)
    device = next(model.parameters()).device
    free_gb = _cuda_free_gb(device)
    if free_gb is None or free_gb >= min_free_gb:
        return

    logger.warning(
        "worker: low free VRAM before request req={} free={:.2f} GB < {:.2f} GB, "
        "forcing heavy cleanup before generation",
        req_tag,
        free_gb,
        min_free_gb,
    )
    _run_request_cleanup(
        model=model,
        cleanup_mode="request_end",
        req_tag=req_tag,
        reason="pre_request_low_free_vram",
    )


def _queue_put_with_cancel(
    target_queue: queue.Queue,
    item,
    *,
    cancel_event: threading.Event | None,
    timeout_sec: float = 0.1,
) -> bool:
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return False
        try:
            target_queue.put(item, timeout=timeout_sec)
            return True
        except queue.Full:
            if cancel_event is not None and cancel_event.is_set():
                return False


def _wait_for_consumer_ack(
    ack_queue: queue.Queue,
    *,
    cancel_event: threading.Event | None,
    ack_timeout_sec: float,
) -> str:
    deadline = time.perf_counter() + ack_timeout_sec
    poll_sec = min(0.25, ack_timeout_sec)

    while True:
        if cancel_event is not None and cancel_event.is_set():
            return ACK_ABORT

        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for pipeline ack after {ack_timeout_sec:.1f}s"
            )

        try:
            return ack_queue.get(timeout=min(poll_sec, remaining))
        except queue.Empty:
            if cancel_event is not None and cancel_event.is_set():
                return ACK_ABORT


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def _cache_max_seq_len(model: BaseTransformer) -> int:
    """
    Max sequence length used for KV cache and buffers. Tuned for streaming (short turns);
    512 tokens ~= 41 s audio. Set FISH_CACHE_MAX_SEQ_LEN to override (e.g. 4096 for long form).
    """
    default = 512
    raw = os.environ.get("FISH_CACHE_MAX_SEQ_LEN", str(default))
    try:
        n = int(raw)
    except ValueError:
        n = default
    return min(max(1, n), model.config.max_seq_len)


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.rand_like(probs_sort)
    q = -torch.log(q)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


RAS_WIN_SIZE = 10
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    indices = torch.arange(sorted_logits.shape[-1], device=sorted_logits.device)
    top_k_mask = indices >= top_k
    sorted_indices_to_remove = (cum_probs > top_p) | top_k_mask
    sorted_indices_to_remove[0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = torch.where(indices_to_remove, float("-Inf"), logits)
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    forward_result = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = forward_result.logits
    hidden_states = forward_result.hidden_states

    biased_logits = logits + semantic_logit_bias

    main_token_normal = sample(
        biased_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )[0]

    high_temp = torch.tensor(
        RAS_HIGH_TEMP, device=temperature.device, dtype=temperature.dtype
    )
    high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=top_p.device, dtype=top_p.dtype)
    main_token_high = sample(
        biased_logits, temperature=high_temp, top_p=high_top_p, top_k=top_k
    )[0]

    if previous_tokens is not None:
        in_window = (previous_tokens[0] == main_token_normal).any()
        is_semantic = (main_token_normal >= model.config.semantic_begin_id) & (
            main_token_normal <= model.config.semantic_end_id
        )
        should_use_high = in_window & is_semantic
        main_token_normal = torch.where(
            should_use_high, main_token_high, main_token_normal
        )

    codebooks = [main_token_normal]

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)

    a = codebooks[0] - model.config.semantic_begin_id
    a = torch.clamp(a, min=0, max=model.config.codebook_size - 1)

    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits

        a = sample(
            short_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)

    del logits, hidden_states, forward_result

    return codebooks.T


def decode_n_tokens(
    model: DualARTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    stream_chunk_size: Optional[int] = None,
    initial_stream_chunk_size: Optional[int] = None,
    compile: bool = False,
) -> Iterator[torch.Tensor]:
    """
    Generate tokens autoregressively.

    When stream_chunk_size is None, yields once with the full tensor (backward compatible).
    When stream_chunk_size is set, yields token chunks for low-TTFA streaming.

    `initial_stream_chunk_size` controls the very first streamed chunk size together with the
    prefill token yielded by generate(). Because generate() already emits the first token, the
    first chunk produced here is `initial_stream_chunk_size - 1`.
    """

    need_normal_state = stream_chunk_size is not None
    if need_normal_state:
        cur_token = cast(torch.Tensor, _to_normal_tensor(cur_token))
        input_pos = cast(torch.Tensor, _to_normal_tensor(input_pos))
        temperature = cast(torch.Tensor, _to_normal_tensor(temperature))
        top_p = cast(torch.Tensor, _to_normal_tensor(top_p))
        semantic_logit_bias = cast(torch.Tensor, _to_normal_tensor(semantic_logit_bias))
        if audio_masks is not None:
            audio_masks = _to_normal_tensor(audio_masks)
        if audio_parts is not None:
            audio_parts = _to_normal_tensor(audio_parts)

    _prev_zeros = torch.zeros(
        (model.config.num_codebooks + 1, RAS_WIN_SIZE),
        dtype=torch.int,
        device=cur_token.device,
    )
    previous_tokens = (
        cast(torch.Tensor, _to_normal_tensor(_prev_zeros))
        if need_normal_state
        else _prev_zeros
    )
    new_tokens: list[torch.Tensor] = []
    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)
    do_stream_log = stream_chunk_size is not None

    current_stream_chunk_size: Optional[int] = None
    if stream_chunk_size is not None:
        if initial_stream_chunk_size is not None and initial_stream_chunk_size > 1:
            current_stream_chunk_size = max(1, initial_stream_chunk_size - 1)
        else:
            current_stream_chunk_size = stream_chunk_size

    llm_critical_free_gb = _env_float("FISH_LLM_CRITICAL_FREE_GB", 1.5)

    for i in tqdm(range(num_new_tokens)):
        if (i + 1) % 4 == 0:
            free_gb = _cuda_free_gb(cur_token.device)
            if free_gb is not None and free_gb < llm_critical_free_gb:
                logger.warning(
                    "stream: low VRAM during generation ({:.2f} GB < {} GB), triggering cleanup",
                    free_gb,
                    llm_critical_free_gb,
                )
                _light_cuda_cleanup()

        if i == 0 and do_stream_log:
            logger.info("stream: decode_n_tokens started generating first token")

        if do_stream_log and i < 3:
            logger.info(
                "stream: decode_n_tokens iter={} cur_token.shape={} input_pos={}",
                i,
                cur_token.shape,
                input_pos.shape,
            )

        try:
            use_math = compile or (
                os.environ.get("FISH_SDPA_MATH", "").strip()
                in ("1", "true", "TRUE", "yes", "YES")
            )
            if use_math:
                with sdpa_kernel(SDPBackend.MATH):
                    next_token = decode_one_token(
                        model=model,
                        x=cur_token,
                        input_pos=input_pos,
                        previous_tokens=previous_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        semantic_logit_bias=semantic_logit_bias,
                        audio_masks=audio_masks,
                        audio_parts=audio_parts,
                    ).clone()
            else:
                next_token = decode_one_token(
                    model=model,
                    x=cur_token,
                    input_pos=input_pos,
                    previous_tokens=previous_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    semantic_logit_bias=semantic_logit_bias,
                    audio_masks=audio_masks,
                    audio_parts=audio_parts,
                ).clone()
        except Exception as e:
            logger.exception(
                "stream: decode_n_tokens FAILED at iter={} (cur_token.shape={}): {}",
                i,
                cur_token.shape,
                e,
            )
            raise

        if need_normal_state:
            next_token = cast(torch.Tensor, _to_normal_tensor(next_token))
            input_pos = cast(torch.Tensor, _to_normal_tensor(input_pos + 1))
            cur_token = cast(
                torch.Tensor,
                _to_normal_tensor(
                    next_token.view(1, model.config.num_codebooks + 1, -1)
                ),
            )
            previous_tokens = cast(
                torch.Tensor, _to_normal_tensor(previous_tokens.roll(-1, dims=1))
            )
            prev_col = next_token.view(model.config.num_codebooks + 1, -1)[:, 0]
            previous_tokens[:, -1].copy_(prev_col)
        else:
            next_token = next_token.detach().clone()
            input_pos = (input_pos + 1).detach().clone()
            cur_token = next_token.view(1, model.config.num_codebooks + 1, -1).clone()
            previous_tokens = previous_tokens.roll(-1, dims=1)
            previous_tokens[:, -1] = next_token.view(
                model.config.num_codebooks + 1, -1
            )[:, 0].clone()

        new_tokens.append(next_token)

        if (
            stream_chunk_size is not None
            and current_stream_chunk_size is not None
            and len(new_tokens) >= current_stream_chunk_size
        ):
            chunk_out = torch.cat(new_tokens, dim=1).detach().clone()
            if do_stream_log:
                logger.info(
                    "stream: decode_n_tokens yielding chunk shape={} after iter={} target_chunk_size={}",
                    chunk_out.shape,
                    i,
                    current_stream_chunk_size,
                )
            yield chunk_out
            new_tokens = []
            current_stream_chunk_size = stream_chunk_size

        if cur_token[0, 0, -1] == im_end_id:
            if do_stream_log:
                logger.info("stream: decode_n_tokens EOS at iter={}", i)
            break

    del cur_token

    if new_tokens:
        remainder = torch.cat(new_tokens, dim=1).detach().clone()
        if do_stream_log:
            logger.info(
                "stream: decode_n_tokens yielding remainder shape={}", remainder.shape
            )
        yield remainder


@torch.no_grad()
def generate(
    *,
    model: DualARTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    num_samples: int = 1,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    cache_len = _cache_max_seq_len(model)

    if T >= cache_len:
        raise ValueError(
            f"Input sequence length {T} exceeds cache_max_seq_len {cache_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > cache_len:
            max_new_tokens = cache_len - T

        T_new = T + max_new_tokens
    else:
        T_new = cache_len
        max_new_tokens = T_new - T

    device = prompt.device
    dtype = next(model.parameters()).dtype

    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=cache_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True

    codebook_dim = 1 + model.config.num_codebooks

    input_pos = torch.arange(0, T, device=device, dtype=torch.long)
    empty = torch.empty((codebook_dim, cache_len), dtype=prompt.dtype, device=device)
    empty[:, :T] = prompt
    seq = empty

    temp_val = sampling_kwargs.get("temperature", 1.0)
    top_p_val = sampling_kwargs.get("top_p", 0.9)
    top_k_val = sampling_kwargs.get("top_k", 30)

    temperature = torch.tensor(temp_val, device=device, dtype=dtype)
    top_p = torch.tensor(top_p_val, device=device, dtype=dtype)

    vocab_size = model.config.vocab_size
    semantic_logit_bias = torch.full(
        (1, 1, vocab_size), float("-inf"), device=device, dtype=dtype
    )

    semantic_logit_bias[
        0, 0, model.config.semantic_begin_id : model.config.semantic_end_id + 1
    ] = 0.0
    semantic_logit_bias[0, 0, model.tokenizer.get_token_id(IM_END_TOKEN)] = 0.0

    prefill_decode = decode_one_token_ar

    x_prefill = prompt.view(1, codebook_dim, -1)
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "mark_dynamic"):
        torch._dynamo.mark_dynamic(x_prefill, 2, min=1, max=cache_len)
        torch._dynamo.mark_dynamic(input_pos, 0, min=1, max=cache_len)

    first_token = prefill_decode(
        model,
        x_prefill,
        input_pos,
        temperature,
        top_p,
        top_k_val,
        semantic_logit_bias,
        audio_masks,
        audio_parts,
    )
    seq[:, T : T + 1] = first_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    stream_chunk_size = sampling_kwargs.pop("stream_chunk_size", None)
    initial_stream_chunk_size = sampling_kwargs.pop("initial_stream_chunk_size", None)

    if stream_chunk_size is not None and initial_stream_chunk_size is None:
        initial_stream_chunk_size = 8

    compile = sampling_kwargs.pop("compile", False)

    decode_iter = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k_val,
        semantic_logit_bias=semantic_logit_bias,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
        stream_chunk_size=stream_chunk_size,
        initial_stream_chunk_size=initial_stream_chunk_size,
        compile=compile,
    )

    if stream_chunk_size is None:
        x = next(iter(decode_iter))
        seq = seq[:, : T + 1 + x.size(1)]
        seq[:, T + 1 :] = x
        del first_token, x, prompt, empty, input_pos
        return seq

    def _stream():
        if initial_stream_chunk_size is not None and initial_stream_chunk_size > 1:
            try:
                first_rest = next(decode_iter)
            except StopIteration:
                yield first_token
                return
            yield torch.cat([first_token, first_rest], dim=1)
        else:
            yield first_token

        yield from decode_iter

    return _stream()


def init_model(checkpoint_path, device, precision, compile=False):
    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)

    model = model.to(device=device, dtype=precision)
    logger.info("Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        eager_decode_one_token = decode_one_token_ar
        decode_one_token = eager_decode_one_token
        logger.info("Using DualARTransformer")
    else:
        raise ValueError("Unsupported model type")

    model.fixed_temperature = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_top_p = torch.tensor(0.7, device=device, dtype=torch.float)
    model.fixed_repetition_penalty = torch.tensor(1.5, device=device, dtype=torch.float)

    model._cache_setup_done = False

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="default" if torch.cuda.is_available() else None,
            fullgraph=True,
            dynamic=True,
        )

    return model.eval(), decode_one_token, eager_decode_one_token


@torch.inference_mode()
def load_codec_model(codec_checkpoint_path, device, precision=torch.bfloat16):
    """Load the DAC codec model for audio encoding/decoding."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config_path = Path(__file__).parent.parent.parent / "configs" / "modded_dac_vq.yaml"
    cfg = OmegaConf.load(str(config_path))
    codec = instantiate(cfg)

    state_dict = torch.load(codec_checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }
    codec.load_state_dict(state_dict, strict=False)
    codec.eval()
    codec.to(device=device, dtype=precision)
    return codec


@torch.inference_mode()
def encode_audio(audio_path, codec, device):
    """Encode an audio file to VQ codes."""
    import torchaudio

    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav.to(device), sr, codec.sample_rate)[0]

    model_dtype = next(codec.parameters()).dtype
    audios = wav[None, None].to(dtype=model_dtype)
    audio_lengths = torch.tensor([len(wav)], device=device, dtype=torch.long)

    indices, feature_lengths = codec.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]]


@torch.inference_mode()
def decode_to_audio(codes, codec):
    """Decode VQ codes to audio waveform."""
    audio = codec.from_indices(codes[None])
    return audio[0, 0]


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def split_text_by_speaker(text: str) -> list[str]:
    """
    Split text into turns based on <|speaker:X|> tags.
    """
    pattern = r"(<\|speaker:\d+\|>)"
    parts = re.split(pattern, text)

    turns = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if re.match(pattern, part):
            if i + 1 < len(parts):
                turn = part + parts[i + 1]
                turns.append(turn.strip())
                i += 2
            else:
                turns.append(part)
                i += 1
        else:
            i += 1

    return turns


def split_text_by_bytes(text: str, max_bytes: int) -> list[str]:
    """
    Split text into chunks of at most max_bytes (UTF-8). Used when there are
    no speaker turns so that chunk_length still controls batch size for streaming.
    """
    if max_bytes <= 0 or len(text.encode("utf-8")) <= max_bytes:
        return [text] if text.strip() else []
    chunks = []
    remaining = text
    while remaining:
        encoded = remaining.encode("utf-8")
        if len(encoded) <= max_bytes:
            chunks.append(remaining)
            break
        cut = max_bytes
        while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
            cut -= 1
        chunks.append(encoded[:cut].decode("utf-8", errors="replace"))
        remaining = encoded[cut:].decode("utf-8", errors="replace")
    return chunks


def group_turns_into_batches(
    turns: list[str], max_speakers: int = 3, max_bytes: int = 300
) -> list[str]:
    """
    Group turns into batches based on speaker count or byte limit.
    """
    batches = []
    current_batch = []
    current_bytes = 0

    for turn in turns:
        turn_bytes = len(turn.encode("utf-8"))

        would_exceed_speakers = len(current_batch) >= max_speakers
        would_exceed_bytes = current_bytes + turn_bytes > max_bytes and current_batch

        if would_exceed_speakers or would_exceed_bytes:
            batches.append("\n".join(current_batch))
            current_batch = [turn]
            current_bytes = turn_bytes
        else:
            current_batch.append(turn)
            current_bytes += turn_bytes

    if current_batch:
        batches.append("\n".join(current_batch))

    return batches


def generate_long(
    *,
    model,
    device: Union[str, torch.device],
    decode_one_token: Callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.9,
    top_k: int = 30,
    repetition_penalty: float = 1.1,
    temperature: float = 1.0,
    compile: bool = False,
    iterative_prompt: bool = True,
    chunk_length: int = 512,
    prompt_text: Optional[Union[str, list[str]]] = None,
    prompt_tokens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
    stream_tokens: bool = False,
    stream_chunk_size: int = 20,
    initial_stream_chunk_size: Optional[int] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError(
            "Model tokenizer is not initialized. Ensure checkpoint directory contains tokenizer files."
        )

    debug_visualize_prompt = _env_flag("FISH_DEBUG_VISUALIZE_PROMPT", False)

    use_prompt = bool(prompt_text) and bool(prompt_tokens)
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    if use_prompt:
        assert len(prompt_text) == len(
            prompt_tokens
        ), "Prompt text and tokens must have the same length"

    if prompt_tokens:
        prompt_tokens = [i.cpu() for i in prompt_tokens]

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    max_length = _cache_max_seq_len(model)

    base_conversation = Conversation()

    if use_prompt:
        tagged_prompt_text = []
        for i, t in enumerate(prompt_text):
            if not re.search(r"<\|speaker:\d+\|>", t):
                tagged_prompt_text.append(f"<|speaker:{i}|>{t}")
            else:
                tagged_prompt_text.append(t)

        system_parts = [
            TextPart(
                text="convert the provided text to speech reference to the following:\n\nText:\n",
                cal_loss=False,
            ),
        ]
        reference_text = "\n".join(tagged_prompt_text)
        system_parts.append(TextPart(text=reference_text, cal_loss=False))
        system_parts.append(TextPart(text="\n\nSpeech:\n", cal_loss=False))
        all_codes = torch.cat([c for c in prompt_tokens], dim=1)
        system_parts.append(VQPart(codes=all_codes, cal_loss=False))
    else:
        system_parts = [
            TextPart(text="convert the provided text to speech", cal_loss=False)
        ]

    base_conversation.append(
        Message(
            role="system",
            parts=system_parts,
            cal_loss=False,
            add_im_start=True,
            add_im_end=True,
        )
    )

    turns = split_text_by_speaker(text)
    if stream_tokens:
        batches = [text]
        logger.info("Token streaming: single batch (no text split)")
        if initial_stream_chunk_size is None:
            initial_stream_chunk_size = 8
    elif turns:
        batches = group_turns_into_batches(
            turns, max_speakers=5, max_bytes=chunk_length
        )
    else:
        batches = split_text_by_bytes(text, chunk_length)

    logger.info(f"Split into {len(turns)} turns, grouped into {len(batches)} batches")

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        conversation = deepcopy(base_conversation)

        for batch_idx, batch_text in enumerate(batches):
            logger.info(
                f"--- Sample {sample_idx}, Batch {batch_idx} "
                f"({len(batch_text.encode('utf-8'))} bytes) ---"
            )
            logger.info(f"Batch text: {batch_text}")

            conversation.append(
                Message(
                    role="user",
                    parts=[TextPart(text=batch_text, cal_loss=False)],
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            conversation_gen = deepcopy(conversation)
            conversation_gen.append(
                Message(
                    role="assistant",
                    parts=[],
                    cal_loss=False,
                    modality="voice",
                    add_im_start=True,
                    add_im_end=False,
                )
            )

            if debug_visualize_prompt:
                logger.info("Visualizing prompt structure:")
                conversation_gen.visualize(
                    tokenizer,
                    merge_audio_tokens=True,
                    merge_semantic_tokens=True,
                )

            encoded, audio_masks, audio_parts = conversation_gen.encode_for_inference(
                tokenizer, num_codebooks=model.config.num_codebooks
            )

            logger.info(f"Encoded prompt shape: {encoded.shape}")
            if audio_parts is not None:
                logger.info(f"Audio parts shape: {audio_parts.shape}")
            if audio_masks is not None:
                logger.info(
                    f"Audio masks non-zero count: {torch.count_nonzero(audio_masks)}"
                )

            prompt_len = encoded.size(1)
            if prompt_len > max_length:
                raise ValueError(
                    f"Prompt length {prompt_len} exceeds KV cache size {max_length}. "
                    f"Increase FISH_CACHE_MAX_SEQ_LEN (e.g. 1024 or 2048) or use a shorter reference."
                )
            if prompt_len + max_new_tokens > max_length:
                max_new_tokens = max_length - prompt_len
                logger.info(
                    "Capping max_new_tokens to {} so prompt+gen fits in cache (prompt={}, cache={})",
                    max_new_tokens,
                    prompt_len,
                    max_length,
                )

            cap_env = os.environ.get("FISH_MAX_NEW_TOKENS_CAP", "").strip()
            if cap_env:
                try:
                    cap = int(cap_env)
                    if cap >= 1 and max_new_tokens > cap:
                        max_new_tokens = cap
                        logger.info(
                            "Capping max_new_tokens to {} (FISH_MAX_NEW_TOKENS_CAP) for VRAM safety",
                            max_new_tokens,
                        )
                except ValueError:
                    pass

            encoded = encoded.to(device=device)
            prompt_length = encoded.size(1)

            if stream_tokens:
                logger.info(
                    "stream: generate_long starting token stream batch_idx={} stream_chunk_size={} initial_stream_chunk_size={}",
                    batch_idx,
                    stream_chunk_size,
                    initial_stream_chunk_size,
                )
                gen = generate(
                    model=model,
                    prompt=encoded,
                    max_new_tokens=max_new_tokens,
                    audio_masks=audio_masks,
                    audio_parts=audio_parts,
                    decode_one_token=decode_one_token,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stream_chunk_size=stream_chunk_size,
                    initial_stream_chunk_size=initial_stream_chunk_size,
                    compile=compile,
                )
                codes_list: list[torch.Tensor] = []
                chunk_idx = 0
                for chunk in gen:
                    codes_chunk = chunk[1:, :].clone()
                    if chunk_idx < 3:
                        logger.info(
                            "stream: generate_long chunk_idx={} chunk.shape={} codes_chunk.shape={}",
                            chunk_idx,
                            chunk.shape,
                            codes_chunk.shape,
                        )
                    if (codes_chunk >= 0).all():
                        yield GenerateResponse(
                            action="sample", codes=codes_chunk, text=batch_text
                        )
                    codes_list.append(chunk)
                    chunk_idx += 1

                logger.info(
                    "stream: generate_long finished chunk_idx={} total_chunks={}",
                    chunk_idx,
                    len(codes_list),
                )

                codes = (
                    torch.cat(codes_list, dim=1)[1:, :].clone() if codes_list else None
                )
                if codes is not None:
                    conversation.append(
                        Message(
                            role="assistant",
                            parts=[VQPart(codes=codes.cpu(), cal_loss=False)],
                            cal_loss=False,
                            modality="voice",
                            add_im_start=True,
                            add_im_end=True,
                        )
                    )

                codes_list.clear()
                del codes_list
                if codes is not None:
                    del codes

                if sample_idx == 0 and batch_idx == 0 and compile:
                    logger.info(
                        f"Compilation time: {time.perf_counter() - t0:.2f} seconds"
                    )

                del encoded
            else:
                y = generate(
                    model=model,
                    prompt=encoded,
                    max_new_tokens=max_new_tokens,
                    audio_masks=audio_masks,
                    audio_parts=audio_parts,
                    decode_one_token=decode_one_token,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    compile=compile,
                )

                if sample_idx == 0 and batch_idx == 0 and compile:
                    logger.info(
                        f"Compilation time: {time.perf_counter() - t0:.2f} seconds"
                    )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                t_batch = time.perf_counter() - t0
                tokens_generated = y.size(1) - prompt_length
                tokens_sec = tokens_generated / t_batch if t_batch > 0 else 0
                logger.info(
                    f"Batch {batch_idx}: Generated {tokens_generated} tokens in "
                    f"{t_batch:.02f} seconds, {tokens_sec:.02f} tokens/sec"
                )
                logger.info(
                    f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
                )

                codes = y[1:, prompt_length:-1].clone()
                assert (codes >= 0).all(), f"Negative code found: {codes}"

                conversation.append(
                    Message(
                        role="assistant",
                        parts=[VQPart(codes=codes.cpu(), cal_loss=False)],
                        cal_loss=False,
                        modality="voice",
                        add_im_start=True,
                        add_im_end=True,
                    )
                )

                yield GenerateResponse(action="sample", codes=codes, text=batch_text)
                del y, encoded

        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[Union[GenerateResponse, Exception]] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def _model_param_memory_gb(module: torch.nn.Module) -> tuple[float, int]:
    """Return (param_memory_gb, param_count) for a module (weights only)."""
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
        profile_cadence = os.getenv("FISH_PROFILE_INFERENCE", "0") in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }

        model, decode_one_token, eager_decode_one_token = init_model(
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

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            req_tag = str(kwargs.pop("req_tag", "na"))
            ack_queue = kwargs.pop("ack_queue", None)
            cancel_event = kwargs.pop("cancel_event", None)
            response_queue = item.response_queue
            t_req_start = time.perf_counter()
            t_last_put = t_req_start
            stream_tokens = kwargs.get("stream_tokens", False)
            streaming_eager_decode = stream_tokens and _env_flag(
                "FISH_STREAMING_EAGER_DECODE", False
            )
            if streaming_eager_decode:
                kwargs["compile"] = False
            put_count = 0
            cleanup_mode = _normalize_cleanup_mode(
                kwargs.pop("cleanup_mode", "request_end")
            )
            control = str(kwargs.pop("control", "")).strip().lower()
            stream_empty_cache = kwargs.pop("stream_empty_cache", None)
            if stream_empty_cache is None:
                stream_empty_cache = _env_flag("FISH_STREAM_EMPTY_CACHE", False)

            ack_timeout_sec = max(
                0.1, _env_float("FISH_STREAM_ACK_TIMEOUT_SEC", 15.0)
            )
            did_run_cleanup = False
            abort_requested = False

            if control == "cleanup":
                try:
                    logger.info(
                        "worker: received control=cleanup req={} cleanup_mode={}",
                        req_tag,
                        cleanup_mode,
                    )
                    _run_request_cleanup(
                        model=model,
                        cleanup_mode="request_end",
                        req_tag=req_tag,
                        reason="control_cleanup",
                    )
                    did_run_cleanup = True
                    _queue_put_with_cancel(
                        response_queue,
                        WrappedGenerateResponse(
                            status="success",
                            response=GenerateResponse(action="next"),
                        ),
                        cancel_event=cancel_event,
                    )
                except Exception as e:
                    logger.error(
                        "worker: control cleanup failed req={} err={}",
                        req_tag,
                        e,
                        exc_info=True,
                    )
                    _queue_put_with_cancel(
                        response_queue,
                        WrappedGenerateResponse(status="error", response=e),
                        cancel_event=cancel_event,
                    )
                continue

            # В режиме session_idle не делаем heavy cleanup всегда,
            # но если VRAM уже опасно мала — чистим до старта запроса.
            _maybe_force_cleanup_before_request(
                model=model,
                req_tag=req_tag,
                cleanup_mode=cleanup_mode,
            )

            if stream_tokens:
                logger.info(
                    "stream: worker got request req={} stream_chunk_size={} initial_stream_chunk_size={} cleanup_mode={}",
                    req_tag,
                    kwargs.get("stream_chunk_size"),
                    kwargs.get("initial_stream_chunk_size"),
                    cleanup_mode,
                )

            try:
                for chunk in generate_long(
                    model=model,
                    decode_one_token=(
                        eager_decode_one_token
                        if streaming_eager_decode
                        else decode_one_token
                    ),
                    **kwargs,
                ):
                    if cancel_event is not None and cancel_event.is_set():
                        abort_requested = True
                        logger.info(
                            "worker: cancel_event observed before queue_put req={} put_count={}",
                            req_tag,
                            put_count,
                        )
                        break

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
                            "queue_put req={} action={} delta_ms={:.1f} total_ms={:.1f} time_since_req_start_ms={:.1f}{}",
                            req_tag,
                            action,
                            delta_ms,
                            total_ms,
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

                    queued = _queue_put_with_cancel(
                        response_queue,
                        WrappedGenerateResponse(status="success", response=out),
                        cancel_event=cancel_event,
                    )
                    if not queued:
                        abort_requested = True
                        logger.info(
                            "worker: response_queue put cancelled req={} put_count={}",
                            req_tag,
                            put_count,
                        )
                        break

                    # Bounded pipeline backpressure:
                    # ждём подтверждение, что chunk принят CPU pipeline.
                    if ack_queue is not None:
                        ack = _wait_for_consumer_ack(
                            ack_queue,
                            cancel_event=cancel_event,
                            ack_timeout_sec=ack_timeout_sec,
                        )

                        if ack == ACK_ABORT:
                            abort_requested = True
                            logger.warning(
                                "worker: got abort ack req={} put_count={}",
                                req_tag,
                                put_count,
                            )
                            break

                    if ack_queue is not None and stream_empty_cache and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if abort_requested:
                    _run_request_cleanup(
                        model=model,
                        cleanup_mode=cleanup_mode,
                        req_tag=req_tag,
                        reason="pipeline_abort",
                    )
                else:
                    _run_request_cleanup(
                        model=model,
                        cleanup_mode=cleanup_mode,
                        req_tag=req_tag,
                        reason="request_success",
                    )
                did_run_cleanup = True

            except Exception as e:
                logger.error(
                    "stream: worker EXCEPTION req={} put_count={}: {}",
                    req_tag,
                    put_count,
                    e,
                    exc_info=True,
                )
                logger.error(traceback.format_exc())

                # Если consumer уже попросил abort, не надо ещё раз слать ошибку наружу.
                if not abort_requested and not (cancel_event and cancel_event.is_set()):
                    _queue_put_with_cancel(
                        response_queue,
                        WrappedGenerateResponse(status="error", response=e),
                        cancel_event=cancel_event,
                    )

            finally:
                if not did_run_cleanup:
                    try:
                        _run_request_cleanup(
                            model=model,
                            cleanup_mode=cleanup_mode,
                            req_tag=req_tag,
                            reason="request_finally",
                        )
                    except Exception as cleanup_err:
                        logger.warning(
                            "worker: cleanup failed req={} cleanup_mode={} err={}",
                            req_tag,
                            cleanup_mode,
                            cleanup_err,
                        )

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


@click.command()
@click.option(
    "--text",
    type=str,
    default="<|speaker:0|>你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option(
    "--prompt-audio",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.9)
@click.option("--top-k", type=int, default=30)
@click.option("--temperature", type=float, default=1.0)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/s2-pro",
)
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=300)
@click.option("--output-dir", type=Path, default="output")
def main(
    text: str,
    prompt_text: Optional[tuple[str, ...]],
    prompt_tokens: Optional[tuple[Path, ...]],
    prompt_audio: Optional[tuple[Path, ...]],
    output: Optional[Path],
    num_samples: int,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16

    if prompt_text and not prompt_audio and not prompt_tokens:
        raise ValueError(
            "--prompt-text requires either --prompt-audio or --prompt-tokens"
        )
    if prompt_text and prompt_tokens and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )
    if prompt_text and prompt_audio and len(prompt_text) != len(prompt_audio):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt audio ({len(prompt_audio)}) should be the same"
        )

    logger.info("Loading model ...")
    t0 = time.time()
    model, decode_one_token, _ = init_model(
        checkpoint_path, device, precision, compile=compile
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=_cache_max_seq_len(model),
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    codec = None
    codec_checkpoint = checkpoint_path / "codec.pth"

    prompt_tokens_list = None
    if prompt_audio:
        logger.info("Loading codec model for audio encoding...")
        codec = load_codec_model(codec_checkpoint, device, precision)
        prompt_tokens_list = [
            encode_audio(p, codec, device).cpu() for p in prompt_audio
        ]
        logger.info(f"Encoded {len(prompt_audio)} audio file(s) to VQ codes")
    elif prompt_tokens is not None:
        prompt_tokens_list = [torch.from_numpy(np.load(p)) for p in prompt_tokens]

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=list(prompt_text) if prompt_text else None,
        prompt_tokens=prompt_tokens_list,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                merged_codes = torch.cat(codes, dim=1)
                codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                np.save(codes_npy_path, merged_codes.cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")

                if output:
                    if codec is None:
                        logger.info("Loading codec model for audio decoding...")
                        codec = load_codec_model(codec_checkpoint, device, precision)
                    audio = decode_to_audio(merged_codes.to(device), codec)
                    import soundfile as sf

                    out_path = (
                        str(output)
                        if num_samples == 1
                        else str(output.with_stem(f"{output.stem}_{idx}"))
                    )
                    sf.write(out_path, audio.cpu().float().numpy(), codec.sample_rate)
                    logger.info(f"Saved audio to {out_path}")

            logger.info("Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from typing import Iterator, Optional, Tuple, cast

import torch
import torch._inductor.config
from loguru import logger
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.generation.sampling import (
    RAS_HIGH_TEMP,
    RAS_HIGH_TOP_P,
    RAS_WIN_SIZE,
    apply_repetition_penalty,
    sample,
)
from fish_speech.models.text2semantic.llama import BaseTransformer, DualARTransformer
from fish_speech.driver.config import load_runtime_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True


def _to_normal_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Copy tensor so it is not an inference tensor.
    Avoids CPU roundtrip for performance.
    """
    if t is None:
        return None
    return t.detach().clone()


def _iter_no_grad(iterator: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
    """
    Iterate a lazy generator under torch.no_grad().
    """
    with torch.no_grad():
        for item in iterator:
            yield item


def _use_sdpa_math() -> bool:
    return load_runtime_config().model.sdpa_math


def _cache_max_seq_len(model: BaseTransformer) -> int:
    configured = load_runtime_config().model.cache_max_seq_len
    return min(max(1, configured), model.config.max_seq_len)


def _model_im_end_id(model: BaseTransformer) -> int:
    im_end_id = int(getattr(model, "im_end_token_id", -1))
    if im_end_id < 0 or im_end_id >= model.config.vocab_size:
        raise RuntimeError("Model sampling buffers are missing a valid IM_END token id")
    return im_end_id


def _mask_logits_to_max_exclusive(
    logits: torch.Tensor,
    max_exclusive: int | None,
) -> torch.Tensor:
    """
    Restrict logits to ids [0, max_exclusive).

    Used for S2-Pro style DAC where the semantic row may need 4096 ids, while
    acoustic rows must stay inside 1024 ids. Masking here prevents invalid
    acoustic codes instead of allowing decoder-side clamp or late validation
    failures.
    """
    if max_exclusive is None:
        return logits

    max_exclusive = int(max_exclusive)
    vocab_size = int(logits.shape[-1])
    if max_exclusive <= 0 or max_exclusive >= vocab_size:
        return logits

    blocked = torch.arange(vocab_size, device=logits.device) >= max_exclusive
    while blocked.ndim < logits.ndim:
        blocked = blocked.unsqueeze(0)

    return logits.masked_fill(blocked, float("-inf"))


def _semantic_logit_bias(
    model: DualARTransformer,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Get cached semantic logit bias tensor.
    """
    if hasattr(model, "_semantic_logit_bias_cache"):
        cache = model._semantic_logit_bias_cache
        if cache.device == device and cache.dtype == dtype:
            return cache

    vocab_size = model.config.vocab_size
    semantic_logit_bias = torch.full(
        (1, 1, vocab_size), float("-inf"), device=device, dtype=dtype
    )

    semantic_logit_bias[0, 0, model.semantic_token_mask_for_sampling] = 0.0
    semantic_logit_bias[0, 0, _model_im_end_id(model)] = 0.0

    model._semantic_logit_bias_cache = semantic_logit_bias
    return semantic_logit_bias


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
    repetition_penalty: float = 1.0,
    disable_fast_first: bool = False,
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

    if not disable_fast_first and repetition_penalty != 1.0 and previous_tokens is not None:
        # Penalize only semantic tokens in the window
        # We assume previous_tokens[0] contains main tokens (semantic + IM_END)
        window = previous_tokens[0]

        biased_logits = apply_repetition_penalty(
            biased_logits,
            window,
            repetition_penalty,
            valid_token_mask=model.repetition_valid_token_mask,
        )

    main_token_normal = sample(
        biased_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )[0]

    if not disable_fast_first:
        high_temp = torch.tensor(
            RAS_HIGH_TEMP, device=temperature.device, dtype=temperature.dtype
        )
        high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=top_p.device, dtype=top_p.dtype)
        main_token_high = sample(
            biased_logits, temperature=high_temp, top_p=high_top_p, top_k=top_k
        )[0]

        if previous_tokens is not None:
            in_window = (previous_tokens[0] == main_token_normal).any()
            is_semantic = model.semantic_token_mask_for_sampling[main_token_normal]

            should_use_high = in_window & is_semantic
            main_token_normal = torch.where(
                should_use_high, main_token_high, main_token_normal
            )

    codebooks = [main_token_normal]

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)

    a = model.semantic_token_to_code_for_sampling[main_token_normal]
    # For non-semantic tokens (like IM_END), a will be -1.
    # We use 0 as a dummy code because codebooks after EOS are ignored.
    a = torch.where(a >= 0, a, torch.zeros_like(a))

    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    acoustic_codebook_size = getattr(model, "acoustic_codebook_size_for_sampling", None)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = _mask_logits_to_max_exclusive(
            logits,
            acoustic_codebook_size,
        )
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
    repetition_penalty: float = 1.0,
    stream_chunk_size: Optional[int] = None,
    initial_stream_chunk_size: Optional[int] = None,
    initial_token_chunk: Optional[torch.Tensor] = None,
    low_latency_first_audio: bool = False,
    compile: bool = False,
) -> Iterator[torch.Tensor]:
    """
    Generate tokens autoregressively.
    """

    cur_token = cur_token.detach().clone()
    input_pos = input_pos.detach().clone()
    temperature = temperature.detach().clone()
    top_p = top_p.detach().clone()
    semantic_logit_bias = semantic_logit_bias.detach().clone()

    if audio_masks is not None:
        audio_masks = audio_masks.detach().clone()
    if audio_parts is not None:
        audio_parts = audio_parts.detach().clone()

    previous_tokens = torch.full(
        (model.config.num_codebooks + 1, RAS_WIN_SIZE),
        -1,
        dtype=torch.long,
        device=cur_token.device,
    )

    new_tokens: list[torch.Tensor] = []
    first_chunk_emitted = stream_chunk_size is None

    if stream_chunk_size is not None:
        if initial_stream_chunk_size is None:
            initial_stream_chunk_size = stream_chunk_size
        initial_stream_chunk_size = max(1, int(initial_stream_chunk_size))
        stream_chunk_size = max(1, int(stream_chunk_size))

        if initial_token_chunk is not None:
            new_tokens.append(initial_token_chunk.detach().clone())

    im_end_id = _model_im_end_id(model)
    do_stream_log = stream_chunk_size is not None
    generated_so_far = 0

    for i in tqdm(range(num_new_tokens)):
        if do_stream_log and i < 3:
            logger.info(
                "stream: decode_n_tokens iter={} cur_token.shape={} input_pos={}",
                i,
                cur_token.shape,
                input_pos.shape,
            )
        disable_fast_first = (
            low_latency_first_audio
            and initial_stream_chunk_size is not None
            and generated_so_far < initial_stream_chunk_size
        )

        try:
            use_math = _use_sdpa_math()
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
                        repetition_penalty=repetition_penalty,
                        disable_fast_first=disable_fast_first,
                    )
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
                    repetition_penalty=repetition_penalty,
                    disable_fast_first=disable_fast_first,
                )
        except Exception as e:
            logger.exception(
                "stream: decode_n_tokens FAILED at iter={} (cur_token.shape={}): {}",
                i,
                cur_token.shape,
                e,
            )
            raise

        next_token = next_token.detach().clone()
        input_pos = (input_pos + 1).detach().clone()
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1).clone()

        # Update previous tokens for RAS window without inplace operations
        prev_col = next_token.view(model.config.num_codebooks + 1, -1)[:, :1]
        previous_tokens = torch.cat(
            [previous_tokens[:, 1:], prev_col], dim=1
        ).detach().clone()

        new_tokens.append(next_token)
        generated_so_far += 1

        if stream_chunk_size is not None:
            target_chunk_size = (
                initial_stream_chunk_size if not first_chunk_emitted else stream_chunk_size
            )
            if len(new_tokens) >= target_chunk_size:
                chunk_out = torch.cat(new_tokens, dim=1).detach().clone()
                if do_stream_log:
                    logger.info(
                        "stream: decode_n_tokens yielding chunk shape={} after iter={} target_chunk_size={}",
                        chunk_out.shape,
                        i,
                        target_chunk_size,
                    )
                yield chunk_out
                new_tokens = []
                first_chunk_emitted = True

        if (cur_token[0, 0, -1] == im_end_id).any():
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

    if num_samples != 1:
        raise ValueError("num_samples > 1 is not supported in this inference path")

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
    stream_chunk_size = sampling_kwargs.get("stream_chunk_size", None)

    if stream_chunk_size is None:
        empty = torch.empty((codebook_dim, cache_len), dtype=prompt.dtype, device=device)
        empty[:, :T] = prompt
        seq = empty
    else:
        seq = None

    temp_val = sampling_kwargs.get("temperature", 1.0)
    top_p_val = sampling_kwargs.get("top_p", 0.9)
    top_k_val = sampling_kwargs.get("top_k", 30)
    repetition_penalty = sampling_kwargs.get("repetition_penalty", 1.0)

    temperature = torch.tensor(temp_val, device=device, dtype=dtype)
    top_p = torch.tensor(top_p_val, device=device, dtype=dtype)

    semantic_logit_bias = _semantic_logit_bias(
        model,
        device=device,
        dtype=dtype,
    )

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
        repetition_penalty=repetition_penalty,
    )

    if seq is not None:
        seq[:, T : T + 1] = first_token

    input_pos = torch.tensor([T], device=device, dtype=torch.long)
    stream_chunk_size = sampling_kwargs.pop("stream_chunk_size", None)
    initial_stream_chunk_size = sampling_kwargs.pop("initial_stream_chunk_size", None)
    low_latency_first_audio = sampling_kwargs.pop("low_latency_first_audio", False)
    compile = sampling_kwargs.pop("compile", False)

    im_end_id = _model_im_end_id(model)
    if first_token[0, 0] == im_end_id:
        if stream_chunk_size is not None:
            return _iter_no_grad(iter([]))
        else:
            return seq[:, : T + 1]

    decode_iter = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max(0, max_new_tokens - 1),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k_val,
        semantic_logit_bias=semantic_logit_bias,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
        repetition_penalty=repetition_penalty,
        stream_chunk_size=stream_chunk_size,
        initial_stream_chunk_size=initial_stream_chunk_size,
        initial_token_chunk=first_token.clone() if stream_chunk_size is not None else None,
        low_latency_first_audio=low_latency_first_audio,
        compile=compile,
    )

    if stream_chunk_size is None:
        try:
            x = next(iter(decode_iter))
        except StopIteration:
            seq = seq[:, : T + 1]
            del first_token, prompt, input_pos
            return seq
        seq = seq[:, : T + 1 + x.size(1)]
        seq[:, T + 1 :] = x
        del first_token, x, prompt, input_pos
        return seq

    return _iter_no_grad(decode_iter)
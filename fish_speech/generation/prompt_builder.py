from __future__ import annotations

import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import torch
from loguru import logger

from fish_speech.content_sequence import TextPart, VQPart
from fish_speech.conversation import Conversation, Message
from fish_speech.driver.config import load_runtime_config
from fish_speech.generation.decode import _cache_max_seq_len, generate


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


def _as_list(value: Any | None) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _to_cpu_list(values: list[Any]) -> list[Any]:
    out = []
    for value in values:
        out.append(value.cpu() if hasattr(value, "cpu") else value)
    return out


def _count_code_frames(values: list[Any]) -> int:
    total = 0
    for value in values:
        if hasattr(value, "shape"):
            total += int(value.shape[-1])
    return total


def generate_committed_segments(
    *,
    model,
    device: Union[str, torch.device],
    decode_one_token: Callable,
    segments: Optional[list[str]] = None,
    text: str | None = None,
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
    continuation_text: Optional[Union[str, list[str]]] = None,
    continuation_tokens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
    stream_tokens: bool = False,
    stream_chunk_size: int = 8,
    initial_stream_chunk_size: int = 10,
):
    committed_segments = [segment for segment in (segments or []) if segment.strip()]
    if not committed_segments and text and text.strip():
        committed_segments = [text]

    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    runtime = load_runtime_config()

    # Reference prompt: speaker identity / voice reference.
    reference_texts = _as_list(prompt_text)
    reference_tokens = _as_list(prompt_tokens)
    use_reference_prompt = bool(reference_texts) and bool(reference_tokens)

    if reference_texts or reference_tokens:
        if not use_reference_prompt:
            raise ValueError(
                "Reference prompt is incomplete: prompt_text and prompt_tokens must be provided together"
            )
        if len(reference_texts) != len(reference_tokens):
            raise ValueError(
                "Prompt text and tokens must have the same length: "
                f"texts={len(reference_texts)} tokens={len(reference_tokens)}"
            )
        reference_tokens = _to_cpu_list(reference_tokens)

    # Continuation prompt: rolling acoustic history from previous commits.
    continuation_texts = _as_list(continuation_text)
    continuation_token_list = _as_list(continuation_tokens)
    use_continuation = bool(continuation_texts) and bool(continuation_token_list)

    if continuation_texts or continuation_token_list:
        if not use_continuation:
            raise ValueError(
                "Continuation is incomplete: continuation_text and continuation_tokens must be provided together"
            )
        if len(continuation_texts) != len(continuation_token_list):
            raise ValueError(
                "Continuation text and tokens must have the same length: "
                f"texts={len(continuation_texts)} tokens={len(continuation_token_list)}"
            )
        continuation_token_list = _to_cpu_list(continuation_token_list)

    logger.info(
        "generation_prompt_inputs: "
        "reference_text_count={} reference_token_count={} reference_token_frames={} "
        "continuation_text_count={} continuation_token_count={} continuation_token_frames={}",
        len(reference_texts),
        len(reference_tokens),
        _count_code_frames(reference_tokens),
        len(continuation_texts),
        len(continuation_token_list),
        _count_code_frames(continuation_token_list),
    )

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    max_length = _cache_max_seq_len(model)

    base_conversation = Conversation()

    if use_reference_prompt:
        tagged_prompt_text = []
        for i, item in enumerate(reference_texts):
            if not re.search(r"<\|speaker:\d+\|>", item):
                tagged_prompt_text.append(f"<|speaker:{i}|>{item}")
            else:
                tagged_prompt_text.append(item)

        system_parts = [
            TextPart(
                text="convert the provided text to speech reference to the following:\n\nText:\n",
                cal_loss=False,
            ),
        ]
        reference_text = "\n".join(tagged_prompt_text)
        system_parts.append(TextPart(text=reference_text, cal_loss=False))
        system_parts.append(TextPart(text="\n\nSpeech:\n", cal_loss=False))

        all_codes = torch.cat([codes for codes in reference_tokens], dim=1)
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

    if use_continuation:
        for old_text, old_codes in zip(continuation_texts, continuation_token_list):
            if not old_text or old_codes is None:
                continue

            base_conversation.append(
                Message(
                    role="user",
                    parts=[TextPart(text=old_text, cal_loss=False)],
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            base_conversation.append(
                Message(
                    role="assistant",
                    parts=[VQPart(codes=old_codes.cpu(), cal_loss=False)],
                    cal_loss=False,
                    modality="voice",
                    add_im_start=True,
                    add_im_end=True,
                )
            )

    logger.info("Generating {} committed segment(s)", len(committed_segments))

    requested_max_new_tokens = int(max_new_tokens or 0)

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        conversation = deepcopy(base_conversation)

        for batch_idx, batch_text in enumerate(committed_segments):
            logger.info(
                "--- Sample {}, Segment {} ({} bytes) ---",
                sample_idx,
                batch_idx,
                len(batch_text.encode("utf-8")),
            )
            logger.info("Segment text: {}", batch_text)

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

            logger.info("Visualizing prompt structure:")
            conversation_gen.visualize(
                tokenizer,
                merge_audio_tokens=True,
                merge_semantic_tokens=True,
            )

            encoded, audio_masks, audio_parts = conversation_gen.encode_for_inference(
                tokenizer, num_codebooks=model.config.num_codebooks
            )

            logger.info("Encoded prompt shape: {}", encoded.shape)
            if audio_parts is not None:
                logger.info("Audio parts shape: {}", audio_parts.shape)
            if audio_masks is not None:
                logger.info(
                    "Audio masks non-zero count: {}",
                    torch.count_nonzero(audio_masks),
                )

            prompt_len = encoded.size(1)
            available_new_tokens = max(0, max_length - prompt_len)

            logger.info(
                "prompt_budget: prompt_len={} cache={} requested_max_new_tokens={} "
                "available_new_tokens={} runtime_cap={} ref_turns={} continuation_turns={} "
                "continuation_frames={}",
                prompt_len,
                max_length,
                requested_max_new_tokens,
                available_new_tokens,
                runtime.model.max_new_tokens_cap,
                len(reference_texts),
                len(continuation_texts),
                _count_code_frames(continuation_token_list),
            )

            if prompt_len > max_length:
                raise ValueError(
                    f"Prompt length {prompt_len} exceeds KV cache size {max_length}. "
                    f"Increase cache_max_seq_len in config/runtime.json or use a shorter reference/continuation."
                )

            if requested_max_new_tokens <= 0:
                effective_max_new_tokens = available_new_tokens
            else:
                effective_max_new_tokens = min(
                    requested_max_new_tokens,
                    available_new_tokens,
                )

            clip_reasons = []
            if requested_max_new_tokens > 0 and available_new_tokens < requested_max_new_tokens:
                clip_reasons.append("cache")

            cap = runtime.model.max_new_tokens_cap
            before_runtime_cap = effective_max_new_tokens
            if cap >= 1 and effective_max_new_tokens > cap:
                effective_max_new_tokens = cap
                clip_reasons.append("runtime_cap")

            logger.info(
                "effective_generation_budget: prompt_len={} cache={} "
                "requested_max_new_tokens={} effective_max_new_tokens={} "
                "available_new_tokens={} runtime_cap={} clip_reasons={} "
                "ref_turns={} continuation_turns={} continuation_frames={} text_len={}",
                prompt_len,
                max_length,
                requested_max_new_tokens,
                effective_max_new_tokens,
                available_new_tokens,
                cap,
                clip_reasons,
                len(reference_texts),
                len(continuation_texts),
                _count_code_frames(continuation_token_list),
                len(batch_text),
            )

            if clip_reasons:
                logger.warning(
                    "generation_budget_clipped: reasons={} prompt_len={} cache={} "
                    "requested_max_new_tokens={} before_runtime_cap={} "
                    "effective_max_new_tokens={} available_new_tokens={} "
                    "text_len={} text={!r}",
                    clip_reasons,
                    prompt_len,
                    max_length,
                    requested_max_new_tokens,
                    before_runtime_cap,
                    effective_max_new_tokens,
                    available_new_tokens,
                    len(batch_text),
                    batch_text[:240],
                )

            if effective_max_new_tokens <= 0:
                raise ValueError(
                    f"No generation budget left: prompt_len={prompt_len}, cache={max_length}. "
                    f"Use shorter reference/continuation or increase cache_max_seq_len."
                )

            encoded = encoded.to(device=device)
            prompt_length = encoded.size(1)

            if stream_tokens:
                logger.info(
                    "stream: generate_committed_segments starting token stream "
                    "segment_idx={} initial_stream_chunk_size={} stream_chunk_size={} "
                    "effective_max_new_tokens={}",
                    batch_idx,
                    initial_stream_chunk_size,
                    stream_chunk_size,
                    effective_max_new_tokens,
                )

                gen = generate(
                    model=model,
                    prompt=encoded,
                    max_new_tokens=effective_max_new_tokens,
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
                total_code_frames = 0

                for chunk in gen:
                    main_tokens = chunk[0]
                    semantic_mask = (
                        (main_tokens >= tokenizer.semantic_begin_id)
                        & (main_tokens <= tokenizer.semantic_end_id)
                    )

                    if semantic_mask.any():
                        codes_chunk = chunk[1:, semantic_mask].clone()
                        code_frames = int(codes_chunk.shape[-1])
                        total_code_frames += code_frames

                        if chunk_idx < 3:
                            logger.info(
                                "stream: generate_committed_segments chunk_idx={} "
                                "chunk.shape={} codes_chunk.shape={} total_code_frames={}",
                                chunk_idx,
                                chunk.shape,
                                codes_chunk.shape,
                                total_code_frames,
                            )

                        yield GenerateResponse(
                            action="sample",
                            codes=codes_chunk,
                            text=batch_text,
                        )
                        codes_list.append(codes_chunk.cpu())

                    chunk_idx += 1

                logger.info(
                    "stream: generate_committed_segments finished chunk_idx={} "
                    "total_chunks={} total_code_frames={} effective_max_new_tokens={}",
                    chunk_idx,
                    len(codes_list),
                    total_code_frames,
                    effective_max_new_tokens,
                )

                truncation_margin = max(2, int(stream_chunk_size))
                if total_code_frames >= max(1, effective_max_new_tokens - truncation_margin):
                    logger.warning(
                        "generation_may_be_truncated: segment_idx={} text_len={} "
                        "code_frames={} effective_max_new_tokens={} prompt_len={} "
                        "cache={} text={!r}",
                        batch_idx,
                        len(batch_text),
                        total_code_frames,
                        effective_max_new_tokens,
                        prompt_len,
                        max_length,
                        batch_text[:240],
                    )

                codes = torch.cat(codes_list, dim=1).clone() if codes_list else None

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
                        "Compilation time: {:.2f} seconds",
                        time.perf_counter() - t0,
                    )

                del encoded

            else:
                y = generate(
                    model=model,
                    prompt=encoded,
                    max_new_tokens=effective_max_new_tokens,
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
                        "Compilation time: {:.2f} seconds",
                        time.perf_counter() - t0,
                    )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                t_batch = time.perf_counter() - t0
                tokens_generated = y.size(1) - prompt_length
                tokens_sec = tokens_generated / t_batch if t_batch > 0 else 0

                logger.info(
                    "Segment {}: Generated {} tokens in {:.2f} seconds, {:.2f} tokens/sec",
                    batch_idx,
                    tokens_generated,
                    t_batch,
                    tokens_sec,
                )
                logger.info(
                    "Bandwidth achieved: {:.2f} GB/s",
                    model_size * tokens_sec / 1e9,
                )

                generated = y[:, prompt_length:]
                main_tokens = generated[0]
                semantic_mask = (
                    (main_tokens >= tokenizer.semantic_begin_id)
                    & (main_tokens <= tokenizer.semantic_end_id)
                )
                codes = generated[1:, semantic_mask].clone()
                assert (codes >= 0).all(), f"Negative code found: {codes}"

                code_frames = int(codes.shape[-1])
                if tokens_generated >= max(1, effective_max_new_tokens - 2):
                    logger.warning(
                        "generation_may_be_truncated: segment_idx={} text_len={} "
                        "tokens_generated={} code_frames={} effective_max_new_tokens={} "
                        "prompt_len={} cache={} text={!r}",
                        batch_idx,
                        len(batch_text),
                        tokens_generated,
                        code_frames,
                        effective_max_new_tokens,
                        prompt_len,
                        max_length,
                        batch_text[:240],
                    )

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
                "GPU Memory used: {:.2f} GB",
                torch.cuda.max_memory_reserved() / 1e9,
            )

        yield GenerateResponse(action="next")
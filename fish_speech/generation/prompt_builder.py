from __future__ import annotations

import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Union

import torch
from loguru import logger

from fish_speech.content_sequence import TextPart, VQPart
from fish_speech.conversation import Conversation, Message
from fish_speech.generation.decode import _cache_max_seq_len, generate
from fish_speech.driver.config import load_runtime_config

@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None


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

    # Normalize prompt (reference)
    use_prompt = bool(prompt_text) and bool(prompt_tokens)
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    if use_prompt:
        assert len(prompt_text) == len(
            prompt_tokens
        ), "Prompt text and tokens must have the same length"
        prompt_tokens = [i.cpu() for i in prompt_tokens]

    # Normalize continuation
    use_continuation = bool(continuation_text) and bool(continuation_tokens)
    if use_continuation and isinstance(continuation_text, str):
        continuation_text = [continuation_text]
        continuation_tokens = [continuation_tokens]

    if use_continuation:
        assert len(continuation_text) == len(
            continuation_tokens
        ), "Continuation text and tokens must have the same length"
        continuation_tokens = [i.cpu() for i in continuation_tokens]

    logger.info(
        "generation_prompt_inputs: prompt_text_count={} prompt_token_count={} "
        "continuation_text_count={} continuation_token_count={} continuation_token_frames={}",
        len(prompt_text or []),
        len(prompt_tokens or []),
        len(continuation_text or []),
        len(continuation_tokens or []),
        sum(int(c.shape[-1]) for c in continuation_tokens or [] if hasattr(c, "shape")),
    )

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
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

    if use_continuation:
        for old_text, old_codes in zip(continuation_text, continuation_tokens):
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

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        conversation = deepcopy(base_conversation)

        for batch_idx, batch_text in enumerate(committed_segments):
            logger.info(
                f"--- Sample {sample_idx}, Segment {batch_idx} "
                f"({len(batch_text.encode('utf-8'))} bytes) ---"
            )
            logger.info(f"Segment text: {batch_text}")

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

            logger.info(f"Encoded prompt shape: {encoded.shape}")
            if audio_parts is not None:
                logger.info(f"Audio parts shape: {audio_parts.shape}")
            if audio_masks is not None:
                logger.info(
                    f"Audio masks non-zero count: {torch.count_nonzero(audio_masks)}"
                )

            prompt_len = encoded.size(1)

            logger.info(
                "prompt_budget: prompt_len={} cache={} max_new_tokens={} ref_turns={} continuation_turns={} continuation_frames={}",
                prompt_len,
                max_length,
                max_new_tokens,
                len(prompt_text or []),
                len(continuation_text or []),
                sum(
                    int(c.shape[-1])
                    for c in continuation_tokens or []
                    if hasattr(c, "shape")
                ),
            )

            if prompt_len > max_length:
                raise ValueError(
                    f"Prompt length {prompt_len} exceeds KV cache size {max_length}. "
                    f"Increase cache_max_seq_len in config/runtime.json or use a shorter reference."
                )
            if prompt_len + max_new_tokens > max_length:
                max_new_tokens = max_length - prompt_len
                logger.info(
                    "Capping max_new_tokens to {} so prompt+gen fits in cache (prompt={}, cache={})",
                    max_new_tokens,
                    prompt_len,
                    max_length,
                )

            cap = runtime.model.max_new_tokens_cap
            if cap >= 1 and max_new_tokens > cap:
                max_new_tokens = cap
                logger.info(
                    "Capping max_new_tokens to {} (runtime.model.max_new_tokens_cap) for VRAM safety",
                    max_new_tokens,
                )

            encoded = encoded.to(device=device)
            prompt_length = encoded.size(1)

            if stream_tokens:
                logger.info(
                    "stream: generate_committed_segments starting token stream segment_idx={} initial_stream_chunk_size={} stream_chunk_size={}",
                    batch_idx,
                    initial_stream_chunk_size,
                    stream_chunk_size,
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
                    main_tokens = chunk[0]
                    semantic_mask = (
                        (main_tokens >= tokenizer.semantic_begin_id)
                        & (main_tokens <= tokenizer.semantic_end_id)
                    )

                    if semantic_mask.any():
                        codes_chunk = chunk[1:, semantic_mask].clone()
                        if chunk_idx < 3:
                            logger.info(
                                "stream: generate_committed_segments chunk_idx={} chunk.shape={} codes_chunk.shape={}",
                                chunk_idx,
                                chunk.shape,
                                codes_chunk.shape,
                            )
                        yield GenerateResponse(
                            action="sample", codes=codes_chunk, text=batch_text
                        )
                        codes_list.append(codes_chunk.cpu())
                    chunk_idx += 1

                logger.info(
                    "stream: generate_committed_segments finished chunk_idx={} total_chunks={}",
                    chunk_idx,
                    len(codes_list),
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
                    f"Segment {batch_idx}: Generated {tokens_generated} tokens in "
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

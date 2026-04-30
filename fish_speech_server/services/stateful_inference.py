from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

import torch
from loguru import logger

from fish_speech import DriverSynthesisRequest, DriverTokenChunkEvent, FishSpeechDriver
from fish_speech_server.api.utils import inference_async
from fish_speech_server.services.continuation import crop_codes_tail
from fish_speech_server.services.synthesis_context import (
    SynthesisTurn,
    estimate_code_frames,
)

def _codes_to_cpu_2d(codes: Any) -> torch.Tensor | None:
    if codes is None:
        return None

    if torch.is_tensor(codes):
        tensor = codes.detach().cpu().long()
    else:
        tensor = torch.tensor(codes, dtype=torch.long)

    while tensor.dim() > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.dim() != 2:
        raise ValueError(
            f"Expected codes with 2 dims [codebooks, frames], got {tuple(tensor.shape)}"
        )

    return tensor.contiguous()


def _concat_code_chunks(chunks: list[Any]) -> torch.Tensor | None:
    tensors: list[torch.Tensor] = []

    for chunk in chunks:
        tensor = _codes_to_cpu_2d(chunk)
        if tensor is not None and tensor.numel() > 0:
            tensors.append(tensor)

    if not tensors:
        return None

    codebook_count = tensors[0].shape[0]
    for tensor in tensors:
        if tensor.shape[0] != codebook_count:
            raise ValueError(
                f"Mismatched codebook count: expected {codebook_count}, got {tensor.shape[0]}"
            )

    return torch.cat(tensors, dim=-1).cpu().contiguous()

if TYPE_CHECKING:
    from fish_speech_server.schema import StatefulTTSRequest
    from fish_speech_server.services.synthesis_context import SynthesisContext


async def stateful_inference_async(
    req: DriverSynthesisRequest | StatefulTTSRequest,
    driver: FishSpeechDriver,
    context: SynthesisContext,
    original_req: StatefulTTSRequest | None = None,
) -> AsyncGenerator[bytes, None]:
    """
    A wrapper around inference_async that updates the synthesis context history
    upon successful completion of the audio stream.
    """

    pcm_bytes = 0
    collected_codes: list[Any] = []
    created_at = time.time()

    # We yield chunks from the underlying inference engine
    async for event in inference_async(req, driver, yield_tokens=True):
        if isinstance(event, bytes):
            pcm_bytes += len(event)
            yield event
        elif isinstance(event, DriverTokenChunkEvent):
            if event.codes is not None:
                collected_codes.append(event.codes)
        else:
            # Skip or handle other non-bytes, non-token events if any
            pass

    # If we reached here, the generation finished successfully.
    # Note: inference_async might raise HTTPException or other errors,
    # which will propagate and skip this part.

    completed_at = time.time()

    # Use original_req if provided for metadata
    meta_req = original_req if original_req is not None else req

    # Concatenate collected codes if any
    final_codes = None
    code_frames = 0

    if collected_codes:
        try:
            final_codes = _concat_code_chunks(collected_codes)

            if final_codes is not None and context.max_history_code_frames > 0:
                final_codes = crop_codes_tail(
                    final_codes,
                    context.max_history_code_frames,
                )

            code_frames = estimate_code_frames(final_codes)

        except Exception as e:
            logger.warning(f"Failed to concatenate collected codes: {e}")
            final_codes = None
            code_frames = 0

    # Create a new turn for the history
    turn = SynthesisTurn(
        commit_seq=getattr(meta_req, "commit_seq", 0),
        text=meta_req.text,
        reason=getattr(meta_req, "commit_reason", "unknown"),
        created_at=created_at,
        completed_at=completed_at,
        pcm_bytes=pcm_bytes,
        codes=final_codes,
        code_frames=code_frames,
    )

    context.append_turn(turn)

    logger.info(
        "[SYNTH_CONTEXT] updated session={} commit_seq={} has_codes={} "
        "collected_code_chunks={} code_frames={} pcm_bytes={} "
        "history_turns={} history_with_codes={} history_code_frames={}",
        context.synthesis_session_id[:8],
        turn.commit_seq,
        final_codes is not None,
        len(collected_codes),
        code_frames,
        pcm_bytes,
        len(context.history),
        context.history_with_codes_count(),
        context.history_code_frames(),
    )

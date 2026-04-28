from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, AsyncGenerator

import torch
from loguru import logger

from fish_speech import DriverSynthesisRequest, DriverTokenChunkEvent, FishSpeechDriver
from fish_speech_server.api.utils import inference_async
from fish_speech_server.services.synthesis_context import (
    SynthesisTurn,
    estimate_code_frames,
)

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
            # In real environment, collected_codes is a list of torch.Tensors
            # with shape (codebooks, frames)
            if all(hasattr(c, "shape") for c in collected_codes):
                final_codes = torch.cat(collected_codes, dim=1).cpu()
                code_frames = estimate_code_frames(final_codes)
            else:
                # Fallback or mixed types (unlikely in practice)
                # If they are not tensors, we still try to move them to CPU if they are objects
                final_codes = []
                for c in collected_codes:
                    if hasattr(c, "cpu"):
                        final_codes.append(c.cpu())
                    else:
                        final_codes.append(c)
                code_frames = sum(estimate_code_frames(c) for c in collected_codes)
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
        "[SYNTH_CONTEXT] updated session=%s commit_seq=%s has_codes=%s "
        "collected_code_chunks=%s code_frames=%s pcm_bytes=%s "
        "history_turns=%s history_with_codes=%s history_code_frames=%s",
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

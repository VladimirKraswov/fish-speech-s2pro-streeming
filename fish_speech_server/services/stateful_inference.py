from __future__ import annotations

import time
from typing import TYPE_CHECKING, AsyncGenerator

from fish_speech import FishSpeechDriver
from fish_speech_server.api.utils import inference_async
from fish_speech_server.services.synthesis_context import SynthesisTurn

if TYPE_CHECKING:
    from fish_speech_server.schema import StatefulTTSRequest
    from fish_speech_server.services.synthesis_context import SynthesisContext


async def stateful_inference_async(
    req: StatefulTTSRequest,
    driver: FishSpeechDriver,
    context: SynthesisContext,
) -> AsyncGenerator[bytes, None]:
    """
    A wrapper around inference_async that updates the synthesis context history
    upon successful completion of the audio stream.
    """

    pcm_bytes = 0
    created_at = time.time()

    # We yield chunks from the underlying inference engine
    async for chunk in inference_async(req, driver):
        if chunk:
            # We skip the WAV header if it's the first chunk of a WAV stream
            # but for simplicity in counting PCM bytes, let's just count everything yielded
            # for now, or we could be more precise.
            # In fish-speech-server, inference_async(req, driver) yields
            # wav_chunk_header(...) first if req.streaming and req.format == 'wav'.
            pcm_bytes += len(chunk)
            yield chunk

    # If we reached here, the generation finished successfully.
    # Note: inference_async might raise HTTPException or other errors,
    # which will propagate and skip this part.

    completed_at = time.time()

    # Create a new turn for the history
    turn = SynthesisTurn(
        commit_seq=req.commit_seq,
        text=req.text,
        reason=req.commit_reason,
        created_at=created_at,
        completed_at=completed_at,
        pcm_bytes=pcm_bytes,
        # codes=None for now as per Step 2 instructions
        codes=None,
        code_frames=0,
    )

    context.append_turn(turn)

import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from fish_speech_server.schema import StatefulTTSRequest
from fish_speech_server.services.synthesis_store import SynthesisSessionStore
from fish_speech_server.services.stateful_inference import stateful_inference_async


class TestStatefulInference(unittest.IsolatedAsyncioTestCase):
    async def test_stateful_inference_updates_history(self):
        # Setup mocks
        driver = MagicMock()
        store = SynthesisSessionStore()
        ctx = await store.create(reference_id="test-ref")

        req = StatefulTTSRequest(
            text="Hello world",
            synthesis_session_id=ctx.synthesis_session_id,
            commit_seq=1,
            commit_reason="test",
            streaming=True,
            reference_id="test-ref"
        )

        # Mock inference_async to yield some fake audio chunks
        # We need to mock the module where it's imported or used.
        # Since stateful_inference_async imports it from fish_speech_server.api.utils,
        # we can mock that.

        fake_chunks = [b"chunk1", b"chunk2"]

        async def fake_inference_async(*args, **kwargs):
            for chunk in fake_chunks:
                yield chunk

        with unittest.mock.patch("fish_speech_server.services.stateful_inference.inference_async", side_effect=fake_inference_async):
            # Run the stateful inference
            yielded_chunks = []
            async for chunk in stateful_inference_async(req, driver, ctx):
                yielded_chunks.append(chunk)

            # Verify chunks were yielded
            self.assertEqual(yielded_chunks, fake_chunks)

            # Verify history was updated
            self.assertEqual(len(ctx.history), 1)
            turn = ctx.history[0]
            self.assertEqual(turn.text, "Hello world")
            self.assertEqual(turn.commit_seq, 1)
            self.assertEqual(turn.pcm_bytes, sum(len(c) for c in fake_chunks))
            self.assertIsNone(turn.codes)
            self.assertEqual(turn.code_frames, 0)

    async def test_stateful_inference_failure_does_not_update_history(self):
        driver = MagicMock()
        store = SynthesisSessionStore()
        ctx = await store.create(reference_id="test-ref")

        req = StatefulTTSRequest(
            text="Hello world",
            synthesis_session_id=ctx.synthesis_session_id,
            commit_seq=1,
            streaming=True
        )

        async def failing_inference_async(*args, **kwargs):
            yield b"chunk1"
            raise RuntimeError("Inference failed")

        with unittest.mock.patch("fish_speech_server.services.stateful_inference.inference_async", side_effect=failing_inference_async):
            with self.assertRaises(RuntimeError):
                async for chunk in stateful_inference_async(req, driver, ctx):
                    pass

            # Verify history was NOT updated because of the exception
            self.assertEqual(len(ctx.history), 0)


if __name__ == "__main__":
    unittest.main()

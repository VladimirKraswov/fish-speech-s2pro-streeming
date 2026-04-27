import unittest
import asyncio
import time
from fish_speech_server.services.synthesis_store import SynthesisSessionStore


class TestSynthesisSessionStore(unittest.IsolatedAsyncioTestCase):
    async def test_create_get_close(self):
        store = SynthesisSessionStore()
        ctx = await store.create(reference_id="ref1")
        self.assertIsNotNone(ctx.synthesis_session_id)
        self.assertEqual(ctx.reference_id, "ref1")

        # Get
        retrieved = await store.get(ctx.synthesis_session_id)
        self.assertEqual(retrieved.synthesis_session_id, ctx.synthesis_session_id)

        # Close
        closed = await store.close(ctx.synthesis_session_id)
        self.assertTrue(closed)

        # Get after close
        not_found = await store.get(ctx.synthesis_session_id)
        self.assertIsNone(not_found)

    async def test_touch(self):
        store = SynthesisSessionStore()
        ctx = await store.create(reference_id="ref1")
        old_updated_at = ctx.updated_at

        await asyncio.sleep(0.01)
        await store.get(ctx.synthesis_session_id, touch=True)
        self.assertGreater(ctx.updated_at, old_updated_at)

    async def test_cleanup_expired(self):
        # Set TTL very short for testing
        store = SynthesisSessionStore(ttl_sec=0)
        ctx = await store.create(reference_id="ref1")

        # Ensure it expires
        await asyncio.sleep(0.01)

        # Cleanup should remove it
        removed_count = await store.cleanup()
        self.assertEqual(removed_count, 1)

        not_found = await store.get(ctx.synthesis_session_id)
        self.assertIsNone(not_found)

    async def test_max_sessions(self):
        store = SynthesisSessionStore(max_sessions=2)
        await store.create(reference_id="ref1")
        await store.create(reference_id="ref2")

        with self.assertRaises(ValueError):
            await store.create(reference_id="ref3")

    async def test_stats(self):
        store = SynthesisSessionStore()
        await store.create(reference_id="ref1")

        stats = await store.stats()
        self.assertEqual(stats["active_sessions"], 1)
        self.assertEqual(len(stats["sessions"]), 1)
        self.assertEqual(stats["sessions"][0]["reference_id"], "ref1")
        self.assertIn("history_turns", stats["sessions"][0])


if __name__ == "__main__":
    unittest.main()

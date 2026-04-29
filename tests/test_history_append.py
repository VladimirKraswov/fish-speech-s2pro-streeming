import unittest
import torch
from fish_speech_server.schema import AppendHistoryRequest
from fish_speech_server.services.synthesis_store import SynthesisSessionStore
from fish_speech_server.services.synthesis_context import SynthesisTurn

class TestHistoryAppend(unittest.IsolatedAsyncioTestCase):
    async def test_append_history_to_context(self):
        store = SynthesisSessionStore()
        ctx = await store.create(reference_id="test-ref")

        codes = [[1, 2, 3], [4, 5, 6]]
        req = AppendHistoryRequest(
            text="Intro text",
            codes=codes,
            reason="cached_intro",
            commit_seq=0
        )

        codes_tensor = torch.tensor(req.codes, dtype=torch.long)
        turn = SynthesisTurn(
            commit_seq=req.commit_seq,
            text=req.text,
            reason=req.reason,
            created_at=1000.0,
            completed_at=1001.0,
            codes=codes_tensor,
        )

        ctx.append_turn(turn)

        self.assertEqual(len(ctx.history), 1)
        self.assertEqual(ctx.history[0].text, "Intro text")
        self.assertEqual(ctx.history[0].code_frames, 3)
        self.assertTrue(torch.equal(ctx.history[0].codes, codes_tensor))

if __name__ == "__main__":
    unittest.main()

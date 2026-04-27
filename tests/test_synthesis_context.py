import unittest
import time
from fish_speech_server.services.synthesis_context import (
    SynthesisTurn,
    SynthesisContext,
    estimate_code_frames,
)


class FakeShape:
    def __init__(self, shape):
        self.shape = shape


class TestSynthesisContext(unittest.TestCase):
    def test_estimate_code_frames(self):
        self.assertEqual(estimate_code_frames(None), 0)
        self.assertEqual(estimate_code_frames(FakeShape((10, 80))), 80)
        self.assertEqual(estimate_code_frames(FakeShape([10, 120])), 120)
        self.assertEqual(estimate_code_frames([1, 2, 3]), 3)
        self.assertEqual(estimate_code_frames([[1, 2, 3], [4, 5, 6]]), 3)
        self.assertEqual(estimate_code_frames("invalid"), 0)

    def test_append_turn(self):
        ctx = SynthesisContext(
            synthesis_session_id="session1",
            reference_id="ref1",
            created_at=time.time(),
            updated_at=time.time(),
        )
        turn = SynthesisTurn(
            commit_seq=1,
            text="hello",
            reason="manual",
            created_at=time.time(),
            codes=[1, 2, 3],
        )
        ctx.append_turn(turn)

        self.assertEqual(len(ctx.history), 1)
        self.assertEqual(ctx.history[0].code_frames, 3)
        self.assertGreaterEqual(ctx.updated_at, turn.created_at)

    def test_trim_by_turns(self):
        ctx = SynthesisContext(
            synthesis_session_id="session1",
            reference_id="ref1",
            created_at=time.time(),
            updated_at=time.time(),
            max_history_turns=2,
        )
        for i in range(5):
            ctx.append_turn(
                SynthesisTurn(
                    commit_seq=i,
                    text=f"text{i}",
                    reason="manual",
                    created_at=time.time(),
                )
            )

        self.assertEqual(len(ctx.history), 2)
        self.assertEqual(ctx.history[0].text, "text3")
        self.assertEqual(ctx.history[1].text, "text4")

    def test_trim_by_chars(self):
        ctx = SynthesisContext(
            synthesis_session_id="session1",
            reference_id="ref1",
            created_at=time.time(),
            updated_at=time.time(),
            max_history_chars=10,
        )
        # text0 (5 chars)
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=0, text="abcde", reason="m", created_at=time.time()
            )
        )
        # text1 (5 chars) -> total 10 chars, okay
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=1, text="fghij", reason="m", created_at=time.time()
            )
        )
        self.assertEqual(len(ctx.history), 2)

        # text2 (5 chars) -> total 15 chars, should trim "abcde"
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=2, text="klmno", reason="m", created_at=time.time()
            )
        )
        self.assertEqual(len(ctx.history), 2)
        self.assertEqual(ctx.history[0].text, "fghij")
        self.assertEqual(ctx.history[1].text, "klmno")

    def test_trim_by_code_frames(self):
        ctx = SynthesisContext(
            synthesis_session_id="session1",
            reference_id="ref1",
            created_at=time.time(),
            updated_at=time.time(),
            max_history_code_frames=100,
        )
        # turn 0: 60 frames
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=0,
                text="a",
                reason="m",
                created_at=time.time(),
                code_frames=60,
            )
        )
        # turn 1: 30 frames -> total 90, okay
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=1,
                text="b",
                reason="m",
                created_at=time.time(),
                code_frames=30,
            )
        )
        self.assertEqual(len(ctx.history), 2)

        # turn 2: 20 frames -> total 110, should trim turn 0
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=2,
                text="c",
                reason="m",
                created_at=time.time(),
                code_frames=20,
            )
        )
        self.assertEqual(len(ctx.history), 2)
        self.assertEqual(ctx.history[0].text, "b")
        self.assertEqual(ctx.history[1].text, "c")

    def test_trim_keeps_last_turn(self):
        ctx = SynthesisContext(
            synthesis_session_id="session1",
            reference_id="ref1",
            created_at=time.time(),
            updated_at=time.time(),
            max_history_chars=2,
        )
        # single turn with 10 chars, should be kept because it is the last one
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=0, text="verylongtext", reason="m", created_at=time.time()
            )
        )
        self.assertEqual(len(ctx.history), 1)
        self.assertEqual(ctx.history[0].text, "verylongtext")

    def test_to_public_dict(self):
        ctx = SynthesisContext(
            synthesis_session_id="session1",
            reference_id="ref1",
            created_at=1000.0,
            updated_at=1000.0,
        )
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=1,
                text="hello",
                reason="manual",
                created_at=1001.0,
                codes=[1, 2, 3],
            )
        )

        d = ctx.to_public_dict()
        self.assertEqual(d["synthesis_session_id"], "session1")
        self.assertEqual(len(d["history"]), 1)
        self.assertEqual(d["history"][0]["text"], "hello")
        self.assertTrue(d["history"][0]["has_codes"])
        self.assertNotIn("codes", d["history"][0])

    def test_build_text_history(self):
        ctx = SynthesisContext(
            synthesis_session_id="session1",
            reference_id="ref1",
            created_at=time.time(),
            updated_at=time.time(),
        )
        codes = [1, 2, 3]
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=1,
                text="hello",
                reason="manual",
                created_at=time.time(),
                codes=codes,
            )
        )

        history = ctx.build_text_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["text"], "hello")
        self.assertEqual(history[0]["codes"], codes)


if __name__ == "__main__":
    unittest.main()

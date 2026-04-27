import unittest
import time
from fish_speech_server.services.synthesis_context import SynthesisTurn, SynthesisContext
from fish_speech_server.services.continuation import (
    select_history_turns_for_continuation,
    build_continuation_debug_summary,
)


class TestContinuation(unittest.TestCase):
    def test_selection_logic(self):
        ctx = SynthesisContext(
            synthesis_session_id="s1",
            reference_id="r1",
            created_at=time.time(),
            updated_at=time.time(),
        )

        # Turn 0: codes=None
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=0, text="t0", reason="m", created_at=time.time(), codes=None
            )
        )
        # Turn 1: codes=[...]
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=1, text="t1", reason="m", created_at=time.time(), codes=[1]
            )
        )
        # Turn 2: codes=[...]
        ctx.append_turn(
            SynthesisTurn(
                commit_seq=2, text="t2", reason="m", created_at=time.time(), codes=[2]
            )
        )

        selected = select_history_turns_for_continuation(ctx)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].commit_seq, 1)
        self.assertEqual(selected[1].commit_seq, 2)

        summary = build_continuation_debug_summary(ctx)
        self.assertEqual(summary["selected_turns"], 2)
        self.assertEqual(summary["commit_seq_list"], [1, 2])

    def test_empty_history(self):
        ctx = SynthesisContext(
            synthesis_session_id="s1",
            reference_id="r1",
            created_at=time.time(),
            updated_at=time.time(),
        )
        selected = select_history_turns_for_continuation(ctx)
        self.assertEqual(len(selected), 0)

        summary = build_continuation_debug_summary(ctx)
        self.assertEqual(summary["selected_turns"], 0)


if __name__ == "__main__":
    unittest.main()

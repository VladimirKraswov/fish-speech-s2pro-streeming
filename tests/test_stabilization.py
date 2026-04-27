import unittest
from fish_speech_server.config import (
    CommitPolicyConfig,
    CommitStageConfig,
    PlaybackConfig,
    ProxyConfig,
    ProxyTTSConfig,
    SessionRuntimeConfig,
)
from fish_speech_server.proxy.pcm import _extract_commits, build_upstream_payload


class TestStabilization(unittest.TestCase):
    def test_extract_commits_no_microcommits(self):
        cfg = CommitPolicyConfig(
            first=CommitStageConfig(
                min_chars=40, target_chars=58, max_chars=84, max_wait_ms=150, allow_partial_after_ms=240
            ),
            next=CommitStageConfig(
                min_chars=40, target_chars=58, max_chars=84, max_wait_ms=150, allow_partial_after_ms=240
            )
        )

        # 1. Very short sentence should NOT be committed
        commits, tail = _extract_commits("Да.", cfg, next_commit_seq=1)
        self.assertEqual(len(commits), 0)
        self.assertEqual(tail, "Да.")

        # 2. Sentence long enough (>= 24 chars) should be committed even if < min_chars
        text = "Это достаточно длинное предложение."
        self.assertGreaterEqual(len(text), 24)
        self.assertLess(len(text), 40)
        commits, tail = _extract_commits(text, cfg, next_commit_seq=1)
        self.assertEqual(len(commits), 1)
        self.assertEqual(commits[0][0], text)
        self.assertEqual(tail, "")

    def test_build_upstream_payload_purity(self):
        # build_upstream_payload should not contain stateful fields
        # they are added in stream_one_commit
        config = ProxyConfig(
            commit=CommitPolicyConfig(
                first=CommitStageConfig(min_chars=1, target_chars=1, max_chars=1, max_wait_ms=1, allow_partial_after_ms=1),
                next=CommitStageConfig(min_chars=1, target_chars=1, max_chars=1, max_wait_ms=1, allow_partial_after_ms=1)
            ),
            tts=ProxyTTSConfig(stateful_synthesis=True),
            playback=PlaybackConfig(),
            session=SessionRuntimeConfig()
        )

        payload = build_upstream_payload("hello", config)
        self.assertNotIn("synthesis_session_id", payload)
        self.assertNotIn("commit_seq", payload)
        self.assertNotIn("commit_reason", payload)


if __name__ == "__main__":
    unittest.main()

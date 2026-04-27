from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SynthesisTurn:
    commit_seq: int
    text: str
    reason: str
    created_at: float
    completed_at: float | None = None
    pcm_bytes: int = 0
    upstream_bytes: int = 0
    code_frames: int = 0
    codes: Any | None = None


def estimate_code_frames(codes: Any | None) -> int:
    """
    Estimates the number of code frames from codes.
    Expects codes to be a torch.Tensor (or object with .shape), list, or tuple.

    - None -> 0
    - object with .shape -> take the last dimension (e.g., [8, 200] -> 200)
    - nested list/tuple [[...], [...]] -> returns length of the FIRST inner element (assumed frames)
    - flat list/tuple [...] -> returns length of the list
    - other -> 0
    """
    if codes is None:
        return 0

    # Handle objects with .shape (e.g. torch.Tensor, numpy.ndarray)
    if hasattr(codes, "shape"):
        try:
            shape = codes.shape
            if len(shape) > 0:
                return int(shape[-1])
        except Exception:
            pass

    # Handle list or tuple
    if isinstance(codes, (list, tuple)):
        if len(codes) == 0:
            return 0

        first_elem = codes[0]
        # If nested, we assume the first element's length is the number of frames
        # Example: [[1, 2, 3], [4, 5, 6]] -> len([1, 2, 3]) -> 3
        if isinstance(first_elem, (list, tuple)):
            return len(first_elem)

        return len(codes)

    return 0


@dataclass
class SynthesisContext:
    synthesis_session_id: str
    reference_id: str
    created_at: float
    updated_at: float
    history: list[SynthesisTurn] = field(default_factory=list)
    max_history_turns: int = 4
    max_history_chars: int = 500
    max_history_code_frames: int = 2000

    def append_turn(self, turn: SynthesisTurn) -> None:
        if turn.code_frames == 0 and turn.codes is not None:
            turn.code_frames = estimate_code_frames(turn.codes)

        self.history.append(turn)
        self.updated_at = time.time()
        self.trim()

    def trim(self) -> None:
        if not self.history:
            return

        # Always keep at least the last turn
        last_turn = self.history[-1]
        other_turns = self.history[:-1]

        # 1. Limit by max_history_turns
        if len(self.history) > self.max_history_turns:
            other_turns = self.history[-(self.max_history_turns) : -1]

        # 2. Limit by max_history_chars
        # We work backwards from the newest 'other' turns
        def get_chars(turns):
            return sum(len(t.text) for t in turns) + len(last_turn.text)

        while other_turns and get_chars(other_turns) > self.max_history_chars:
            other_turns.pop(0)

        # 3. Limit by max_history_code_frames
        def get_frames(turns):
            return sum(t.code_frames for t in turns) + last_turn.code_frames

        while other_turns and get_frames(other_turns) > self.max_history_code_frames:
            other_turns.pop(0)

        self.history = other_turns + [last_turn]

    def history_chars(self) -> int:
        return sum(len(t.text) for t in self.history)

    def history_code_frames(self) -> int:
        return sum(t.code_frames for t in self.history)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "synthesis_session_id": self.synthesis_session_id,
            "reference_id": self.reference_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "max_history_turns": self.max_history_turns,
            "max_history_chars": self.max_history_chars,
            "max_history_code_frames": self.max_history_code_frames,
            "history_turns": len(self.history),
            "history_chars": self.history_chars(),
            "history_code_frames": self.history_code_frames(),
            "history": [
                {
                    "commit_seq": t.commit_seq,
                    "text": t.text,
                    "reason": t.reason,
                    "created_at": t.created_at,
                    "completed_at": t.completed_at,
                    "pcm_bytes": t.pcm_bytes,
                    "upstream_bytes": t.upstream_bytes,
                    "code_frames": t.code_frames,
                    "has_codes": t.codes is not None,
                }
                for t in self.history
            ],
        }

    def build_text_history(self) -> list[dict[str, Any]]:
        """
        Returns a list of turns in a format suitable for prompt building.
        Typically used to construct the 'history' part of the prompt.
        """
        return [
            {
                "text": t.text,
                "codes": t.codes,
            }
            for t in self.history
        ]

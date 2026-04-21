from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


SPEAKER_TAG_RE = re.compile(r"<\|speaker:\d+\|>")
MULTISPACE_RE = re.compile(r"[ \t]+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?。，！？；：…])")
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

_SENTENCE_ENDINGS = {".", "!", "?", ";", "…", "。", "！", "？", "；"}
_HARD_BREAKS = {"\n"}


@dataclass(frozen=True)
class BufferEmit:
    text: str
    reason: Literal["punct", "hard_limit", "force", "final"]
    words: int
    chars: int


class StreamingTextBuffer:
    """
    Буфер для realtime TTS.

    Цели:
    - накапливать маленькие дельты текста от LLM;
    - не слать слишком короткие куски в TTS;
    - как можно раньше флашить на естественных границах фразы;
    - не раздувать буфер бесконечно.
    """

    def __init__(
        self,
        *,
        min_words: int = 3,
        soft_limit_chars: int = 120,
        hard_limit_chars: int = 220,
    ) -> None:
        if min_words < 1:
            raise ValueError("min_words must be >= 1")
        if soft_limit_chars < 16:
            raise ValueError("soft_limit_chars must be >= 16")
        if hard_limit_chars < soft_limit_chars:
            raise ValueError("hard_limit_chars must be >= soft_limit_chars")

        self.min_words = min_words
        self.soft_limit_chars = soft_limit_chars
        self.hard_limit_chars = hard_limit_chars
        self._buffer = ""
        self._is_first_chunk = True

    @property
    def text(self) -> str:
        return self._buffer

    def empty(self) -> bool:
        return not self._buffer.strip()

    def clear(self) -> None:
        self._buffer = ""
        self._is_first_chunk = True

    def replace(self, text: str) -> None:
        self._buffer = self._cleanup_text(text)
        self._is_first_chunk = True

    def push(
        self,
        text: str,
        *,
        force: bool = False,
        final: bool = False,
    ) -> list[BufferEmit]:
        if text:
            self._buffer = self._append_fragment(self._buffer, text)

        return self._drain_ready(force=force, final=final)

    def flush(self, *, final: bool = False) -> list[BufferEmit]:
        return self._drain_ready(force=True, final=final)

    def _drain_ready(
        self,
        *,
        force: bool = False,
        final: bool = False,
    ) -> list[BufferEmit]:
        out: list[BufferEmit] = []

        while True:
            if self.empty():
                self._buffer = ""
                break

            split = self._find_split(force=force, final=final)
            if split is None:
                break

            raw_chunk = self._buffer[:split]
            self._buffer = self._buffer[split:].lstrip()

            chunk = self._cleanup_text(raw_chunk)
            if not chunk:
                continue

            if final:
                reason: Literal["punct", "hard_limit", "force", "final"] = "final"
            elif force:
                reason = "force"
            else:
                stripped = raw_chunk.rstrip()
                if stripped and stripped[-1] in _SENTENCE_ENDINGS.union(_HARD_BREAKS):
                    reason = "punct"
                else:
                    reason = "hard_limit"

            out.append(
                BufferEmit(
                    text=chunk,
                    reason=reason,
                    words=self._count_words(chunk),
                    chars=len(chunk),
                )
            )
            self._is_first_chunk = False

            if force or final:
                break

        return out

    def _find_split(self, *, force: bool, final: bool) -> int | None:
        buf = self._buffer.strip()
        if not buf:
            return None

        total_words = self._count_words(buf)

        if final or force:
            return len(self._buffer)

        # Optimization for TTFA: emit the very first chunk as soon as we have at least one word.
        if self._is_first_chunk and total_words >= 1:
            # We still prefer splitting at a sentence boundary if possible
            punct_split = self._find_sentence_boundary(buf)
            if punct_split is not None:
                return self._map_stripped_index_to_raw_index(punct_split)

            # If no punctuation but we have some text, just split at the end of the current buffer
            # (up to hard limit) to get the first audio fast.
            return len(self._buffer)

        if total_words < self.min_words and len(buf) < self.hard_limit_chars:
            return None

        punct_split = self._find_sentence_boundary(buf)
        if punct_split is not None:
            return self._map_stripped_index_to_raw_index(punct_split)

        if len(buf) >= self.hard_limit_chars:
            hard_split = self._find_hard_limit_boundary(buf)
            return self._map_stripped_index_to_raw_index(hard_split)

        return None

    def _find_sentence_boundary(self, buf: str) -> int | None:
        rightmost_valid: int | None = None
        limit = min(len(buf), self.hard_limit_chars)

        for i, ch in enumerate(buf[:limit], start=1):
            if ch not in _SENTENCE_ENDINGS and ch not in _HARD_BREAKS:
                continue

            prefix = buf[:i].strip()
            if self._count_words(prefix) < self.min_words:
                continue

            rightmost_valid = i

        return rightmost_valid

    def _find_hard_limit_boundary(self, buf: str) -> int:
        limit = min(len(buf), self.hard_limit_chars)
        candidate = buf.rfind(" ", 0, limit)

        if candidate != -1:
            prefix = buf[:candidate].strip()
            if self._count_words(prefix) >= self.min_words:
                return candidate

        return limit

    def _append_fragment(self, base: str, fragment: str) -> str:
        frag = self._cleanup_text(fragment)
        if not frag:
            return base

        if not base:
            return frag

        if self._needs_space_between(base, frag):
            return self._cleanup_text(f"{base} {frag}")

        return self._cleanup_text(f"{base}{frag}")

    @staticmethod
    def _needs_space_between(left: str, right: str) -> bool:
        if not left or not right:
            return False

        if left.endswith(tuple(_HARD_BREAKS)):
            return False

        if right[0] in ",.;:!?。，！？；：…":
            return False

        if left[-1] in "([{/«„\"'":
            return False

        if right.startswith((")", "]", "}", "/")):
            return False

        return not left.endswith(" ")

    @staticmethod
    def _cleanup_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = MULTISPACE_RE.sub(" ", text)
        text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _count_words(text: str) -> int:
        plain = SPEAKER_TAG_RE.sub(" ", text)
        return len(WORD_RE.findall(plain))

    def _map_stripped_index_to_raw_index(self, stripped_index: int) -> int:
        raw = self._buffer
        stripped = raw.strip()

        if raw == stripped:
            return stripped_index

        left_trim = len(raw) - len(raw.lstrip())
        return left_trim + stripped_index
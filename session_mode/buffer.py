from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


SPEAKER_TAG_RE = re.compile(r"<\|speaker:\d+\|>")
MULTISPACE_RE = re.compile(r"[ \t]+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?。，！？；：…])")
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

_SENTENCE_ENDINGS = {".", "!", "?", ";", "…", "。", "！", "？", "；"}
_CLAUSE_ENDINGS = {",", ":", "，", "："}
_HARD_BREAKS = {"\n"}
_BOUNDARY_CHARS = _SENTENCE_ENDINGS.union(_HARD_BREAKS).union(
    _CLAUSE_ENDINGS
)
_TRAILING_QUOTES = set("\"'”’»)")

EmitReason = Literal["punct", "hard_limit", "force", "final"]


@dataclass(frozen=True)
class BufferEmit:
    text: str
    reason: EmitReason
    words: int
    chars: int
    starts_with_full_word: bool
    ends_with_full_word: bool


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
        self._last_fragment_ended_at_boundary = False
        self._buffer_ended_at_boundary = False

    @property
    def text(self) -> str:
        return self._buffer

    def empty(self) -> bool:
        return not self._buffer.strip()

    def clear(self) -> None:
        self._buffer = ""
        self._is_first_chunk = True
        self._last_fragment_ended_at_boundary = False
        self._buffer_ended_at_boundary = False

    def replace(self, text: str) -> None:
        self._buffer = self._cleanup_text(text)
        self._is_first_chunk = True
        self._last_fragment_ended_at_boundary = self._raw_text_ended_at_boundary(text)
        self._buffer_ended_at_boundary = self._last_fragment_ended_at_boundary

    def push(
        self,
        text: str,
        *,
        force: bool = False,
        final: bool = False,
    ) -> list[BufferEmit]:
        if text:
            fragment_started_at_boundary = self._raw_text_started_at_boundary(text)
            fragment_ended_at_boundary = self._raw_text_ended_at_boundary(text)
            self._buffer = self._append_fragment(
                self._buffer,
                text,
                boundary_between=(
                    self._buffer_ended_at_boundary or fragment_started_at_boundary
                ),
            )
            self._last_fragment_ended_at_boundary = fragment_ended_at_boundary
            self._buffer_ended_at_boundary = fragment_ended_at_boundary

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

            split_info = self._find_split(force=force, final=final)
            if split_info is None:
                break

            split, reason = split_info

            raw_chunk = self._buffer[:split]
            buffer_ended_at_boundary = self._buffer_ended_at_boundary
            self._buffer = self._buffer[split:].lstrip()
            self._last_fragment_ended_at_boundary = False
            self._buffer_ended_at_boundary = bool(
                self._buffer and buffer_ended_at_boundary
            )

            chunk = self._cleanup_text(raw_chunk)
            if not chunk:
                continue

            out.append(
                BufferEmit(
                    text=chunk,
                    reason=reason,
                    words=self._count_words(chunk),
                    chars=len(chunk),
                    starts_with_full_word=self.starts_with_full_word(chunk),
                    ends_with_full_word=self.ends_with_full_word(chunk),
                )
            )
            self._is_first_chunk = False

            if force or final:
                break

        return out

    def _find_split(
        self,
        *,
        force: bool,
        final: bool,
    ) -> tuple[int, EmitReason] | None:
        raw = self._buffer
        buf = raw.strip()
        if not buf:
            return None

        if final:
            return len(raw), "final"

        if force:
            force_split = self._find_force_boundary(buf)
            if force_split is None:
                return None
            return self._map_stripped_index_to_raw_index(force_split), "force"

        total_words = self._count_words(buf)

        # Первый чанк можно отдать раньше, но только если последний входной
        # фрагмент закончился на безопасной границе. Это защищает от озвучки
        # обрезков слов вроде "Hel" / "при".
        if self._is_first_chunk and total_words >= self.min_words:
            punct_split = self._find_sentence_boundary(buf)
            if punct_split is not None:
                return self._map_stripped_index_to_raw_index(punct_split), "punct"

            if len(buf) >= self.hard_limit_chars:
                hard_split = self._find_hard_limit_boundary(buf)
                return self._map_stripped_index_to_raw_index(hard_split), "hard_limit"

            return None

        if total_words < self.min_words and len(buf) < self.hard_limit_chars:
            return None

        punct_split = self._find_sentence_boundary(buf)
        if punct_split is not None:
            return self._map_stripped_index_to_raw_index(punct_split), "punct"

        if len(buf) >= self.soft_limit_chars:
            clause_split = self._find_clause_boundary(buf)
            if clause_split is not None:
                return self._map_stripped_index_to_raw_index(clause_split), "punct"

        if len(buf) >= self.hard_limit_chars:
            hard_split = self._find_hard_limit_boundary(buf)
            return self._map_stripped_index_to_raw_index(hard_split), "hard_limit"

        return None

    def _find_sentence_boundary(self, buf: str) -> int | None:
        return self._find_punctuation_boundary(
            buf,
            boundary_chars=_SENTENCE_ENDINGS.union(_HARD_BREAKS),
            limit=min(len(buf), self.hard_limit_chars),
        )

    def _find_clause_boundary(self, buf: str) -> int | None:
        return self._find_punctuation_boundary(
            buf,
            boundary_chars=_CLAUSE_ENDINGS,
            limit=min(len(buf), self.hard_limit_chars),
        )

    def _find_punctuation_boundary(
        self,
        buf: str,
        *,
        boundary_chars: set[str],
        limit: int,
    ) -> int | None:
        rightmost_valid: int | None = None

        for i, ch in enumerate(buf[:limit], start=1):
            if ch not in boundary_chars:
                continue

            prefix = buf[:i].strip()
            if self._count_words(prefix) < self.min_words:
                continue

            rightmost_valid = i

        return rightmost_valid

    def _find_hard_limit_boundary(self, buf: str) -> int:
        limit = min(len(buf), self.hard_limit_chars)
        candidate = self._find_last_safe_split_before(buf, limit)
        if candidate is not None:
            prefix = buf[:candidate].strip()
            if self._count_words(prefix) >= self.min_words:
                return candidate

        # If the current word crosses the hard limit, wait a little for its
        # natural end instead of cutting inside it. For pathological very long
        # tokens we still have an emergency cap to keep the buffer bounded.
        overrun_limit = min(len(buf), self.hard_limit_chars + max(48, self.hard_limit_chars // 3))
        for i in range(limit, overrun_limit):
            ch = buf[i]
            if ch.isspace() or ch in _BOUNDARY_CHARS:
                return i + (0 if ch.isspace() else 1)

        if len(buf) <= overrun_limit:
            return len(buf)
        return overrun_limit

    def _find_force_boundary(self, buf: str) -> int | None:
        if self._last_fragment_ended_at_boundary:
            if self._count_words(buf) >= self.min_words:
                return len(buf)

        split = self._find_last_safe_split_before(buf, len(buf))
        if split is None:
            return None

        prefix = buf[:split].strip()
        if self._count_words(prefix) < self.min_words:
            return None
        return split

    @staticmethod
    def _find_last_safe_split_before(buf: str, limit: int) -> int | None:
        limit = min(limit, len(buf))
        if limit <= 0:
            return None

        for i in range(limit - 1, -1, -1):
            ch = buf[i]
            if ch.isspace():
                return i
            if ch in _BOUNDARY_CHARS:
                return i + 1
        return None

    def _append_fragment(
        self,
        base: str,
        fragment: str,
        *,
        boundary_between: bool,
    ) -> str:
        frag = self._cleanup_text(fragment)
        if not frag:
            return base

        if not base:
            return frag

        if self._needs_space_between(base, frag, boundary_between=boundary_between):
            return self._cleanup_text(f"{base} {frag}")

        return self._cleanup_text(f"{base}{frag}")

    @staticmethod
    def _needs_space_between(
        left: str,
        right: str,
        *,
        boundary_between: bool,
    ) -> bool:
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

        if left[-1].isalnum() and right[0].isalnum():
            return boundary_between

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

    @staticmethod
    def starts_with_full_word(text: str) -> bool:
        stripped = text.lstrip()
        if not stripped:
            return True
        first = stripped[0]
        return first.isalnum() or first in "<([{«\"'"

    @staticmethod
    def ends_with_full_word(text: str) -> bool:
        stripped = text.rstrip()
        if not stripped:
            return True
        while stripped and stripped[-1] in _TRAILING_QUOTES:
            stripped = stripped[:-1].rstrip()
        if not stripped:
            return True
        last = stripped[-1]
        return last.isalnum() or last in _BOUNDARY_CHARS

    @staticmethod
    def _raw_text_ended_at_boundary(text: str) -> bool:
        if not text:
            return False
        last = text[-1]
        return last.isspace() or last in _BOUNDARY_CHARS

    @staticmethod
    def _raw_text_started_at_boundary(text: str) -> bool:
        if not text:
            return False
        first = text[0]
        return first.isspace() or first in _BOUNDARY_CHARS

    def _map_stripped_index_to_raw_index(self, stripped_index: int) -> int:
        raw = self._buffer
        stripped = raw.strip()

        if raw == stripped:
            return stripped_index

        left_trim = len(raw) - len(raw.lstrip())
        return left_trim + stripped_index

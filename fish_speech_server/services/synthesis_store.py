from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fish_speech_server.services.synthesis_context import SynthesisContext


class SynthesisSessionStore:
    def __init__(
        self,
        ttl_sec: int = 1800,
        max_sessions: int = 128,
        default_max_history_turns: int = 4,
        default_max_history_chars: int = 500,
        default_max_history_code_frames: int = 2000,
    ) -> None:
        self.ttl_sec = ttl_sec
        self.max_sessions = max_sessions
        self.default_max_history_turns = default_max_history_turns
        self.default_max_history_chars = default_max_history_chars
        self.default_max_history_code_frames = default_max_history_code_frames

        self._items: dict[str, SynthesisContext] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        reference_id: str,
        *,
        max_history_turns: int | None = None,
        max_history_chars: int | None = None,
        max_history_code_frames: int | None = None,
    ) -> SynthesisContext:
        await self.cleanup()

        async with self._lock:
            if len(self._items) >= self.max_sessions:
                raise ValueError(f"Maximum number of sessions ({self.max_sessions}) exceeded")

            now = time.time()
            synthesis_session_id = uuid.uuid4().hex

            ctx = SynthesisContext(
                synthesis_session_id=synthesis_session_id,
                reference_id=reference_id,
                created_at=now,
                updated_at=now,
                max_history_turns=max_history_turns or self.default_max_history_turns,
                max_history_chars=max_history_chars or self.default_max_history_chars,
                max_history_code_frames=max_history_code_frames or self.default_max_history_code_frames,
            )

            self._items[synthesis_session_id] = ctx
            return ctx

    async def get(self, synthesis_session_id: str, touch: bool = True) -> SynthesisContext | None:
        await self.cleanup()

        async with self._lock:
            ctx = self._items.get(synthesis_session_id)
            if ctx is None:
                return None

            if touch:
                ctx.updated_at = time.time()

            return ctx

    async def close(self, synthesis_session_id: str) -> bool:
        async with self._lock:
            removed = self._items.pop(synthesis_session_id, None)
            return removed is not None

    async def cleanup(self) -> int:
        now = time.time()
        async with self._lock:
            expired_ids = [
                sid
                for sid, ctx in self._items.items()
                if ctx.updated_at + self.ttl_sec < now
            ]
            for sid in expired_ids:
                del self._items[sid]

            return len(expired_ids)

    async def stats(self) -> dict[str, Any]:
        await self.cleanup()

        async with self._lock:
            now = time.time()
            sessions_stats = []
            for ctx in self._items.values():
                sessions_stats.append({
                    "synthesis_session_id": ctx.synthesis_session_id,
                    "reference_id": ctx.reference_id,
                    "age_sec": round(now - ctx.created_at, 1),
                    "idle_sec": round(now - ctx.updated_at, 1),
                    "history_turns": len(ctx.history),
                    "history_chars": ctx.history_chars(),
                    "history_code_frames": ctx.history_code_frames(),
                })

            return {
                "active_sessions": len(self._items),
                "ttl_sec": self.ttl_sec,
                "max_sessions": self.max_sessions,
                "sessions": sessions_stats,
            }

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReferenceCache:
    by_id: dict[str, Any] = field(default_factory=dict)
    by_hash: dict[str, Any] = field(default_factory=dict)

    def clear_id(self, reference_id: str) -> None:
        self.by_id.pop(reference_id, None)

    def clear(self) -> None:
        self.by_id.clear()
        self.by_hash.clear()

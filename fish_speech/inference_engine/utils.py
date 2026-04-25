from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import numpy as np


@dataclass
class InferenceResult:
    code: Literal["tokens", "segment", "error", "final"]
    audio: Optional[Tuple[int, np.ndarray]]
    error: Optional[Exception]
    tokens: Any | None = None

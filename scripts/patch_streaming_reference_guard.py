#!/usr/bin/env python3
from pathlib import Path
import sys


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in [cur] + list(cur.parents):
        if (
            (candidate / "fish_speech").exists()
            and (candidate / "tools").exists()
            and (candidate / "pyproject.toml").exists()
        ):
            return candidate
    raise RuntimeError(
        f"Не удалось найти корень проекта вверх от: {start}\n"
        f"Ожидались: fish_speech/, tools/, pyproject.toml"
    )


SCRIPT_PATH = Path(__file__).resolve()
ROOT = find_repo_root(SCRIPT_PATH.parent)


def replace_text(path: Path, old: str, new: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old in text:
        text = text.replace(old, new, 1)
        path.write_text(text, encoding="utf-8")
        return True
    if new in text:
        return False
    raise RuntimeError(
        f"Не найден ожидаемый фрагмент в файле {path.relative_to(ROOT)}:\n---\n{old}\n---"
    )


def patch_file(rel_path: str, replacements: list[tuple[str, str]]) -> None:
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    changed_any = False
    for old, new in replacements:
        changed = replace_text(path, old, new)
        changed_any = changed_any or changed

    if changed_any:
        print(f"[patched] {rel_path}")
    else:
        print(f"[ok] {rel_path} уже был обновлён")


PATCHES = {
    "tools/api_server.py": [
        (
            '''import re
import warnings
from threading import Lock

# Suppress torch.compile (Inductor) spam: "Logical operators 'and'/'or' deprecated, use '&'/'|'"
# Source: torch._inductor.runtime.triton_helpers — fix belongs in PyTorch upstream
warnings.filterwarnings(
    "ignore",
    message=".*Logical operators 'and' and 'or' are deprecated for non-scalar tensors.*",
    category=UserWarning,
    module="torch._inductor",
)''',
            '''import re
import warnings
from threading import Lock

# Suppress noisy torch.compile / Inductor warnings from Triton helpers.
warnings.filterwarnings(
    "ignore",
    message=".*Logical operators 'and' and 'or' are deprecated for non-scalar tensors.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Enable tracemalloc to get the object allocation traceback.*",
    category=UserWarning,
)'''
        ),
    ],

    "fish_speech/inference_engine/reference_loader.py": [
        (
            '''import io
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal, Tuple

import torch
import torchaudio
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC
from fish_speech.utils.file import (
    AUDIO_EXTENSIONS,
    audio_to_bytes,
    list_files,
    read_ref_text,
)
from fish_speech.utils.schema import ServeReferenceAudio


class ReferenceLoader:''',
            '''import io
import os
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal, Tuple

import torch
import torchaudio
from loguru import logger

from fish_speech.models.dac.modded_dac import DAC
from fish_speech.utils.file import (
    AUDIO_EXTENSIONS,
    audio_to_bytes,
    list_files,
    read_ref_text,
)
from fish_speech.utils.schema import ServeReferenceAudio


def _trim_reference_text(text: str) -> str:
    text = " ".join(text.strip().split())

    raw = os.getenv("FISH_REF_TEXT_MAX_BYTES", "220").strip()
    try:
        max_bytes = int(raw)
    except ValueError:
        max_bytes = 220

    if max_bytes <= 0:
        return text

    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    cut = encoded[:max_bytes]
    while cut and (cut[-1] & 0xC0) == 0x80:
        cut = cut[:-1]

    trimmed = cut.decode("utf-8", errors="ignore").rstrip()

    last_stop = max(
        trimmed.rfind("."),
        trimmed.rfind("!"),
        trimmed.rfind("?"),
        trimmed.rfind("…"),
    )
    if last_stop >= 40:
        trimmed = trimmed[: last_stop + 1]

    return trimmed.strip()


def _trim_reference_codes(codes: torch.Tensor | None) -> torch.Tensor | None:
    if codes is None:
        return None

    raw = os.getenv("FISH_REF_MAX_FRAMES", "128").strip()
    try:
        max_frames = int(raw)
    except ValueError:
        max_frames = 128

    if max_frames <= 0:
        return codes

    if codes.shape[-1] <= max_frames:
        return codes

    logger.warning(
        "Reference codes trimmed from {} to {} frames for cache safety",
        codes.shape[-1],
        max_frames,
    )
    return codes[..., :max_frames].contiguous()


class ReferenceLoader:'''
        ),

        (
            '''                prompt_texts.append(read_ref_text(str(lab_path)))''',
            '''                ref_text = _trim_reference_text(read_ref_text(str(lab_path)))
                prompt_texts.append(ref_text)'''
        ),

        (
            '''                    prompt_tokens.append(
                        loaded if isinstance(loaded, torch.Tensor) else loaded[0]
                    )''',
            '''                    loaded_codes = loaded if isinstance(loaded, torch.Tensor) else loaded[0]
                    prompt_tokens.append(_trim_reference_codes(loaded_codes))'''
        ),

        (
            '''                    prompt_tokens.append(
                        self.encode_reference(
                            reference_audio=audio_to_bytes(str(audio_path)),
                            enable_reference_audio=True,
                        )
                    )''',
            '''                    prompt_tokens.append(
                        _trim_reference_codes(
                            self.encode_reference(
                                reference_audio=audio_to_bytes(str(audio_path)),
                                enable_reference_audio=True,
                            )
                        )
                    )'''
        ),

        (
            '''                prompt_tokens.append(
                    self.encode_reference(
                        reference_audio=ref.audio,
                        enable_reference_audio=True,
                    )
                )
                prompt_texts.append(ref.text)
                self.ref_by_hash[audio_hashes[i]] = (prompt_tokens[-1], ref.text)''',
            '''                trimmed_codes = _trim_reference_codes(
                    self.encode_reference(
                        reference_audio=ref.audio,
                        enable_reference_audio=True,
                    )
                )
                trimmed_text = _trim_reference_text(ref.text)
                prompt_tokens.append(trimmed_codes)
                prompt_texts.append(trimmed_text)
                self.ref_by_hash[audio_hashes[i]] = (trimmed_codes, trimmed_text)'''
        ),
    ],

    "tools/proxy/fish_proxy_pcm.py": [
        (
            '''async def pcm_stream(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str | None = Query(default=None),
):
    req_id = uuid.uuid4().hex[:8]
    payload = {"text": text, "streaming": True}
    if reference_id:
        payload["reference_id"] = reference_id''',
            '''async def pcm_stream(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str = Query(default="ref"),
):
    req_id = uuid.uuid4().hex[:8]
    payload = {"text": text, "streaming": True, "reference_id": reference_id}'''
        ),
    ],

    "tools/proxy/fish_proxy_server.py": [
        (
            '''    reference_id: str | None = Query(None),''',
            '''    reference_id: str = Query("ref"),'''
        ),
        (
            '''    reference_id: str | None = Query(None),''',
            '''    reference_id: str = Query("ref"),'''
        ),
    ],

    "scripts/warmup_5090.sh": [
        (
            '''curl -sf -X POST "$BASE_URL/v1/tts" \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"${WARMUP_TEXT}\",\"streaming\":true}" \''',
            '''curl -sf -X POST "$BASE_URL/v1/tts" \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"${WARMUP_TEXT}\",\"streaming\":true,\"reference_id\":\"ref\"}" \'''
        ),
    ],

    "scripts/up_5090.sh": [
        (
            '''FISH_CACHE_MAX_SEQ_LEN="${FISH_CACHE_MAX_SEQ_LEN:-320}"
FISH_MAX_NEW_TOKENS_CAP="${FISH_MAX_NEW_TOKENS_CAP:-64}"''',
            '''FISH_CACHE_MAX_SEQ_LEN="${FISH_CACHE_MAX_SEQ_LEN:-320}"
FISH_MAX_NEW_TOKENS_CAP="${FISH_MAX_NEW_TOKENS_CAP:-64}"
FISH_REF_MAX_FRAMES="${FISH_REF_MAX_FRAMES:-128}"
FISH_REF_TEXT_MAX_BYTES="${FISH_REF_TEXT_MAX_BYTES:-220}"'''
        ),
        (
            '''echo "  CHECKPOINTS_DIR=$CHECKPOINTS_DIR"''',
            '''echo "  CHECKPOINTS_DIR=$CHECKPOINTS_DIR"
echo "  FISH_REF_MAX_FRAMES=$FISH_REF_MAX_FRAMES"
echo "  FISH_REF_TEXT_MAX_BYTES=$FISH_REF_TEXT_MAX_BYTES"'''
        ),
        (
            '''    -e FISH_CACHE_MAX_SEQ_LEN="$FISH_CACHE_MAX_SEQ_LEN" \
    -e FISH_MAX_NEW_TOKENS_CAP="$FISH_MAX_NEW_TOKENS_CAP" \''',
            '''    -e FISH_CACHE_MAX_SEQ_LEN="$FISH_CACHE_MAX_SEQ_LEN" \
    -e FISH_MAX_NEW_TOKENS_CAP="$FISH_MAX_NEW_TOKENS_CAP" \
    -e FISH_REF_MAX_FRAMES="$FISH_REF_MAX_FRAMES" \
    -e FISH_REF_TEXT_MAX_BYTES="$FISH_REF_TEXT_MAX_BYTES" \'''
        ),
    ],

    "scripts/run_server_32gb.sh": [
        (
            '''# 4) Run server (tuned for 32GB: cache=384, max_new_tokens cap=80; warmup when COMPILE=1)''',
            '''FISH_REF_MAX_FRAMES="${FISH_REF_MAX_FRAMES:-128}"
FISH_REF_TEXT_MAX_BYTES="${FISH_REF_TEXT_MAX_BYTES:-220}"

# 4) Run server (tuned for 32GB: cache=384, max_new_tokens cap=80; warmup when COMPILE=1)'''
        ),
        (
            '''  -e FISH_CACHE_MAX_SEQ_LEN="${FISH_CACHE_MAX_SEQ_LEN:-320}" \
  -e FISH_MAX_NEW_TOKENS_CAP="${FISH_MAX_NEW_TOKENS_CAP:-64}" \''',
            '''  -e FISH_CACHE_MAX_SEQ_LEN="${FISH_CACHE_MAX_SEQ_LEN:-320}" \
  -e FISH_MAX_NEW_TOKENS_CAP="${FISH_MAX_NEW_TOKENS_CAP:-64}" \
  -e FISH_REF_MAX_FRAMES="${FISH_REF_MAX_FRAMES}" \
  -e FISH_REF_TEXT_MAX_BYTES="${FISH_REF_TEXT_MAX_BYTES}" \'''
        ),
    ],
}


def main() -> int:
    print(f"[info] script: {SCRIPT_PATH}")
    print(f"[info] repo root: {ROOT}")

    try:
        for rel_path, replacements in PATCHES.items():
            patch_file(rel_path, replacements)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    print()
    print("Готово.")
    print("Что изменено:")
    print("  - proxy по умолчанию шлёт reference_id=ref")
    print("  - warmup тоже идёт с reference_id=ref")
    print("  - референсный текст режется по байтам")
    print("  - референсные VQ-коды режутся по кадрам")
    print("  - лимиты пробрасываются через up_5090.sh / run_server_32gb.sh")
    print("  - шумные warning от torch.compile приглушены")
    print()
    print("Новые дефолты:")
    print("  FISH_REF_MAX_FRAMES=128")
    print("  FISH_REF_TEXT_MAX_BYTES=220")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
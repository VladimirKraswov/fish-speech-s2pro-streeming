#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_step2")
    if not backup.exists():
        shutil.copy2(path, backup)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"[{path}] не найден фрагмент:\n{old}")
    return text.replace(old, new, 1)


def patch_warmup(repo_root: Path) -> None:
    path = repo_root / "scripts" / "warmup_5090.sh"
    backup_file(path)
    text = read_text(path)

    if 'WARMUP_REFERENCE_ID="${WARMUP_REFERENCE_ID:-ref}"' not in text:
        text = replace_once(
            text,
            'WARMUP_TEXT="${WARMUP_TEXT:-Привет. Это дополнительный прогрев стримингового режима для Fish Speech.}"\nOUT_FILE="${OUT_FILE:-$REPO_ROOT/logs/warmup_stream.wav}"\n',
            'WARMUP_TEXT="${WARMUP_TEXT:-Привет. Это дополнительный прогрев стримингового режима для Fish Speech.}"\n'
            'WARMUP_REFERENCE_ID="${WARMUP_REFERENCE_ID:-ref}"\n'
            'OUT_FILE="${OUT_FILE:-$REPO_ROOT/logs/warmup_stream.wav}"\n',
            path=path,
        )

    if 'echo "Using warmup reference: $WARMUP_REFERENCE_ID"' not in text:
        text = replace_once(
            text,
            'echo "Sending one extra warmup streaming request..."\n\n',
            'echo "Sending one extra warmup streaming request..."\n'
            'echo "Using warmup reference: $WARMUP_REFERENCE_ID"\n\n',
            path=path,
        )

    old_json = """curl -sf -X POST "$BASE_URL/v1/tts" \\
  -H 'Content-Type: application/json' \\
  -d "{\\"text\\":\\"${WARMUP_TEXT}\\",\\"streaming\\":true}" \\
  --output "$OUT_FILE" \\
  --max-time 180 >/dev/null
"""
    new_json = """curl -sf -X POST "$BASE_URL/v1/tts" \\
  -H 'Content-Type: application/json' \\
  -d "{\\"text\\":\\"${WARMUP_TEXT}\\",\\"streaming\\":true,\\"reference_id\\":\\"${WARMUP_REFERENCE_ID}\\"}" \\
  --output "$OUT_FILE" \\
  --max-time 180 >/dev/null
"""
    if old_json in text:
        text = text.replace(old_json, new_json, 1)

    write_text(path, text)
    print(f"OK: {path}")


def patch_up(repo_root: Path) -> None:
    path = repo_root / "scripts" / "up_5090.sh"
    backup_file(path)
    text = read_text(path)

    if 'DEFAULT_REFERENCE_ID="${DEFAULT_REFERENCE_ID:-ref}"' not in text:
        text = replace_once(
            text,
            'CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/s2-pro}"\n\n'
            'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"\n',
            'CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/s2-pro}"\n'
            'DEFAULT_REFERENCE_ID="${DEFAULT_REFERENCE_ID:-ref}"\n\n'
            'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"\n',
            path=path,
        )

    if 'echo "  DEFAULT_REFERENCE_ID=$DEFAULT_REFERENCE_ID"' not in text:
        text = replace_once(
            text,
            'echo "  CHECKPOINTS_DIR=$CHECKPOINTS_DIR"\n',
            'echo "  CHECKPOINTS_DIR=$CHECKPOINTS_DIR"\n'
            'echo "  DEFAULT_REFERENCE_ID=$DEFAULT_REFERENCE_ID"\n',
            path=path,
        )

    old_warmup_call = '''  echo "Sending one extra warmup streaming request..."
  BASE_URL="http://127.0.0.1:${PORT}" \\
    bash "$REPO_ROOT/scripts/warmup_5090.sh"
'''
    new_warmup_call = '''  echo "Sending one extra warmup streaming request..."
  BASE_URL="http://127.0.0.1:${PORT}" \\
  WARMUP_REFERENCE_ID="$DEFAULT_REFERENCE_ID" \\
    bash "$REPO_ROOT/scripts/warmup_5090.sh"
'''
    if old_warmup_call in text:
        text = text.replace(old_warmup_call, new_warmup_call, 1)

    old_proxy_call = '''  echo "[5/5] Starting proxy..."
  PROXY_PORT="$PROXY_PORT" bash "$REPO_ROOT/scripts/run_proxy.sh"
'''
    new_proxy_call = '''  echo "[5/5] Starting proxy..."
  PROXY_PORT="$PROXY_PORT" DEFAULT_REFERENCE_ID="$DEFAULT_REFERENCE_ID" bash "$REPO_ROOT/scripts/run_proxy.sh"
'''
    if old_proxy_call in text:
        text = text.replace(old_proxy_call, new_proxy_call, 1)

    old_ready = 'echo "Proxy stream:  http://127.0.0.1:${PROXY_PORT}/pcm-stream?text=Привет"\n'
    new_ready = 'echo "Proxy stream:  http://127.0.0.1:${PROXY_PORT}/pcm-stream?text=Привет&reference_id=${DEFAULT_REFERENCE_ID}"\n'
    if old_ready in text:
        text = text.replace(old_ready, new_ready, 1)

    write_text(path, text)
    print(f"OK: {path}")


def patch_pcm_proxy(repo_root: Path) -> None:
    path = repo_root / "tools" / "proxy" / "fish_proxy_pcm.py"
    backup_file(path)
    text = read_text(path)

    if 'import os\n' not in text:
        text = replace_once(
            text,
            "import json\n",
            "import json\nimport os\n",
            path=path,
        )

    if 'DEFAULT_REFERENCE_ID = os.environ.get("DEFAULT_REFERENCE_ID", "ref")' not in text:
        text = replace_once(
            text,
            'UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)\n',
            'UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)\n'
            'DEFAULT_REFERENCE_ID = os.environ.get("DEFAULT_REFERENCE_ID", "ref")\n',
            path=path,
        )

    old_block = """async def pcm_stream(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str | None = Query(default=None),
):
    req_id = uuid.uuid4().hex[:8]
    payload = {"text": text, "streaming": True}
    if reference_id:
        payload["reference_id"] = reference_id

    logger.info("REQ %s start text_len=%s reference_id=%s", req_id, len(text), reference_id)
"""
    new_block = """async def pcm_stream(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str | None = Query(default=None),
):
    req_id = uuid.uuid4().hex[:8]
    effective_reference_id = (reference_id or DEFAULT_REFERENCE_ID).strip()
    payload = {"text": text, "streaming": True, "reference_id": effective_reference_id}

    logger.info(
        "REQ %s start text_len=%s reference_id=%s (query=%s default=%s)",
        req_id,
        len(text),
        effective_reference_id,
        reference_id,
        DEFAULT_REFERENCE_ID,
    )
"""
    if old_block in text:
        text = text.replace(old_block, new_block, 1)
    else:
        raise RuntimeError(f"[{path}] не найден блок pcm_stream для замены")

    write_text(path, text)
    print(f"OK: {path}")


def patch_stream_proxy(repo_root: Path) -> None:
    path = repo_root / "tools" / "proxy" / "fish_proxy_server.py"
    backup_file(path)
    text = read_text(path)

    if 'DEFAULT_REFERENCE_ID = os.environ.get("DEFAULT_REFERENCE_ID", "ref")' not in text:
        text = replace_once(
            text,
            'REQUEST_TIMEOUT = httpx.Timeout(connect=CONNECT_TIMEOUT, read=None, write=WRITE_TIMEOUT, pool=None)\n',
            'REQUEST_TIMEOUT = httpx.Timeout(connect=CONNECT_TIMEOUT, read=None, write=WRITE_TIMEOUT, pool=None)\n'
            'DEFAULT_REFERENCE_ID = os.environ.get("DEFAULT_REFERENCE_ID", "ref")\n',
            path=path,
        )

    old_stream = """async def stream(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str | None = Query(None),
) -> StreamingResponse:
    payload: dict[str, object] = {
        "text": text,
        "streaming": True,
        "format": "wav",
    }
    if reference_id:
        payload["reference_id"] = reference_id
"""
    new_stream = """async def stream(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str | None = Query(None),
) -> StreamingResponse:
    effective_reference_id = (reference_id or DEFAULT_REFERENCE_ID).strip()
    payload: dict[str, object] = {
        "text": text,
        "streaming": True,
        "format": "wav",
        "reference_id": effective_reference_id,
    }
"""
    if old_stream in text:
        text = text.replace(old_stream, new_stream, 1)
    else:
        raise RuntimeError(f"[{path}] не найден блок stream для замены")

    old_segments = """async def stream_segments(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str | None = Query(None),
) -> StreamingResponse:
    \"""
    Convert the upstream chunked WAV response into NDJSON where each line contains
    one complete WAV segment as base64.
    This avoids the browser waiting on a long chunked WAV and lets JS play each
    segment immediately via AudioContext.decodeAudioData.
    \"""
    import base64

    payload: dict[str, object] = {
        "text": text,
        "streaming": True,
        "format": "wav",
    }
    if reference_id:
        payload["reference_id"] = reference_id
"""
    new_segments = """async def stream_segments(
    text: str = Query(..., min_length=1, max_length=500),
    reference_id: str | None = Query(None),
) -> StreamingResponse:
    \"""
    Convert the upstream chunked WAV response into NDJSON where each line contains
    one complete WAV segment as base64.
    This avoids the browser waiting on a long chunked WAV and lets JS play each
    segment immediately via AudioContext.decodeAudioData.
    \"""
    import base64

    effective_reference_id = (reference_id or DEFAULT_REFERENCE_ID).strip()
    payload: dict[str, object] = {
        "text": text,
        "streaming": True,
        "format": "wav",
        "reference_id": effective_reference_id,
    }
"""
    if old_segments in text:
        text = text.replace(old_segments, new_segments, 1)
    else:
        raise RuntimeError(f"[{path}] не найден блок stream_segments для замены")

    write_text(path, text)
    print(f"OK: {path}")


def main() -> int:
    repo_root = Path.cwd()

    for rel in [
        Path("scripts/warmup_5090.sh"),
        Path("scripts/up_5090.sh"),
        Path("tools/proxy/fish_proxy_pcm.py"),
        Path("tools/proxy/fish_proxy_server.py"),
    ]:
        if not (repo_root / rel).exists():
            print(f"Ошибка: не найден файл {rel}", file=sys.stderr)
            return 1

    try:
        patch_warmup(repo_root)
        patch_up(repo_root)
        patch_pcm_proxy(repo_root)
        patch_stream_proxy(repo_root)
    except Exception as e:
        print(f"\nPATCH FAILED: {e}", file=sys.stderr)
        return 2

    print("\nГотово.")
    print("Бэкапы: *.bak_step2")
    print("После этого перезапусти систему.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
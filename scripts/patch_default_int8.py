#!/usr/bin/env python3
from pathlib import Path
import sys


def find_repo_root(start: Path) -> Path:
    """
    Ищем корень проекта вверх по дереву.
    Признаки корня:
      - есть папка fish_speech
      - есть папка tools
      - есть pyproject.toml
    """
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

PATCHES = {
    "tools/server/api_utils.py": [
        (
            '''    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default="checkpoints/s2-pro",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/s2-pro/codec.pth",
    )''',
            '''    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default="checkpoints/fs-1.2-int8-s2-pro-int8",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/s2-pro/codec.pth",
    )'''
        ),
    ],

    "docker/Dockerfile": [
        (
            'ARG LLAMA_CHECKPOINT_PATH="checkpoints/s2-pro"',
            'ARG LLAMA_CHECKPOINT_PATH="checkpoints/fs-1.2-int8-s2-pro-int8"',
        ),
    ],

    "scripts/up_5090.sh": [
        (
            'CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/s2-pro}"',
            'LLAMA_CHECKPOINTS_DIR="${LLAMA_CHECKPOINTS_DIR:-checkpoints/fs-1.2-int8-s2-pro-int8}"\nDECODER_CHECKPOINT_PATH="${DECODER_CHECKPOINT_PATH:-checkpoints/s2-pro/codec.pth}"',
        ),
        (
            'echo "  CHECKPOINTS_DIR=$CHECKPOINTS_DIR"',
            'echo "  LLAMA_CHECKPOINTS_DIR=$LLAMA_CHECKPOINTS_DIR"\necho "  DECODER_CHECKPOINT_PATH=$DECODER_CHECKPOINT_PATH"',
        ),
        (
            '''if [[ ! -d "$CHECKPOINTS_DIR" ]] || [[ -z "$(ls -A "$CHECKPOINTS_DIR" 2>/dev/null)" ]]; then
  echo "ERROR: checkpoints not found in $CHECKPOINTS_DIR" >&2
  exit 1
fi''',
            '''if [[ ! -d "$LLAMA_CHECKPOINTS_DIR" ]] || [[ -z "$(ls -A "$LLAMA_CHECKPOINTS_DIR" 2>/dev/null)" ]]; then
  echo "ERROR: llama checkpoints not found in $LLAMA_CHECKPOINTS_DIR" >&2
  exit 1
fi

if [[ ! -f "$DECODER_CHECKPOINT_PATH" ]]; then
  echo "ERROR: decoder checkpoint not found at $DECODER_CHECKPOINT_PATH" >&2
  exit 1
fi'''
        ),
        (
            '''    --llama-checkpoint-path "/workspace/$CHECKPOINTS_DIR" \\
    --decoder-checkpoint-path "/workspace/$CHECKPOINTS_DIR/codec.pth" \\''',
            '''    --llama-checkpoint-path "/workspace/$LLAMA_CHECKPOINTS_DIR" \\
    --decoder-checkpoint-path "/workspace/$DECODER_CHECKPOINT_PATH" \\'''
        ),
    ],

    "scripts/run_server_32gb.sh": [
        (
            'CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/s2-pro}"',
            'LLAMA_CHECKPOINTS_DIR="${LLAMA_CHECKPOINTS_DIR:-checkpoints/fs-1.2-int8-s2-pro-int8}"\nDECODER_CHECKPOINT_PATH="${DECODER_CHECKPOINT_PATH:-checkpoints/s2-pro/codec.pth}"',
        ),
        (
            '''  CHECKPOINTS_REL="$CHECKPOINTS_DIR"''',
            '''  LLAMA_CHECKPOINTS_REL="$LLAMA_CHECKPOINTS_DIR"
  DECODER_CHECKPOINT_REL="$DECODER_CHECKPOINT_PATH"'''
        ),
        (
            '''echo "  MOUNT_ROOT=$MOUNT_ROOT  REPO_REL=$REPO_REL  CHECKPOINTS=$CHECKPOINTS_REL"
echo "  IMAGE=$IMAGE  CONTAINER=$CONTAINER  PORT=$PORT  COMPILE=$COMPILE"''',
            '''echo "  MOUNT_ROOT=$MOUNT_ROOT  REPO_REL=$REPO_REL"
echo "  LLAMA_CHECKPOINTS=$LLAMA_CHECKPOINTS_REL"
echo "  DECODER_CHECKPOINT=$DECODER_CHECKPOINT_REL"
echo "  IMAGE=$IMAGE  CONTAINER=$CONTAINER  PORT=$PORT  COMPILE=$COMPILE"'''
        ),
        (
            '''if [[ -z "$WORKSPACE_DIR" ]]; then
  if [[ ! -d "$REPO_ROOT/$CHECKPOINTS_DIR" ]] || [[ -z "$(ls -A "$REPO_ROOT/$CHECKPOINTS_DIR" 2>/dev/null)" ]]; then
    echo "Downloading s2-pro checkpoints to $CHECKPOINTS_DIR ..."
    mkdir -p "$REPO_ROOT/$CHECKPOINTS_DIR"
    docker run --rm \
      --entrypoint /app/.venv/bin/huggingface-cli \
      -v "$REPO_ROOT":/workspace -w /workspace \
      "$IMAGE" \
      download fishaudio/s2-pro --local-dir "$CHECKPOINTS_DIR"
    echo "Download done."
  else
    echo "Checkpoints found at $CHECKPOINTS_DIR"
  fi
else
  if [[ ! -d "$WORKSPACE_DIR/$CHECKPOINTS_DIR" ]] || [[ -z "$(ls -A "$WORKSPACE_DIR/$CHECKPOINTS_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: WORKSPACE_DIR set but $WORKSPACE_DIR/$CHECKPOINTS_DIR not found. Create it or run without WORKSPACE_DIR to download into repo."
    exit 1
  fi
  echo "Checkpoints found at $WORKSPACE_DIR/$CHECKPOINTS_DIR"
fi''',
            '''if [[ -z "$WORKSPACE_DIR" ]]; then
  if [[ ! -d "$REPO_ROOT/$LLAMA_CHECKPOINTS_DIR" ]] || [[ -z "$(ls -A "$REPO_ROOT/$LLAMA_CHECKPOINTS_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: llama checkpoints not found at $LLAMA_CHECKPOINTS_DIR"
    echo "Expected quantized model dir, for example: checkpoints/fs-1.2-int8-s2-pro-int8"
    exit 1
  else
    echo "LLAMA checkpoints found at $LLAMA_CHECKPOINTS_DIR"
  fi

  if [[ ! -f "$REPO_ROOT/$DECODER_CHECKPOINT_PATH" ]]; then
    echo "ERROR: decoder checkpoint not found at $DECODER_CHECKPOINT_PATH"
    exit 1
  else
    echo "Decoder checkpoint found at $DECODER_CHECKPOINT_PATH"
  fi
else
  if [[ ! -d "$WORKSPACE_DIR/$LLAMA_CHECKPOINTS_DIR" ]] || [[ -z "$(ls -A "$WORKSPACE_DIR/$LLAMA_CHECKPOINTS_DIR" 2>/dev/null)" ]]; then
    echo "ERROR: WORKSPACE_DIR set but $WORKSPACE_DIR/$LLAMA_CHECKPOINTS_DIR not found."
    exit 1
  fi

  if [[ ! -f "$WORKSPACE_DIR/$DECODER_CHECKPOINT_PATH" ]]; then
    echo "ERROR: WORKSPACE_DIR set but $WORKSPACE_DIR/$DECODER_CHECKPOINT_PATH not found."
    exit 1
  fi

  echo "LLAMA checkpoints found at $WORKSPACE_DIR/$LLAMA_CHECKPOINTS_DIR"
  echo "Decoder checkpoint found at $WORKSPACE_DIR/$DECODER_CHECKPOINT_PATH"
fi'''
        ),
        (
            '''  --llama-checkpoint-path "/workspace/$CHECKPOINTS_REL" \\
  --decoder-checkpoint-path "/workspace/$CHECKPOINTS_REL/codec.pth" \\''',
            '''  --llama-checkpoint-path "/workspace/$LLAMA_CHECKPOINTS_REL" \\
  --decoder-checkpoint-path "/workspace/$DECODER_CHECKPOINT_REL" \\'''
        ),
    ],
}


def patch_file(rel_path: str, replacements: list[tuple[str, str]]) -> None:
    path = ROOT / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    text = path.read_text(encoding="utf-8")
    original = text

    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
        elif new in text:
            # уже заменено раньше
            pass
        else:
            raise RuntimeError(
                f"Не найден ожидаемый фрагмент в {rel_path}:\n---\n{old}\n---"
            )

    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"[patched] {rel_path}")
    else:
        print(f"[ok] {rel_path} уже был обновлён")


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
    print("Теперь по умолчанию:")
    print("  LLAMA   -> checkpoints/fs-1.2-int8-s2-pro-int8")
    print("  DECODER -> checkpoints/s2-pro/codec.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
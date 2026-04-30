#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path.cwd()
BACKUP_SUFFIX = ".bak_fish_fix"


def _path(rel: str) -> Path:
    return ROOT / rel


def _backup(path: Path) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)


def _read(rel: str) -> str:
    path = _path(rel)
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл: {rel}")
    return path.read_text(encoding="utf-8")


def _write(rel: str, text: str) -> None:
    path = _path(rel)
    _backup(path)
    path.write_text(text, encoding="utf-8")


def replace_once(rel: str, old: str, new: str) -> None:
    text = _read(rel)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{rel}: ожидалось 1 совпадение для блока, найдено {count}. "
            "Файл уже изменён или отличается от code bundle."
        )
    _write(rel, text.replace(old, new, 1))
    print(f"OK: {rel}")


def regex_replace_once(rel: str, pattern: str, replacement: str) -> None:
    text = _read(rel)
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(
            f"{rel}: regex-блок не найден. "
            "Файл уже изменён или отличается от code bundle."
        )
    _write(rel, new_text)
    print(f"OK: {rel}")


def patch_content_sequence() -> None:
    rel = "fish_speech/content_sequence.py"

    old_encoded = """@dataclass(kw_only=True)
class EncodedMessage:
    tokens: torch.Tensor
    labels: torch.Tensor
    vq_mask_tokens: torch.Tensor | None = None
    vq_mask_labels: torch.Tensor | None = None
    vq_parts: list[torch.Tensor]
    vq_require_losses: torch.Tensor | None = None
    audio_parts: list[torch.Tensor]
    audio_masks: torch.Tensor | None = None
    metadata: dict | None = None
"""
    new_encoded = """@dataclass(kw_only=True)
class EncodedMessage:
    tokens: torch.Tensor
    labels: torch.Tensor
    vq_parts: list[torch.Tensor] = field(default_factory=list)
    audio_parts: list[torch.Tensor] = field(default_factory=list)
    vq_mask_tokens: torch.Tensor | None = None
    vq_mask_labels: torch.Tensor | None = None
    vq_require_losses: torch.Tensor | None = None
    audio_masks: torch.Tensor | None = None
    metadata: dict | None = None
"""
    replace_once(rel, old_encoded, new_encoded)

    new_visualize = """    def visualize(
        self: "ContentSequence",
        tokenizer: FishTokenizer,
        ignore_loss_tokens: list[str] | None = None,
        merge_semantic_tokens: bool = False,
        merge_audio_tokens: bool = False,
        use_color: bool = True,
    ):
        # Visualize the encoded sequence with optional color-coded token groups.
        ignore_loss_tokens = ignore_loss_tokens or []
        encoded = self.encode(
            tokenizer, add_shift=False, ignore_loss_tokens=ignore_loss_tokens
        )

        colors = {
            "blue": "\\033[94m",
            "cyan": "\\033[96m",
            "green": "\\033[92m",
            "dark_green": "\\033[32m",
        }
        blue_idx = 0
        green_idx = 0

        def color(name: str) -> str:
            return colors[name] if use_color else ""

        def reset() -> str:
            return "\\033[0m" if use_color else ""

        def print_in_blue(x):
            nonlocal blue_idx
            current = color("blue") if blue_idx % 2 == 0 else color("cyan")
            print(f"{current}{x}{reset()}", end="")
            blue_idx += 1

        def print_in_green(x):
            nonlocal green_idx
            current = color("green") if green_idx % 2 == 0 else color("dark_green")
            print(f"{current}{x}{reset()}", end="")
            green_idx += 1

        def print_group(label: str, contributes_to_loss: bool, count: int):
            val = f"[{label}x{count}]"
            if contributes_to_loss:
                print_in_blue(val)
            else:
                print_in_green(val)

        count_semantic_tokens = 0
        semantic_contributes_to_loss: bool | None = None

        count_audio_tokens = 0
        audio_contributes_to_loss: bool | None = None

        def flush_semantic_group():
            nonlocal count_semantic_tokens, semantic_contributes_to_loss
            if count_semantic_tokens > 0:
                print_group(
                    "<|semantic|>",
                    bool(semantic_contributes_to_loss),
                    count_semantic_tokens,
                )
                count_semantic_tokens = 0
                semantic_contributes_to_loss = None

        def flush_audio_group():
            nonlocal count_audio_tokens, audio_contributes_to_loss
            if count_audio_tokens > 0:
                print_group(
                    "<|audio|>",
                    bool(audio_contributes_to_loss),
                    count_audio_tokens,
                )
                count_audio_tokens = 0
                audio_contributes_to_loss = None

        audio_masks = encoded.audio_masks
        if audio_masks is None or len(audio_masks) != len(encoded.tokens):
            audio_masks = torch.zeros_like(encoded.tokens, dtype=torch.bool)

        for tok, lab, audio_mask in zip(encoded.tokens, encoded.labels, audio_masks):
            token_id = int(tok.item())
            contributes_to_loss = bool(lab != -100)

            if merge_audio_tokens and bool(audio_mask):
                flush_semantic_group()
                if (
                    audio_contributes_to_loss is None
                    or audio_contributes_to_loss == contributes_to_loss
                ):
                    count_audio_tokens += 1
                    audio_contributes_to_loss = contributes_to_loss
                    continue
                flush_audio_group()
                count_audio_tokens = 1
                audio_contributes_to_loss = contributes_to_loss
                continue
            else:
                flush_audio_group()

            if merge_semantic_tokens:
                is_semantic = (
                    tokenizer.semantic_begin_id <= token_id <= tokenizer.semantic_end_id
                )
                if is_semantic and (
                    semantic_contributes_to_loss is None
                    or semantic_contributes_to_loss == contributes_to_loss
                ):
                    count_semantic_tokens += 1
                    semantic_contributes_to_loss = contributes_to_loss
                    continue
                flush_semantic_group()

            val = tokenizer.decode([token_id])
            if not val:
                val = f"<{token_id}>"

            if contributes_to_loss:
                print_in_blue(val)
            else:
                print_in_green(val)

        flush_audio_group()
        flush_semantic_group()
        print()
"""

    regex_replace_once(
        rel,
        r"""    def visualize\(
        self: "ContentSequence",
        tokenizer: FishTokenizer,
        ignore_loss_tokens: list\[str\] = \[\],
        merge_semantic_tokens: bool = False,
    \):
        .*?
        print\(\)
""",
        new_visualize,
    )


def patch_conversation() -> None:
    rel = "fish_speech/conversation.py"
    new_visualize = """    def visualize(
        self: "Conversation",
        tokenizer: PreTrainedTokenizerFast,
        ignore_loss_tokens: list[str] | None = None,
        merge_semantic_tokens: bool = False,
        merge_audio_tokens: bool = False,
        use_color: bool = True,
    ):
        # Visualize the encoded sequence with optional grouping and ANSI colors.
        content_seq = self._build_content_sequence()
        content_seq.visualize(
            tokenizer,
            ignore_loss_tokens=ignore_loss_tokens,
            merge_semantic_tokens=merge_semantic_tokens,
            merge_audio_tokens=merge_audio_tokens,
            use_color=use_color,
        )
"""
    regex_replace_once(
        rel,
        r"""    def visualize\(
        self: "Conversation",
        tokenizer: PreTrainedTokenizerFast,
        ignore_loss_tokens: list\[str\] = \[\],
        merge_semantic_tokens: bool = False,
        merge_audio_tokens: bool = False,
        use_color: bool = True,
    \):
        .*?
        content_seq\.visualize\(
            tokenizer,
            ignore_loss_tokens=ignore_loss_tokens,
            merge_semantic_tokens=merge_semantic_tokens,
        \)
""",
        new_visualize,
    )


def patch_codes() -> None:
    rel = "fish_speech/codec/codes.py"

    replace_once(
        rel,
        """import io
from pathlib import Path
""",
        """import inspect
import io
from pathlib import Path
""",
    )

    helper = """def _torch_load_weights_only(source: str | Path | io.BytesIO) -> Any:
    # Load a torch payload without falling back to unsafe pickle loading.
    try:
        supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        supports_weights_only = True

    if not supports_weights_only:
        raise RuntimeError(
            "This PyTorch version does not support torch.load(weights_only=True). "
            "Upgrade PyTorch or convert code artifacts to a non-pickle format."
        )

    return torch.load(source, map_location="cpu", weights_only=True)


"""
    replace_once(
        rel,
        """def load_codes_pt(
""",
        helper + """def load_codes_pt(
""",
    )

    replace_once(
        rel,
        """    try:
        try:
            loaded = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(source, map_location="cpu")

        payload = _extract_codes_payload(loaded, name=name)
""",
        """    try:
        loaded = _torch_load_weights_only(source)
        payload = _extract_codes_payload(loaded, name=name)
""",
    )


def patch_prompt_builder() -> None:
    rel = "fish_speech/generation/prompt_builder.py"

    replace_once(
        rel,
        """                codes_list: list[torch.Tensor] = []
                chunk_idx = 0
                total_code_frames = 0
""",
        """                need_generated_history = (
                    iterative_prompt
                    and batch_idx < len(committed_segments) - 1
                    and model_cfg.long_form_context_policy != "none"
                )

                codes_list: list[torch.Tensor] = []
                chunk_idx = 0
                total_code_frames = 0
""",
    )

    replace_once(
        rel,
        """                        yield GenerateResponse(
                            action="sample",
                            codes=codes_chunk,
                            text=batch_text,
                        )
                        codes_list.append(codes_chunk.cpu())
""",
        """                        yield GenerateResponse(
                            action="sample",
                            codes=codes_chunk,
                            text=batch_text,
                        )
                        if need_generated_history:
                            codes_list.append(codes_chunk.detach().cpu())
""",
    )

    replace_once(
        rel,
        """                codes = torch.cat(codes_list, dim=1).clone() if codes_list else None

                if iterative_prompt and codes is not None:
                    generated_history.append((batch_text, codes.cpu()))

                codes_list.clear()
                del codes_list

                if codes is not None:
                    del codes
""",
        """                codes = (
                    torch.cat(codes_list, dim=1).contiguous()
                    if codes_list
                    else None
                )

                if need_generated_history and codes is not None:
                    generated_history.append((batch_text, codes))

                codes_list.clear()
                del codes_list

                if codes is not None:
                    del codes
""",
    )


def patch_inference_engine() -> None:
    rel = "fish_speech/inference_engine/__init__.py"

    replace_once(
        rel,
        """            segments = []
            all_codes = []
            seg_idx = 0
""",
        """            segments = []
            all_codes = []
            emitted_audio = False
            seg_idx = 0
""",
    )

    replace_once(
        rel,
        """                                    yield InferenceResult(
                                        code="segment",
                                        audio=(sample_rate, segment),
                                        error=None,
                                    )
                                    segments.append(segment)
""",
        """                                    emitted_audio = True
                                    yield InferenceResult(
                                        code="segment",
                                        audio=(sample_rate, segment),
                                        error=None,
                                    )
""",
    )

    replace_once(
        rel,
        """            if not segments and not all_codes:
""",
        """            if not emitted_audio and not segments and not all_codes:
""",
    )

    replace_once(
        rel,
        """                _mark("stream_done", total_segments=len(segments))
""",
        """                _mark("stream_done", total_segments=seg_idx)
""",
    )


def patch_build_prefix_cache() -> None:
    rel = "fish_speech/tools/build_prefix_cache.py"

    replace_once(
        rel,
        """WORD_RE = re.compile(r"[^\\W_]+", re.UNICODE)
MAX_PREFIX_WORDS = 5
""",
        """WORD_RE = re.compile(r"[^\\W_]+", re.UNICODE)
CACHE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,160}$")
MAX_PREFIX_WORDS = 5
""",
    )

    replace_once(
        rel,
        """def _validate_cache_id(cache_id: str, *, item_label: str) -> str:
    cache_id = cache_id.strip()
    if not cache_id:
        raise ValueError(f"{item_label} has empty cache_id")
    if cache_id in {".", ".."} or "/" in cache_id or "\\\\" in cache_id:
        raise ValueError(
            f"Invalid cache_id {cache_id!r}; path separators are not allowed"
        )
    return cache_id
""",
        """def _validate_cache_id(cache_id: str, *, item_label: str) -> str:
    cache_id = cache_id.strip()
    if not CACHE_ID_RE.fullmatch(cache_id):
        raise ValueError(
            f"{item_label} has invalid cache_id {cache_id!r}; "
            "use only letters, numbers, dot, underscore, and hyphen "
            "(1-160 characters)"
        )
    if cache_id in {".", ".."}:
        raise ValueError(f"Invalid cache_id {cache_id!r}")
    return cache_id
""",
    )


def main() -> int:
    patches = [
        patch_content_sequence,
        patch_conversation,
        patch_codes,
        patch_prompt_builder,
        patch_inference_engine,
        patch_build_prefix_cache,
    ]

    for patch in patches:
        patch()

    print()
    print("Готово. Изменённые файлы сохранены, рядом созданы *.bak_fish_fix резервные копии.")
    print("Рекомендуемая проверка:")
    print("  python -m compileall fish_speech")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

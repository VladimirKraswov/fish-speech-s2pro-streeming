# Internal prompt representation for Fish Speech driver generation.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Union

import numpy as np
import torch

from fish_speech.tokenizer import (
    IM_END_TOKEN,
    MODALITY_TOKENS,
    FishTokenizer,
)


def restore_ndarray(obj, to_tensor: bool = False):
    if isinstance(obj, dict) and "__ndarray__" in obj:
        obj = np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])

    if to_tensor and isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj.copy())

    return obj


@dataclass
class BasePart:
    type: Literal["text", "vq", "audio"] | None = None
    cal_loss: bool | None = None


@dataclass(kw_only=True)
class VQPart(BasePart):
    type = "vq"
    codes: torch.Tensor

    def __post_init__(self: "VQPart"):
        self.type = "vq"
        self.codes = restore_ndarray(self.codes, to_tensor=True)


@dataclass(kw_only=True)
class TextPart(BasePart):
    type = "text"
    text: str | None = None
    tokens: list[int] | torch.Tensor | None = None

    def __post_init__(self: "TextPart"):
        self.type = "text"
        if self.text is None and self.tokens is None:
            raise ValueError("Either text or tokens must be provided")


@dataclass(kw_only=True)
class AudioPart(BasePart):
    type = "audio"
    features: torch.Tensor

    def __post_init__(self: "AudioPart"):
        self.type = "audio"
        self.features = restore_ndarray(self.features, to_tensor=True)


@dataclass(kw_only=True)
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


@dataclass
class ContentSequence:
    """
    Flexible sequence of content parts that supports interleaved multimodal format.
    Example format: <|interleave|><|speaker:1|> TEXT AUDIO <|im_end|><|speaker:2|> TEXT AUDIO <|im_end|>
    """

    parts: list[BasePart] = field(default_factory=list)
    modality: Literal["text", "voice", "interleave"] | None = None
    metadata: dict | None = None

    def __init__(
        self: "ContentSequence",
        parts: list[BasePart | dict] | None = None,
        modality: Literal["text", "voice", "interleave"] | None = None,
        metadata: dict | None = None,
    ):
        self.modality = modality
        self.metadata = metadata or {}

        fixed_parts = []
        for part in parts or []:
            if isinstance(part, dict):
                if part["type"] == "vq":
                    part = VQPart(**part)
                elif part["type"] == "audio":
                    part = AudioPart(**part)
                elif part["type"] == "text":
                    part = TextPart(**part)
                else:
                    raise ValueError(f"Unsupported part type: {part['type']}")
            fixed_parts.append(part)

        self.parts = fixed_parts

        if self.modality and not (
            len(self.parts) > 0
            and isinstance(self.parts[0], TextPart)
            and self.parts[0].text is not None
            and self.parts[0].text.startswith(MODALITY_TOKENS[self.modality])
        ):
            modality_token = MODALITY_TOKENS[self.modality]
            self.parts.insert(0, TextPart(text=modality_token))

    def append(
        self: "ContentSequence",
        part_or_parts: Union[BasePart, List[BasePart]],
        add_end: bool = False,
        speaker: Union[str, int] | None = None,
    ):
        parts_to_add = (
            [part_or_parts] if not isinstance(part_or_parts, list) else part_or_parts
        )

        if speaker is not None:
            speaker_token = f"<|speaker:{speaker}|>"
            self.parts.append(TextPart(text=speaker_token))

        self.parts.extend(parts_to_add)

        if add_end:
            self.parts.append(
                TextPart(text=IM_END_TOKEN, cal_loss=self.parts[-1].cal_loss)
            )

    def encode(
        self: "ContentSequence",
        tokenizer: FishTokenizer,
        add_shift: bool = True,
        ignore_loss_tokens: list[str] | None = None,
    ) -> EncodedMessage:
        ignore_loss_tokens = ignore_loss_tokens or []

        all_tokens = []
        all_labels = []

        vq_parts = []
        vq_masks = []
        vq_require_losses = []

        audio_parts = []
        audio_masks = []

        ignore_loss_token_ids = []
        if ignore_loss_tokens:
            ignore_loss_token_ids = [
                tokenizer.get_token_id(i) for i in ignore_loss_tokens
            ]

        for part in self.parts:
            if isinstance(part, TextPart):
                if part.tokens is None:
                    assert part.text is not None
                    tokens = tokenizer.encode(part.text, add_special_tokens=False)
                    tokens = torch.tensor(tokens, dtype=torch.long)
                else:
                    if isinstance(part.tokens, torch.Tensor):
                        tokens = part.tokens.detach().to(dtype=torch.long)
                    else:
                        tokens = torch.tensor(part.tokens, dtype=torch.long)

                if tokens.ndim != 1:
                    tokens = tokens.reshape(-1)

            elif isinstance(part, VQPart):
                curr_codes = part.codes.clone().to(torch.long)
                if curr_codes.ndim != 2:
                    raise ValueError(
                        f"VQ codes must be 2D [C, T], got shape={tuple(curr_codes.shape)}"
                    )

                semantic_codes = curr_codes[0]

                if (semantic_codes < 0).any() or (
                    semantic_codes >= tokenizer.semantic_map_tensor.numel()
                ).any():
                    raise ValueError("Semantic VQ code out of tokenizer semantic range")

                semantic_map = tokenizer.semantic_map_tensor.to(
                    device=semantic_codes.device
                )
                tokens = semantic_map[semantic_codes].to(torch.long)

                vq_parts.append(curr_codes)
                vq_require_losses.append(part.cal_loss is True)

            elif isinstance(part, AudioPart):
                raise NotImplementedError("AudioPart is not supported by encode() yet")

            else:
                raise ValueError(f"Unsupported part type: {type(part)}")

            all_tokens.append(tokens)

            if isinstance(part, VQPart):
                vq_masks.append(torch.ones_like(tokens, dtype=torch.bool))
                audio_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
            else:
                vq_masks.append(torch.zeros_like(tokens, dtype=torch.bool))
                audio_masks.append(torch.zeros_like(tokens, dtype=torch.bool))

            if getattr(part, "cal_loss", None) is True:
                all_labels.append(tokens.clone())
            else:
                all_labels.append(torch.full_like(tokens, -100))

        if not all_tokens:
            tokens = torch.empty(0, dtype=torch.long)
            labels = torch.empty(0, dtype=torch.long)
            vq_masks_tensor = torch.empty(0, dtype=torch.bool)
            audio_masks_tensor = torch.empty(0, dtype=torch.bool)
        else:
            tokens = torch.cat(all_tokens, dim=0)
            labels = torch.cat(all_labels, dim=0)
            vq_masks_tensor = torch.cat(vq_masks, dim=0)
            audio_masks_tensor = torch.cat(audio_masks, dim=0)

        vq_require_losses_tensor = torch.tensor(vq_require_losses, dtype=torch.bool)

        vq_mask_tokens = vq_masks_tensor
        vq_mask_labels = vq_masks_tensor

        if add_shift and len(tokens) > 0:
            tokens = tokens[:-1]
            labels = labels[1:]
            vq_masks_tensor = vq_masks_tensor[:-1]
            vq_mask_tokens = vq_mask_tokens[:-1]
            vq_mask_labels = vq_mask_labels[1:]
            audio_masks_tensor = audio_masks_tensor[:-1]

        for token_id in ignore_loss_token_ids:
            if token_id is not None:
                labels[labels == token_id] = -100

        return EncodedMessage(
            tokens=tokens,
            labels=labels,
            vq_parts=vq_parts,
            vq_mask_tokens=vq_mask_tokens,
            vq_mask_labels=vq_mask_labels,
            vq_require_losses=vq_require_losses_tensor,
            audio_parts=audio_parts,
            audio_masks=audio_masks_tensor,
            metadata=self.metadata,
        )

    def encode_for_inference(
        self: "ContentSequence",
        tokenizer: FishTokenizer,
        num_codebooks: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        encoded = self.encode(tokenizer, add_shift=False)
        tokens = encoded.tokens

        values = torch.zeros((num_codebooks + 1, len(tokens)), dtype=torch.long)
        values[0] = tokens

        if not encoded.vq_parts and not encoded.audio_parts:
            return values, None, None

        audio_parts = None
        audio_masks = None

        if encoded.vq_parts:
            vq_parts = encoded.vq_parts
            if len(vq_parts) > 1:
                all_vq_codes = torch.cat(vq_parts, dim=1)
            else:
                all_vq_codes = vq_parts[0]

            if all_vq_codes.ndim != 2:
                raise ValueError(f"VQ codes must be 2D, got {all_vq_codes.ndim}D")

            if all_vq_codes.shape[0] != num_codebooks:
                raise ValueError(
                    f"VQ codes and model codebooks mismatch: "
                    f"codes.C={all_vq_codes.shape[0]}, model.C={num_codebooks}"
                )

            if encoded.vq_mask_tokens is None:
                raise ValueError("Missing VQ token mask")

            num_vq_tokens = int(encoded.vq_mask_tokens.sum().item())
            if all_vq_codes.shape[-1] != num_vq_tokens:
                raise ValueError(
                    f"VQ codes and mask mismatch: codes.T={all_vq_codes.shape[-1]}, "
                    f"mask={num_vq_tokens}"
                )

            values[1:, encoded.vq_mask_tokens] = all_vq_codes.to(dtype=torch.long)

        if encoded.audio_parts:
            audio_parts = torch.cat(encoded.audio_parts, dim=0)
            if encoded.audio_masks is None:
                raise ValueError("Missing audio mask")
            audio_masks = encoded.audio_masks[None, :]

        return values, audio_masks, audio_parts

    def visualize(
        self: "ContentSequence",
        tokenizer: FishTokenizer,
        ignore_loss_tokens: list[str] | None = None,
        merge_semantic_tokens: bool = False,
    ):
        ignore_loss_tokens = ignore_loss_tokens or []

        encoded = self.encode(
            tokenizer,
            add_shift=False,
            ignore_loss_tokens=ignore_loss_tokens,
        )

        colors = {
            "blue": "\033[94m",
            "cyan": "\033[96m",
            "green": "\033[92m",
            "dark_green": "\033[32m",
        }
        blue_idx = 0
        green_idx = 0

        def print_in_blue(x):
            nonlocal blue_idx
            color = colors["blue"] if blue_idx % 2 == 0 else colors["cyan"]
            print(f"{color}{x}\033[0m", end="")
            blue_idx += 1

        def print_in_green(x):
            nonlocal green_idx
            color = colors["green"] if green_idx % 2 == 0 else colors["dark_green"]
            print(f"{color}{x}\033[0m", end="")
            green_idx += 1

        def print_semantic_token(label_value, count):
            val = f"[<|semantic|>x{count}]"
            if int(label_value) == -100:
                print_in_green(val)
            else:
                print_in_blue(val)

        count_semantic_tokens = 0
        semantic_label = None

        for tok, lab in zip(encoded.tokens, encoded.labels):
            token_id = int(tok.item())
            label_value = int(lab.item())

            if merge_semantic_tokens:
                is_semantic = (
                    tokenizer.semantic_begin_id
                    <= token_id
                    <= tokenizer.semantic_end_id
                )
                same_label = semantic_label is None or semantic_label == label_value

                if is_semantic and same_label:
                    count_semantic_tokens += 1
                    semantic_label = label_value
                    continue

                if count_semantic_tokens > 0:
                    print_semantic_token(semantic_label, count_semantic_tokens)
                    count_semantic_tokens = 0
                    semantic_label = None

            val = tokenizer.decode([token_id])
            if not val:
                val = f"<{token_id}>"

            if label_value == -100:
                print_in_green(val)
            else:
                print_in_blue(val)

        if merge_semantic_tokens and count_semantic_tokens > 0:
            print_semantic_token(semantic_label, count_semantic_tokens)

        print()
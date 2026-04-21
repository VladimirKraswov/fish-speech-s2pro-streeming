from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np
import torch

from fish_speech.tokenizer import (
    AUDIO_EMBED_TOKEN,
    AUDIO_END_TOKEN,
    AUDIO_START_TOKEN,
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
    cal_loss: bool = False


@dataclass(kw_only=True)
class VQPart(BasePart):
    type = "vq"
    codes: torch.Tensor

    def __post_init__(self: "VQPart"):
        self.type = "vq"
        self.codes = restore_ndarray(self.codes, to_tensor=True)
        if not isinstance(self.codes, torch.Tensor):
            raise TypeError("VQPart.codes must be a torch.Tensor")
        if self.codes.ndim != 2:
            raise ValueError(
                f"VQPart.codes must have shape [num_codebooks, time], got {self.codes.shape}"
            )


@dataclass(kw_only=True)
class TextPart(BasePart):
    type = "text"
    text: str | None = None
    tokens: list[int] | None = None

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
        if not isinstance(self.features, torch.Tensor):
            raise TypeError("AudioPart.features must be a torch.Tensor")
        if self.features.ndim != 2:
            raise ValueError(
                f"AudioPart.features must have shape [time, dim], got {self.features.shape}"
            )


@dataclass(kw_only=True)
class EncodedMessage:
    tokens: torch.Tensor
    labels: torch.Tensor
    vq_mask_tokens: torch.Tensor | None = None
    vq_mask_labels: torch.Tensor | None = None
    vq_parts: list[torch.Tensor] = field(default_factory=list)
    vq_require_losses: torch.Tensor | None = None
    audio_parts: list[torch.Tensor] = field(default_factory=list)
    audio_masks: torch.Tensor | None = None
    metadata: dict | None = None


@dataclass
class ContentSequence:
    """
    Flexible sequence of content parts that supports interleaved multimodal format.
    Example format:
    <|interleave|><|speaker:1|> TEXT AUDIO <|im_end|><|speaker:2|> TEXT AUDIO <|im_end|>
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
        part_or_parts: Union[BasePart, list[BasePart]],
        add_end: bool = False,
        speaker: Union[str, int] | None = None,
    ):
        """
        Append a part or list of parts to the sequence.

        Args:
            part_or_parts: A single part or list of parts to add
            add_end: Whether to add the IM_END_TOKEN after these parts
            speaker: Optional speaker identifier (name or ID) to add before the parts
        """
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

    @staticmethod
    def _require_token_id(tokenizer: FishTokenizer, token: str) -> int:
        token_id = tokenizer.get_token_id(token)
        if token_id is None or token_id < 0:
            raise ValueError(f"Tokenizer does not contain required token: {token}")
        return int(token_id)

    @staticmethod
    def _map_semantic_codes_to_token_ids(
        tokenizer: FishTokenizer,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        if codes.ndim != 1:
            raise ValueError(f"Expected 1D semantic code tensor, got shape {codes.shape}")

        codes = codes.to(dtype=torch.long, device="cpu")

        if codes.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        if torch.any(codes < 0) or torch.any(codes >= 4096):
            bad_min = int(codes.min().item())
            bad_max = int(codes.max().item())
            raise ValueError(
                f"Semantic code ids must be in [0, 4095], got range [{bad_min}, {bad_max}]"
            )

        if hasattr(tokenizer, "semantic_id_to_token_id"):
            unique_codes = torch.unique(codes).tolist()
            missing = [
                int(code)
                for code in unique_codes
                if int(code) not in tokenizer.semantic_id_to_token_id
            ]
            if missing:
                preview = ", ".join(map(str, missing[:8]))
                raise ValueError(
                    f"Tokenizer is missing semantic token ids for codes: {preview}"
                )

        mapping = tokenizer.semantic_map_tensor.to(device="cpu")
        return mapping[codes]

    def encode(
        self: "ContentSequence",
        tokenizer: FishTokenizer,
        add_shift: bool = True,
        ignore_loss_tokens: list[str] = [],
    ) -> EncodedMessage:
        """
        Encode the sequence parts into tokens for the model.

        Args:
            tokenizer: The tokenizer to use
            add_shift: Whether to shift tokens for next-token prediction
            ignore_loss_tokens: List of token strings to ignore when calculating loss

        Returns:
            EncodedMessage with tensors ready for the model
        """
        all_tokens: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        vq_parts: list[torch.Tensor] = []
        vq_masks: list[torch.Tensor] = []
        vq_require_losses: list[bool] = []

        audio_parts: list[torch.Tensor] = []
        audio_masks: list[torch.Tensor] = []

        ignore_loss_token_ids = []
        if ignore_loss_tokens:
            ignore_loss_token_ids = [
                tokenizer.get_token_id(i) for i in ignore_loss_tokens
            ]

        audio_start_id = self._require_token_id(tokenizer, AUDIO_START_TOKEN)
        audio_embed_id = self._require_token_id(tokenizer, AUDIO_EMBED_TOKEN)
        audio_end_id = self._require_token_id(tokenizer, AUDIO_END_TOKEN)

        for part in self.parts:
            if isinstance(part, TextPart):
                if part.tokens is None:
                    assert part.text is not None
                    tokens = tokenizer.encode(part.text, add_special_tokens=False)
                else:
                    tokens = part.tokens

                tokens_tensor = torch.tensor(tokens, dtype=torch.long, device="cpu")

                vq_masks.append(torch.zeros_like(tokens_tensor, dtype=torch.bool))
                audio_masks.append(torch.zeros_like(tokens_tensor, dtype=torch.bool))

            elif isinstance(part, VQPart):
                curr_codes = part.codes.clone().to(dtype=torch.long, device="cpu")
                semantic_tokens = self._map_semantic_codes_to_token_ids(
                    tokenizer, curr_codes[0]
                )

                tokens_tensor = semantic_tokens
                vq_parts.append(curr_codes)
                vq_require_losses.append(part.cal_loss)

                vq_masks.append(torch.ones_like(tokens_tensor, dtype=torch.bool))
                audio_masks.append(torch.zeros_like(tokens_tensor, dtype=torch.bool))

            elif isinstance(part, AudioPart):
                features = part.features.clone().to(device="cpu")
                frame_count = int(features.shape[0])

                tokens_tensor = torch.tensor(
                    [audio_start_id, *([audio_embed_id] * frame_count), audio_end_id],
                    dtype=torch.long,
                    device="cpu",
                )

                audio_mask = torch.zeros_like(tokens_tensor, dtype=torch.bool)
                if frame_count > 0:
                    audio_mask[1:-1] = True

                vq_masks.append(torch.zeros_like(tokens_tensor, dtype=torch.bool))
                audio_masks.append(audio_mask)
                audio_parts.append(features)

            else:
                raise ValueError(f"Unsupported part type: {type(part)}")

            all_tokens.append(tokens_tensor)

            if part.cal_loss and not isinstance(part, AudioPart):
                all_labels.append(tokens_tensor.clone())
            else:
                all_labels.append(torch.full_like(tokens_tensor, -100))

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

        for i in ignore_loss_token_ids:
            if i is not None:
                labels[labels == i] = -100

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

        if (encoded.vq_parts is None or len(encoded.vq_parts) == 0) and (
            encoded.audio_parts is None or len(encoded.audio_parts) == 0
        ):
            return values, None, None

        audio_parts = None
        audio_masks = None

        if encoded.vq_parts is not None and len(encoded.vq_parts) > 0:
            if len(encoded.vq_parts) > 1:
                all_vq_codes = torch.cat(encoded.vq_parts, dim=1)
            else:
                all_vq_codes = encoded.vq_parts[0]

            values[1:, encoded.vq_mask_tokens] = all_vq_codes.to(dtype=torch.long)

        if encoded.audio_parts is not None and len(encoded.audio_parts) > 0:
            audio_parts = torch.cat(encoded.audio_parts, dim=0)
            audio_masks = encoded.audio_masks[None, :]

        return values, audio_masks, audio_parts

    def visualize(
        self: "ContentSequence",
        tokenizer: FishTokenizer,
        ignore_loss_tokens: list[str] = [],
        merge_semantic_tokens: bool = False,
    ):
        """
        Visualize the encoded sequence with color-coded tokens.
        Blue/cyan tokens contribute to loss, green tokens do not.
        """
        encoded = self.encode(
            tokenizer, add_shift=False, ignore_loss_tokens=ignore_loss_tokens
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

        def print_semantic_token(x, count):
            val = f"[<|semantic|>x{count}]"
            if x == -100:
                print_in_green(val)
            else:
                print_in_blue(val)

        count_semantic_tokens = 0
        semantic_label = None

        for tok, lab in zip(encoded.tokens, encoded.labels):
            token_id = int(tok.item())

            if merge_semantic_tokens:
                if (
                    tokenizer.semantic_begin_id <= token_id <= tokenizer.semantic_end_id
                    and (semantic_label is None or semantic_label == lab)
                ):
                    count_semantic_tokens += 1
                    semantic_label = lab
                    continue
                elif count_semantic_tokens > 0:
                    print_semantic_token(semantic_label, count_semantic_tokens)
                    count_semantic_tokens = 0
                    semantic_label = None

            val = tokenizer.decode([token_id])
            if not val:
                val = f"<{token_id}>"

            if lab == -100:
                print_in_green(val)
            else:
                print_in_blue(val)

        if merge_semantic_tokens and count_semantic_tokens > 0:
            print_semantic_token(semantic_label, count_semantic_tokens)

        print()
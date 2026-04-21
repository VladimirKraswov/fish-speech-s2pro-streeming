import io
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


def _stem_for_codes_file(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix(".codes") if stem.endswith(".codes") else stem


class ReferenceLoader:
    def __init__(self) -> None:
        """
        Component of the TTSInferenceEngine class.
        Loads and manages the cache for the reference audio and text.
        """
        self.ref_by_id: dict = {}
        self.ref_by_hash: dict = {}

        self.decoder_model: DAC
        self.encode_reference: Callable

        try:
            backends = torchaudio.list_audio_backends()
        except AttributeError:
            backends = []

        if "ffmpeg" in backends:
            self.backend = "ffmpeg"
        else:
            self.backend = "soundfile"

    def load_by_id(
        self,
        id: str,
        use_cache: Literal["on", "off"],
    ) -> Tuple:
        ref_folder = Path("references") / id
        ref_folder.mkdir(parents=True, exist_ok=True)

        ref_audios = list_files(
            ref_folder,
            AUDIO_EXTENSIONS,
            recursive=True,
            sort=False,
        )
        ref_codes = list(ref_folder.glob("*.codes.pt"))

        stems = {p.stem for p in ref_audios} | {_stem_for_codes_file(p) for p in ref_codes}
        stems = sorted(stems)

        if use_cache == "off" or id not in self.ref_by_id:
            prompt_tokens = []
            prompt_texts = []

            for stem in stems:
                lab_path = ref_folder / f"{stem}.lab"
                codes_path = ref_folder / f"{stem}.codes.pt"
                audio_path = next(
                    (
                        ref_folder / f"{stem}{ext}"
                        for ext in AUDIO_EXTENSIONS
                        if (ref_folder / f"{stem}{ext}").exists()
                    ),
                    None,
                )

                if not lab_path.exists():
                    logger.warning("Reference stem {} missing .lab, skipping", stem)
                    continue

                prompt_texts.append(read_ref_text(str(lab_path)))

                if codes_path.exists():
                    loaded = torch.load(
                        codes_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    prompt_tokens.append(
                        loaded if isinstance(loaded, torch.Tensor) else loaded[0]
                    )
                    logger.info(
                        "Loaded pre-encoded reference {} from {}",
                        stem,
                        codes_path.name,
                    )
                elif audio_path is not None:
                    prompt_tokens.append(
                        self.encode_reference(
                            reference_audio=audio_to_bytes(str(audio_path)),
                            enable_reference_audio=True,
                        )
                    )
                else:
                    logger.warning(
                        "Reference stem {} has .lab but no .codes.pt or audio, skipping",
                        stem,
                    )
                    prompt_texts.pop()

            self.ref_by_id[id] = (prompt_tokens, prompt_texts)
        else:
            logger.info("Use same references")
            prompt_tokens, prompt_texts = self.ref_by_id[id]

        return prompt_tokens, prompt_texts

    def load_by_hash(
        self,
        references: list[ServeReferenceAudio],
        use_cache: Literal["on", "off"],
    ) -> Tuple:
        audio_hashes = [sha256(ref.audio).hexdigest() for ref in references]

        cache_used = False
        prompt_tokens, prompt_texts = [], []

        for i, ref in enumerate(references):
            if use_cache == "off" or audio_hashes[i] not in self.ref_by_hash:
                prompt_tokens.append(
                    self.encode_reference(
                        reference_audio=ref.audio,
                        enable_reference_audio=True,
                    )
                )
                prompt_texts.append(ref.text)
                self.ref_by_hash[audio_hashes[i]] = (prompt_tokens[-1], ref.text)
            else:
                cached_token, cached_text = self.ref_by_hash[audio_hashes[i]]
                prompt_tokens.append(cached_token)
                prompt_texts.append(cached_text)
                cache_used = True

        if cache_used:
            logger.info("Use same references")

        return prompt_tokens, prompt_texts

    def load_audio(self, reference_audio: bytes | bytearray | memoryview | str | Path, sr: int):
        """
        Load audio data either from raw bytes or from a filesystem path.
        """
        if isinstance(reference_audio, (bytes, bytearray, memoryview)):
            source = io.BytesIO(bytes(reference_audio))
        elif isinstance(reference_audio, (str, Path)):
            source_path = Path(reference_audio)
            if not source_path.exists():
                raise FileNotFoundError(f"Reference audio file not found: {source_path}")
            source = str(source_path)
        else:
            raise TypeError(
                "reference_audio must be bytes, bytearray, memoryview, str or Path"
            )

        waveform, original_sr = torchaudio.load(source, backend=self.backend)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if original_sr != sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr,
                new_freq=sr,
            )
            waveform = resampler(waveform)

        audio = waveform.squeeze(0).cpu().numpy()
        return audio

    def list_reference_ids(self) -> list[str]:
        """
        List all valid reference IDs.

        Valid reference:
        - at least one <stem>.lab
        - and either corresponding audio <stem>.* or pre-encoded <stem>.codes.pt
        """
        ref_base_path = Path("references")
        if not ref_base_path.exists():
            return []

        valid_ids = []

        for ref_dir in ref_base_path.iterdir():
            if not ref_dir.is_dir():
                continue

            audio_files = list_files(
                ref_dir,
                AUDIO_EXTENSIONS,
                recursive=False,
                sort=False,
            )
            code_files = list(ref_dir.glob("*.codes.pt"))

            stems = {p.stem for p in audio_files} | {_stem_for_codes_file(p) for p in code_files}
            if not stems:
                continue

            has_valid_pair = False
            for stem in stems:
                lab_file = ref_dir / f"{stem}.lab"
                code_file = ref_dir / f"{stem}.codes.pt"
                audio_exists = any(
                    (ref_dir / f"{stem}{ext}").exists() for ext in AUDIO_EXTENSIONS
                )
                if lab_file.exists() and (audio_exists or code_file.exists()):
                    has_valid_pair = True
                    break

            if has_valid_pair:
                valid_ids.append(ref_dir.name)

        return sorted(valid_ids)

    def add_reference_encoded(
        self, id: str, codes_bytes: bytes, lab_text: str, stem: str | None = None
    ) -> Literal["created", "updated", "unchanged"]:
        """
        Add or update a pre-encoded reference.
        Writes references/<id>/<stem>.codes.pt and <stem>.lab.
        """
        import re

        if not re.match(r"^[a-zA-Z0-9\-_ ]+$", id):
            raise ValueError(
                "Reference ID contains invalid characters. Only alphanumeric, hyphens, underscores, and spaces are allowed."
            )
        if len(id) > 255:
            raise ValueError(
                "Reference ID is too long. Maximum length is 255 characters."
            )

        stem = stem or id
        if not re.match(r"^[a-zA-Z0-9\-_ ]+$", stem):
            raise ValueError("Stem contains invalid characters.")

        ref_dir = Path("references") / id
        payload = codes_bytes + lab_text.encode("utf-8")
        content_hash = sha256(payload).hexdigest()
        hash_file = ref_dir / f".{stem}.hash"

        if ref_dir.exists() and hash_file.exists():
            if hash_file.read_text(encoding="utf-8").strip() == content_hash:
                logger.info(
                    "Reference %s/%s unchanged (hash match), skip write",
                    id,
                    stem,
                )
                return "unchanged"

        existed_before = ref_dir.exists() and (ref_dir / f"{stem}.codes.pt").exists()
        ref_dir.mkdir(parents=True, exist_ok=True)

        (ref_dir / f"{stem}.codes.pt").write_bytes(codes_bytes)
        (ref_dir / f"{stem}.lab").write_text(lab_text, encoding="utf-8")
        hash_file.write_text(content_hash, encoding="utf-8")

        if id in self.ref_by_id:
            del self.ref_by_id[id]

        status = "updated" if existed_before else "created"
        logger.info("Reference %s/%s %s", id, stem, status)
        return status

    def add_reference(self, id: str, wav_file_path: str, reference_text: str) -> None:
        """
        Add a new reference voice by creating a new directory and copying files.
        """
        import re
        import shutil

        if not re.match(r"^[a-zA-Z0-9\-_ ]+$", id):
            raise ValueError(
                "Reference ID contains invalid characters. Only alphanumeric, hyphens, underscores, and spaces are allowed."
            )

        if len(id) > 255:
            raise ValueError(
                "Reference ID is too long. Maximum length is 255 characters."
            )

        ref_dir = Path("references") / id
        if ref_dir.exists():
            raise FileExistsError(f"Reference ID '{id}' already exists")

        audio_path = Path(wav_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_file_path}")

        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}"
            )

        try:
            ref_dir.mkdir(parents=True, exist_ok=False)

            target_audio_path = ref_dir / f"sample{audio_path.suffix}"
            shutil.copy2(audio_path, target_audio_path)

            lab_path = ref_dir / "sample.lab"
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(reference_text)

            if id in self.ref_by_id:
                del self.ref_by_id[id]

            logger.info(f"Successfully added reference voice with ID: {id}")

        except Exception as e:
            if ref_dir.exists():
                shutil.rmtree(ref_dir)
            raise e

    def delete_reference(self, id: str) -> None:
        """
        Delete a reference voice by removing its directory and files.
        """
        import shutil

        ref_dir = Path("references") / id
        if not ref_dir.exists():
            raise FileNotFoundError(f"Reference ID '{id}' does not exist")

        try:
            shutil.rmtree(ref_dir)

            if id in self.ref_by_id:
                del self.ref_by_id[id]

            logger.info(f"Successfully deleted reference voice with ID: {id}")

        except Exception as e:
            logger.error(f"Failed to delete reference '{id}': {e}")
            raise OSError(f"Failed to delete reference '{id}': {e}")
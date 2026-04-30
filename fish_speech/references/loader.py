# Driver reference loading, caching, and filesystem-backed reference store.
from __future__ import annotations

import io
import re
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal, Tuple

import torch
import torchaudio
from loguru import logger

from fish_speech.codec.codes import (
    load_codes_pt,
    save_codes_pt,
    validate_codes_for_decoder,
)
from fish_speech.driver.types import DriverReference
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.references.cache import ReferenceCache
from fish_speech.references.store import get_references_dir
from fish_speech.utils.file import (
    AUDIO_EXTENSIONS,
    audio_to_bytes,
    list_files,
    read_ref_text,
)


REFERENCE_ID_RE = re.compile(r"^[a-zA-Z0-9\-_ ]+$")


def _validate_reference_id_value(value: str, *, name: str = "Reference ID") -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} cannot be empty")
    if not REFERENCE_ID_RE.match(value):
        raise ValueError(
            f"{name} contains invalid characters. Only alphanumeric, hyphens, "
            "underscores, and spaces are allowed."
        )
    if len(value) > 255:
        raise ValueError(f"{name} is too long. Maximum length is 255 characters.")


class ReferenceLoader:
    def __init__(self) -> None:
        """
        Component of the TTSInferenceEngine class.
        Loads and manages the cache for the reference audio and text.
        """
        self.cache = ReferenceCache()
        self.ref_by_id = self.cache.by_id
        self.ref_by_hash = self.cache.by_hash
        self.references_dir = get_references_dir()

        self.decoder_model: DAC
        self.encode_reference: Callable

        backends = torchaudio.list_audio_backends()
        if "ffmpeg" in backends:
            self.backend = "ffmpeg"
        else:
            self.backend = "soundfile"

    def load_by_id(
        self,
        id: str,
        use_cache: Literal["on", "off"],
    ) -> Tuple:
        """
        Load references by ID.

        Each reference can be:
        - audio + .lab, encoded at load time;
        - *.codes.pt + .lab, loaded pre-encoded.
        """
        _validate_reference_id_value(id)

        ref_folder = self.references_dir / id
        if not ref_folder.is_dir():
            raise FileNotFoundError(f"Reference ID '{id}' does not exist")

        ref_audios = list_files(
            ref_folder, AUDIO_EXTENSIONS, recursive=False, sort=False
        )
        ref_codes = list(ref_folder.glob("*.codes.pt"))

        def _stem_for_codes(p: Path) -> str:
            s = p.stem
            return s.removesuffix(".codes") if s.endswith(".codes") else s

        stems = {p.stem for p in ref_audios} | {_stem_for_codes(p) for p in ref_codes}
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
                    try:
                        loaded = load_codes_pt(
                            codes_path, name=f"reference {id}/{stem}"
                        )
                        loaded = validate_codes_for_decoder(
                            loaded,
                            getattr(self, "decoder_model", None),
                            name=f"reference {id}/{stem}",
                        )

                        prompt_tokens.append(loaded)
                        logger.info(
                            "Loaded pre-encoded reference {} from {} (shape={})",
                            stem,
                            codes_path.name,
                            loaded.shape,
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to load pre-encoded reference from {}: {}",
                            codes_path,
                            e,
                        )
                        raise ValueError(
                            f"Invalid pre-encoded reference at {codes_path}: {e}"
                        ) from e

                elif audio_path is not None:
                    encoded = self.encode_reference(
                        reference_audio=audio_to_bytes(str(audio_path)),
                        enable_reference_audio=True,
                    )
                    encoded = validate_codes_for_decoder(
                        encoded,
                        getattr(self, "decoder_model", None),
                        name=f"reference {id}/{stem}",
                    )
                    prompt_tokens.append(encoded)

                else:
                    logger.warning(
                        "Reference stem {} has .lab but no .codes.pt or audio, skipping",
                        stem,
                    )
                    prompt_texts.pop()

            if not prompt_tokens:
                raise ValueError(
                    f"Reference ID '{id}' has no valid (audio/.codes.pt + .lab) pairs"
                )

            self.ref_by_id[id] = (prompt_tokens, prompt_texts)

        else:
            logger.info("Use same references")
            prompt_tokens, prompt_texts = self.ref_by_id[id]

        return prompt_tokens, prompt_texts

    def load_by_hash(
        self,
        references: list[DriverReference],
        use_cache: Literal["on", "off"],
    ) -> Tuple:
        """
        Load request-provided references by audio hash.

        Cache key is audio-only. Text stays request-local so callers can reuse
        the same audio reference with different prompt transcripts.
        """
        audio_hashes = [sha256(ref.audio).hexdigest() for ref in references]

        cache_used = False
        prompt_tokens, prompt_texts = [], []

        for i, ref in enumerate(references):
            audio_hash = audio_hashes[i]

            if use_cache == "off" or audio_hash not in self.ref_by_hash:
                prompt_token = self.encode_reference(
                    reference_audio=ref.audio,
                    enable_reference_audio=True,
                )
                prompt_token = validate_codes_for_decoder(
                    prompt_token,
                    getattr(self, "decoder_model", None),
                    name=f"reference hash {audio_hash[:12]}",
                )
                prompt_tokens.append(prompt_token)
                prompt_texts.append(ref.text)
                self.ref_by_hash[audio_hash] = prompt_token

            else:
                cached_value = self.ref_by_hash[audio_hash]
                if isinstance(cached_value, (tuple, list)):
                    if not cached_value:
                        raise ValueError(
                            f"Cached reference for hash {audio_hash} is empty"
                        )
                    cached_token = cached_value[0]
                    self.ref_by_hash[audio_hash] = cached_token
                else:
                    cached_token = cached_value

                cached_token = validate_codes_for_decoder(
                    cached_token,
                    getattr(self, "decoder_model", None),
                    name=f"cached reference hash {audio_hash[:12]}",
                )
                prompt_tokens.append(cached_token)
                prompt_texts.append(ref.text)
                cache_used = True

        if cache_used:
            logger.info("Use same references")

        return prompt_tokens, prompt_texts

    def load_audio(self, reference_audio: bytes | str | bytearray | Path, sr: int):
        """
        Load audio data from a file path or bytes.
        """
        if isinstance(reference_audio, (bytes, bytearray)):
            reference_audio = io.BytesIO(reference_audio)
        elif isinstance(reference_audio, (str, Path)):
            audio_path = Path(reference_audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Reference audio not found: {audio_path}")
            reference_audio = audio_path
        else:
            try:
                reference_audio = io.BytesIO(reference_audio)
            except Exception:
                raise TypeError(
                    f"Unsupported reference_audio type: {type(reference_audio)}"
                )

        waveform, original_sr = torchaudio.load(reference_audio, backend=self.backend)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if original_sr != sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=sr
            )
            waveform = resampler(waveform)

        audio = waveform.squeeze().numpy()
        return audio

    def list_reference_ids(self) -> list[str]:
        """
        List all valid reference IDs.

        Valid reference:
        - audio + .lab;
        - or .codes.pt + .lab.
        """
        ref_base_path = self.references_dir
        if not ref_base_path.exists():
            return []

        valid_ids = []

        for ref_dir in ref_base_path.iterdir():
            if not ref_dir.is_dir():
                continue

            audio_files = list_files(
                ref_dir, AUDIO_EXTENSIONS, recursive=False, sort=False
            )
            code_files = list(ref_dir.glob("*.codes.pt"))

            if not audio_files and not code_files:
                continue

            def _stem_for_codes(p: Path) -> str:
                s = p.stem
                return s.removesuffix(".codes") if s.endswith(".codes") else s

            stems = {p.stem for p in audio_files} | {
                _stem_for_codes(p) for p in code_files
            }

            has_valid_pair = any((ref_dir / f"{stem}.lab").exists() for stem in stems)

            if has_valid_pair:
                valid_ids.append(ref_dir.name)

        return sorted(valid_ids)

    def add_reference_encoded(
        self,
        id: str,
        codes_bytes: bytes,
        lab_text: str,
        stem: str | None = None,
    ) -> Literal["created", "updated", "unchanged"]:
        """
        Add or update a pre-encoded reference.

        Writes:
        - references/<id>/<stem>.codes.pt
        - references/<id>/<stem>.lab

        Skips write when content hash matches.
        """
        _validate_reference_id_value(id)

        stem = stem or id
        _validate_reference_id_value(stem, name="Stem")

        ref_dir = self.references_dir / id
        payload = codes_bytes + lab_text.encode("utf-8")
        content_hash = sha256(payload).hexdigest()
        hash_file = ref_dir / f".{stem}.hash"

        if ref_dir.exists() and hash_file.exists():
            if hash_file.read_text(encoding="utf-8").strip() == content_hash:
                logger.info(
                    "Reference {}/{} unchanged (hash match), skip write", id, stem
                )
                return "unchanged"

        codes = load_codes_pt(codes_bytes, name=f"reference {id}/{stem}")
        codes = validate_codes_for_decoder(
            codes,
            getattr(self, "decoder_model", None),
            name=f"reference {id}/{stem}",
        )

        existed_before = ref_dir.exists() and (ref_dir / f"{stem}.codes.pt").exists()
        ref_dir.mkdir(parents=True, exist_ok=True)

        save_codes_pt(codes, ref_dir / f"{stem}.codes.pt", name=f"reference {id}/{stem}")
        (ref_dir / f"{stem}.lab").write_text(lab_text, encoding="utf-8")
        hash_file.write_text(content_hash, encoding="utf-8")

        if id in self.ref_by_id:
            del self.ref_by_id[id]

        status = "updated" if existed_before else "created"
        logger.info("Reference {}/{} {}", id, stem, status)
        return status

    def add_reference(self, id: str, wav_file_path: str, reference_text: str) -> None:
        """
        Add a new reference voice by creating a new directory and copying files.

        Args:
            id: Reference ID (directory name)
            wav_file_path: Path to the audio file to copy
            reference_text: Text content for the .lab file

        Raises:
            FileExistsError: If the reference ID already exists
            FileNotFoundError: If the audio file doesn't exist
            OSError: If file operations fail
        """
        _validate_reference_id_value(id)

        ref_dir = self.references_dir / id
        if ref_dir.exists():
            raise FileExistsError(f"Reference ID '{id}' already exists")

        audio_path = Path(wav_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_file_path}")

        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported formats: {', '.join(AUDIO_EXTENSIONS)}"
            )

        try:
            ref_dir.mkdir(parents=True, exist_ok=False)

            target_audio_path = ref_dir / f"sample{audio_path.suffix}"

            import shutil

            shutil.copy2(audio_path, target_audio_path)

            lab_path = ref_dir / "sample.lab"
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(reference_text)

            if id in self.ref_by_id:
                del self.ref_by_id[id]

            logger.info(f"Successfully added reference voice with ID: {id}")

        except Exception as e:
            if ref_dir.exists():
                import shutil

                shutil.rmtree(ref_dir)
            raise e

    def delete_reference(self, id: str) -> None:
        """
        Delete a reference voice by removing its directory and files.

        Args:
            id: Reference ID (directory name) to delete

        Raises:
            FileNotFoundError: If the reference ID doesn't exist
            OSError: If file operations fail
        """
        _validate_reference_id_value(id)

        ref_dir = self.references_dir / id
        if not ref_dir.exists():
            raise FileNotFoundError(f"Reference ID '{id}' does not exist")

        try:
            import shutil

            shutil.rmtree(ref_dir)

            if id in self.ref_by_id:
                del self.ref_by_id[id]

            logger.info(f"Successfully deleted reference voice with ID: {id}")

        except Exception as e:
            logger.error(f"Failed to delete reference '{id}': {e}")
            raise OSError(f"Failed to delete reference '{id}': {e}")
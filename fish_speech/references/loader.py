# Driver reference loading, caching, and filesystem-backed reference store.
import io
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal, Tuple

import torch
import torchaudio
from loguru import logger

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

        # Make Pylance happy (attribut/method not defined...)
        self.decoder_model: DAC
        self.encode_reference: Callable

        # Define the torchaudio backend
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
        # Load the references audio and text by id.
        # Each reference can be: (a) .wav + .lab → encode at load, or (b) .codes.pt + .lab → load pre-encoded (no encoder run).
        ref_folder = self.references_dir / id
        if not ref_folder.is_dir():
            raise FileNotFoundError(f"Reference ID '{id}' does not exist")

        ref_audios = list_files(
            ref_folder, AUDIO_EXTENSIONS, recursive=True, sort=False
        )
        ref_codes = list(ref_folder.glob("*.codes.pt"))

        # .codes.pt filename stem is "en.codes" (Path.stem); we want logical stem "en" for en.lab / en.codes.pt
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
                    # Pre-encoded: load from disk (no encoder run). File must be tensor shape (num_codebooks, T) from DAC encode.
                    # map_location="cpu" = put tensor in RAM (worker will .to(device) later). Loading to GPU here would save one tiny copy but is negligible for KB-sized tensors.
                    try:
                        loaded = torch.load(
                            codes_path, map_location="cpu", weights_only=True
                        )
                        if isinstance(loaded, (tuple, list)):
                            loaded = loaded[0]

                        if not isinstance(loaded, torch.Tensor):
                            raise TypeError(
                                f"Expected torch.Tensor, got {type(loaded).__name__}"
                            )

                        # Standardize to [num_codebooks, T]
                        if loaded.ndim == 3:
                            if loaded.shape[0] == 1:
                                loaded = loaded[0]
                            else:
                                raise ValueError(
                                    f"Unexpected 3D tensor shape {loaded.shape}, "
                                    f"expected [1, num_codebooks, T] or [num_codebooks, T]"
                                )

                        if loaded.ndim != 2:
                            raise ValueError(
                                f"Unexpected tensor ndim {loaded.ndim}, shape {loaded.shape}, "
                                f"expected [num_codebooks, T]"
                            )

                        if loaded.size(-1) == 0:
                            raise ValueError("Acoustic tensor is empty (T=0)")

                        # Final validation of num_codebooks if possible
                        if hasattr(self, "decoder_model") and hasattr(
                            self.decoder_model, "quantizer"
                        ):
                            from fish_speech.models.dac.rvq import (
                                DownsampleResidualVectorQuantize,
                            )

                            quantizer = self.decoder_model.quantizer
                            if isinstance(quantizer, DownsampleResidualVectorQuantize):
                                # expected = residual_n_codebooks + 1 semantic codebook
                                expected_cb = quantizer.quantizer.n_codebooks + 1
                            else:
                                expected_cb = getattr(quantizer, "n_codebooks", None)

                            if expected_cb is not None and loaded.shape[0] != expected_cb:
                                raise ValueError(
                                    f"Reference {stem} codebook count mismatch at {codes_path}: "
                                    f"got {loaded.shape[0]}, expected {expected_cb}. Shape: {loaded.shape}"
                                )

                        prompt_tokens.append(loaded.long().cpu())
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
            if not prompt_tokens:
                raise ValueError(
                    f"Reference ID '{id}' has no valid (audio/.codes.pt + .lab) pairs"
                )

            self.ref_by_id[id] = (prompt_tokens, prompt_texts)

        else:
            # Reuse already encoded references
            logger.info("Use same references")
            prompt_tokens, prompt_texts = self.ref_by_id[id]

        return prompt_tokens, prompt_texts

    def load_by_hash(
        self,
        references: list[DriverReference],
        use_cache: Literal["on", "off"],
    ) -> Tuple:
        # Load the references audio and text by hash
        audio_hashes = [sha256(ref.audio).hexdigest() for ref in references]

        cache_used = False
        prompt_tokens, prompt_texts = [], []
        for i, ref in enumerate(references):
            if use_cache == "off" or audio_hashes[i] not in self.ref_by_hash:
                # If the references are not already loaded, encode them
                prompt_tokens.append(
                    self.encode_reference(
                        reference_audio=ref.audio,
                        enable_reference_audio=True,
                    )
                )
                prompt_texts.append(ref.text)
                self.ref_by_hash[audio_hashes[i]] = (prompt_tokens[-1], ref.text)

            else:
                # Reuse already encoded references
                cached_token, cached_text = self.ref_by_hash[audio_hashes[i]]
                prompt_tokens.append(cached_token)
                prompt_texts.append(cached_text)
                cache_used = True

        if cache_used:
            logger.info("Use same references")

        return prompt_tokens, prompt_texts

    def load_audio(self, reference_audio: bytes | str | bytearray | Path, sr: int):
        """
        Load the audio data from a file or bytes.
        """
        if isinstance(reference_audio, (bytes, bytearray)):
            reference_audio = io.BytesIO(reference_audio)
        elif isinstance(reference_audio, (str, Path)):
            audio_path = Path(reference_audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"Reference audio not found: {audio_path}")
            reference_audio = audio_path
        else:
            # Fallback for other potential types, try wrapping in BytesIO
            try:
                reference_audio = io.BytesIO(reference_audio)
            except Exception:
                raise TypeError(f"Unsupported reference_audio type: {type(reference_audio)}")

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
          - audio + .lab
          - or .codes.pt + .lab
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
        self, id: str, codes_bytes: bytes, lab_text: str, stem: str | None = None
    ) -> Literal["created", "updated", "unchanged"]:
        """
        Add or update a pre-encoded reference (e.g. from preencode_references.py).
        Writes references/<id>/<stem>.codes.pt and <stem>.lab (stem defaults to id). Skips if hash matches.
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

        existed_before = ref_dir.exists() and (ref_dir / f"{stem}.codes.pt").exists()
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / f"{stem}.codes.pt").write_bytes(codes_bytes)
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
        # Validate ID format
        import re

        if not re.match(r"^[a-zA-Z0-9\-_ ]+$", id):
            raise ValueError(
                "Reference ID contains invalid characters. Only alphanumeric, hyphens, underscores, and spaces are allowed."
            )

        if len(id) > 255:
            raise ValueError(
                "Reference ID is too long. Maximum length is 255 characters."
            )

        # Check if reference already exists
        ref_dir = self.references_dir / id
        if ref_dir.exists():
            raise FileExistsError(f"Reference ID '{id}' already exists")

        # Check if audio file exists
        audio_path = Path(wav_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_file_path}")

        # Validate audio file extension
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. Supported formats: {', '.join(AUDIO_EXTENSIONS)}"
            )

        try:
            # Create reference directory
            ref_dir.mkdir(parents=True, exist_ok=False)

            # Determine the target audio filename with original extension
            target_audio_path = ref_dir / f"sample{audio_path.suffix}"

            # Copy audio file
            import shutil

            shutil.copy2(audio_path, target_audio_path)

            # Create .lab file
            lab_path = ref_dir / "sample.lab"
            with open(lab_path, "w", encoding="utf-8") as f:
                f.write(reference_text)

            # Clear cache for this ID if it exists
            if id in self.ref_by_id:
                del self.ref_by_id[id]

            logger.info(f"Successfully added reference voice with ID: {id}")

        except Exception as e:
            # Clean up on failure
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
        # Check if reference exists
        ref_dir = self.references_dir / id
        if not ref_dir.exists():
            raise FileNotFoundError(f"Reference ID '{id}' does not exist")

        try:
            # Remove the entire reference directory
            import shutil

            shutil.rmtree(ref_dir)

            # Clear cache for this ID if it exists
            if id in self.ref_by_id:
                del self.ref_by_id[id]

            logger.info(f"Successfully deleted reference voice with ID: {id}")

        except Exception as e:
            logger.error(f"Failed to delete reference '{id}': {e}")
            raise OSError(f"Failed to delete reference '{id}': {e}")

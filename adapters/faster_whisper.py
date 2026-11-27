"""PhoWhisper adapter via CTranslate2 (faster-whisper), fp32.

Backend specifics kept here:

1. faster-whisper does not read Hugging Face checkpoints: the checkpoint is converted
   once with `quantization=float32` (`scripts/download_models.py`) and run with
   `compute_type: float32`.
2. Whisper only has a prompt mechanism: hotwords go to faster-whisper's `hotwords`
   argument, or, when `hotword_prompt` is set, the formatted template goes to
   `initial_prompt`. There is no boost weight.
3. The temperature fallback ladder is disabled: one deterministic decode per utterance.
"""

from __future__ import annotations

from collections.abc import Sequence

from adapters.base import Segment, check_boost_level, read_waveform, weight_files_mb


class FasterWhisperSession:
    def __init__(self, adapter: "FasterWhisperAdapter", hotwords: list[str]) -> None:
        self._adapter = adapter
        words = " ".join(hotwords)
        template = adapter.config.get("hotword_prompt")
        self._prompt = template.format(words=words) if template and words else None
        self._hotwords = None if self._prompt else (words or None)

    def transcribe(self, segment: Segment) -> str:
        # Pass the waveform, not the path: with a path faster-whisper decodes through
        # PyAV (`av.open(..., metadata_errors=...)`), which breaks on newer PyAV releases.
        samples, _ = read_waveform(segment.audio)
        segments, _ = self._adapter.model.transcribe(
            samples,
            language=str(self._adapter.config.get("language", "vi")),
            beam_size=int(self._adapter.config.get("beam_size", 5)),
            hotwords=self._hotwords,
            initial_prompt=self._prompt,
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=None,
            log_prob_threshold=None,
            no_speech_threshold=None,
        )
        return " ".join(part.text.strip() for part in segments).strip()  # consumes the generator

    def close(self) -> None:
        pass


class FasterWhisperAdapter:
    needs_ground_truth = False

    def load(self, config: dict) -> None:
        import ctranslate2
        from faster_whisper import WhisperModel

        self.config = dict(config)
        device = "cuda" if config.get("provider") == "cuda" else "cpu"
        compute_type = str(config["compute_type"])
        # CTranslate2 silently falls back to CPU when CUDA libraries are missing
        if device == "cuda" and compute_type not in ctranslate2.get_supported_compute_types("cuda"):
            raise RuntimeError(f"faster-whisper: CTranslate2 cannot run {compute_type} on CUDA")
        self.model = WhisperModel(config["path"], device=device, compute_type=compute_type, local_files_only=True)

    def weights_mb(self) -> float:
        return weight_files_mb(self.config["path"])

    def open_session(self, hotwords: Sequence[str], boost_level: str) -> FasterWhisperSession:
        on = check_boost_level(boost_level) == "on"
        return FasterWhisperSession(self, list(hotwords) if on else [])

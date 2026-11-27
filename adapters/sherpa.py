"""sherpa-onnx transducer adapters: offline (zipformer-30m) and streaming (zipformer-291m).

Backend specifics kept here, out of the harness:

1. API: `OfflineRecognizer` decodes the whole padded waveform once; `OnlineRecognizer`
   gets the whole padded waveform, `input_finished()`, then is drained. Endpointing
   is off: each file is one sentence.
2. Hotwords need `hotwords_file` when the recognizer is built, so the recognizer is
   (re)built in `open_session()`. They are encoded with the model's BPE vocabulary
   (`modeling_unit="bpe"`, `bpe_vocab=tokens.txt`); a hotword whose casing does not
   exist in that vocabulary is silently ignored, hence `hotword_case: upper` for both
   models (zipformer-30m is all uppercase, PengChengStarling writes Latin in uppercase).
3. Hotwords are only used by `modified_beam_search` (enforced by the registry). `off`
   uses the same decoding method without hotwords.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from adapters.base import Segment, check_boost_level, pad_tail, read_waveform, weight_files_mb

MODEL_FILES = ("encoder", "decoder", "joiner", "tokens")


class SherpaSession:
    def __init__(self, recognizer, streaming: bool, hotwords_path: Path | None) -> None:
        self._recognizer = recognizer
        self._streaming = streaming
        self._hotwords_path = hotwords_path

    def transcribe(self, segment: Segment) -> str:
        samples, sample_rate = read_waveform(segment.audio)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, pad_tail(samples, sample_rate))
        if not self._streaming:
            self._recognizer.decode_stream(stream)
            return stream.result.text
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        return self._recognizer.get_result(stream).strip()

    def close(self) -> None:
        self._recognizer = None
        if self._hotwords_path is not None:
            self._hotwords_path.unlink(missing_ok=True)


class SherpaOfflineAdapter:
    needs_ground_truth = False
    streaming = False

    def load(self, config: dict) -> None:
        self._config = dict(config)
        for key in MODEL_FILES:
            if not config.get(key) or not Path(config[key]).is_file():
                raise FileNotFoundError(f"{key}: {config.get(key)} (run scripts/download_models.py)")

    def weights_mb(self) -> float:
        return weight_files_mb(*(self._config.get(key) for key in ("encoder", "decoder", "joiner")))

    def open_session(self, hotwords: Sequence[str], boost_level: str) -> SherpaSession:
        import sherpa_onnx

        unique = [item for item in dict.fromkeys(hotwords) if item.strip()]
        hotwords_path, kwargs = None, {}
        if check_boost_level(boost_level) == "on" and unique:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
                handle.write("\n".join(unique) + "\n")
            hotwords_path = Path(handle.name)
            kwargs = {
                "hotwords_file": str(hotwords_path),
                "hotwords_score": float(self._config["hotword_score"]),
                "modeling_unit": "bpe",
                "bpe_vocab": self._config["tokens"],
            }
        common = {
            "encoder": self._config["encoder"],
            "decoder": self._config["decoder"],
            "joiner": self._config["joiner"],
            "tokens": self._config["tokens"],
            "num_threads": int(self._config.get("num_threads", 4)),
            "sample_rate": 16000,
            "feature_dim": 80,
            "decoding_method": str(self._config["decoding_method"]),
            "max_active_paths": int(self._config["max_active_paths"]),
            "provider": self._config.get("provider", "cpu"),
            **kwargs,
        }
        if self.streaming:
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(enable_endpoint_detection=False, **common)
        else:
            recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(model_type="transducer", **common)
        return SherpaSession(recognizer, self.streaming, hotwords_path)


class SherpaOnlineAdapter(SherpaOfflineAdapter):
    streaming = True

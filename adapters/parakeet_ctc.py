"""NVIDIA parakeet-ctc-0.6b-Vietnamese adapter (NeMo), fp32.

Backend specifics kept here:

1. fp32 end to end (docs/OVERVIEW.md); the 2.4 GB weights fit a T4.
2. Hotwords go through NeMo's `pyctcdecode` decoding strategy
   (`pyctcdecode_cfg.hotwords` / `hotword_weight`). `off` uses the same strategy with
   no hotwords, so the conditions differ by the hotwords alone. The tokenizer emits
   case; acronyms are uppercase, hence `hotword_case: upper`.
3. NeMo's `transcribe()` reads audio itself; `batch_size=1` keeps one call per utterance.
"""

from __future__ import annotations

from collections.abc import Sequence

from adapters.base import Segment, check_boost_level, tensor_bytes_mb


class ParakeetSession:
    def __init__(self, model) -> None:
        self._model = model

    def transcribe(self, segment: Segment) -> str:
        output = self._model.transcribe([str(segment.audio)], batch_size=1, verbose=False)
        first = output[0] if output else ""
        return str(getattr(first, "text", first)).replace("<|endoftext|>", "").strip()

    def close(self) -> None:
        pass


class ParakeetAdapter:
    needs_ground_truth = False

    def load(self, config: dict) -> None:
        from nemo.collections.asr.models import ASRModel

        self._config = dict(config)
        device = "cuda" if config.get("provider") == "cuda" else "cpu"
        model = ASRModel.restore_from(config["path"], map_location=device).float().to(device)
        if next(model.parameters()).device.type != device:
            raise RuntimeError(f"parakeet: model did not move to {device}")
        self._model = model.eval()

    def weights_mb(self) -> float:
        return tensor_bytes_mb(self._model)

    def open_session(self, hotwords: Sequence[str], boost_level: str) -> ParakeetSession:
        from omegaconf import open_dict

        on = check_boost_level(boost_level) == "on"
        unique = [item for item in dict.fromkeys(hotwords) if item.strip()] if on else []
        cfg = self._model.cfg.decoding
        with open_dict(cfg):
            # `strategy: beam` hard-codes KenLM; the `pyctcdecode` strategy accepts hotwords
            cfg.strategy = "pyctcdecode"
            cfg.beam.beam_size = int(self._config.get("beam_size", 8))
            cfg.beam.pyctcdecode_cfg.hotwords = unique or None
            cfg.beam.pyctcdecode_cfg.hotword_weight = float(self._config.get("hotword_score") or 0.0)
        self._model.change_decoding_strategy(cfg)
        return ParakeetSession(self._model)

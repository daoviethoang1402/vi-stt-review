"""ChunkFormer CTC adapter (khanhld/chunkformer-ctc-large-vie), PyTorch fp32.

Backend specifics kept here:

1. `ChunkFormerModel.endless_decode()` takes a file path, chunks the audio itself and
   exposes no logits, so the CTC head's `log_softmax` is wrapped to capture them.
2. The model has no hotword API: hotwords go through pyctcdecode's beam search on
   those logits (`hotwords=`, character-trie boost, no LM). `off` uses the same beam
   search without hotwords, so the two conditions differ by the hotwords alone.
3. The model's `char_dict` (lowercase pieces) is the pyctcdecode label set, hence
   `hotword_case: lower`.
"""

from __future__ import annotations

from collections.abc import Sequence

from adapters.base import Segment, check_boost_level, tensor_bytes_mb

CHUNKING_DEFAULTS = {"chunk_size": 64, "left_context_size": 128, "right_context_size": 128}


def build_labels(char_dict: dict) -> list[str]:
    """`{index: piece}` → pyctcdecode labels: index 0 blank (""), labels unique.

    Special tokens (`<unk>`, ...) get a unique private-use placeholder so indices
    still line up with the logits.
    """
    labels = [f"{index}" for index in range(max(char_dict) + 1)]
    for index, piece in char_dict.items():
        if index == 0:
            labels[index] = ""
        elif not str(piece).startswith("<"):
            labels[index] = str(piece)
    return labels


class ChunkFormerSession:
    def __init__(self, adapter: "ChunkFormerAdapter", hotwords: list[str]) -> None:
        self._adapter = adapter
        self._hotwords = hotwords

    def transcribe(self, segment: Segment) -> str:
        model = self._adapter.model
        captured: dict = {}
        ctc = model.model.ctc
        original = ctc.log_softmax

        def capture(*args, **kwargs):
            captured["log_probs"] = original(*args, **kwargs)
            return captured["log_probs"]

        ctc.log_softmax = capture
        try:
            model.endless_decode(audio_path=str(segment.audio), total_batch_duration=600,
                                 return_timestamps=False, **self._adapter.chunking)
        finally:
            ctc.log_softmax = original
        if "log_probs" not in captured:
            raise RuntimeError("chunkformer: CTC log-probs were not captured")
        logits = captured["log_probs"].squeeze(0).float().cpu().numpy()
        kwargs = {"hotwords": self._hotwords, "hotword_weight": self._adapter.weight} if self._hotwords else {}
        return self._adapter.decoder.decode(logits, **kwargs)

    def close(self) -> None:
        pass


class ChunkFormerAdapter:
    needs_ground_truth = False

    def load(self, config: dict) -> None:
        from chunkformer import ChunkFormerModel
        from pyctcdecode import build_ctcdecoder

        device = "cuda" if config.get("provider") == "cuda" else "cpu"
        self.model = ChunkFormerModel.from_pretrained(config["path"]).to(device).eval()
        if next(self.model.parameters()).device.type != device:
            raise RuntimeError(f"chunkformer: model did not move to {device}")
        self.chunking = {key: int(config.get(key, value)) for key, value in CHUNKING_DEFAULTS.items()}
        self.weight = float(config.get("hotword_score") or 0.0)
        self.decoder = build_ctcdecoder(build_labels(self.model.char_dict))

    def weights_mb(self) -> float:
        return tensor_bytes_mb(self.model)

    def open_session(self, hotwords: Sequence[str], boost_level: str) -> ChunkFormerSession:
        on = check_boost_level(boost_level) == "on"
        return ChunkFormerSession(self, [item for item in dict.fromkeys(hotwords) if item.strip()] if on else [])

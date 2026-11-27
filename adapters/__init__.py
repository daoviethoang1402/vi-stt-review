"""Adapter registry — the only place that knows which adapters exist.

Adding a runtime means adding one file and one entry in `FACTORIES` and `CONTRACTS`;
neither the harness nor the scoring changes.
"""

from __future__ import annotations

from collections.abc import Callable

from adapters.base import Adapter
from adapters.chunkformer import ChunkFormerAdapter
from adapters.faster_whisper import FasterWhisperAdapter
from adapters.parakeet_ctc import ParakeetAdapter
from adapters.qwen3_asr import Qwen3AsrAdapter
from adapters.qwen3_asr_api import Qwen3AsrApiAdapter
from adapters.sherpa import SherpaOfflineAdapter, SherpaOnlineAdapter

FACTORIES: dict[str, Callable[[], Adapter]] = {
    "sherpa-offline": SherpaOfflineAdapter,
    "sherpa-online": SherpaOnlineAdapter,
    "chunkformer": ChunkFormerAdapter,
    "parakeet-ctc": ParakeetAdapter,
    "faster-whisper": FasterWhisperAdapter,
    "qwen3-asr": Qwen3AsrAdapter,
    "qwen3-asr-api": Qwen3AsrApiAdapter,
}

#: `config` keys every adapter may declare.
COMMON_KEYS = frozenset({
    "adapter", "dtype", "note", "provider",
    "hotword_enabled", "hotword_case", "hotword_score", "hotword_score_param", "hotword_prompt",
})

_SHERPA = {
    "allowed": frozenset({"encoder", "decoder", "joiner", "tokens", "decoding_method", "max_active_paths"}),
    "required": frozenset({"encoder", "decoder", "joiner", "tokens", "decoding_method", "max_active_paths"}),
    "hotword_param": "hotwords_score",
    "needs_prompt": False,
    # sherpa-onnx only uses hotwords with this method; others accept and ignore them
    "requires_with_hotword": {"decoding_method": "modified_beam_search"},
}

#: Per-adapter `config` contract, checked by `harness/registry.py` at load time.
#: `hotword_param` is the runtime's name for the boost weight (`None` = no weight);
#: `needs_prompt` means hotwords only reach the model through `hotword_prompt`;
#: `dtype_flag` = (boolean key, {value: dtype}): the key is required and `dtype` must
#: match it, so the reported precision is always the one actually run.
CONTRACTS: dict[str, dict] = {
    "sherpa-offline": _SHERPA,
    "sherpa-online": _SHERPA,
    "chunkformer": {
        "allowed": frozenset({"path", "chunk_size", "left_context_size", "right_context_size"}),
        "required": frozenset({"path"}),
        "hotword_param": "hotword_weight",
        "needs_prompt": False,
    },
    "parakeet-ctc": {
        "allowed": frozenset({"path", "beam_size"}),
        "required": frozenset({"path"}),
        "hotword_param": "hotword_weight",
        "needs_prompt": False,
    },
    "faster-whisper": {
        "allowed": frozenset({"path", "compute_type", "beam_size", "language"}),
        "required": frozenset({"path", "compute_type"}),
        "hotword_param": None,
        "needs_prompt": False,
    },
    "qwen3-asr": {
        "allowed": frozenset({"path", "max_new_tokens", "language", "convert_to_fp32"}),
        "required": frozenset({"path", "max_new_tokens"}),
        "hotword_param": None,
        "needs_prompt": True,
        "dtype_flag": ("convert_to_fp32", {True: "fp32", False: "bf16"}),
    },
    "qwen3-asr-api": {
        "allowed": frozenset({"url", "file_field", "context_field", "response_field", "timeout_s", "ping_attempts"}),
        "required": frozenset({"url"}),
        "hotword_param": None,
        "needs_prompt": True,
    },
}


def create(name: str) -> Adapter:
    try:
        return FACTORIES[name]()
    except KeyError:
        raise KeyError(f"unknown adapter: {name!r}; known {sorted(FACTORIES)}") from None

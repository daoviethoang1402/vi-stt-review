"""Qwen3-ASR-1.7B adapter over the company HTTP API (model served in bf16).

Backend specifics kept here:

1. **Reachability**: `load()` sends up to `ping_attempts` real requests (one second of
   silence). If none succeeds it raises `ModelUnavailable` and the harness skips the
   model (e.g. on Kaggle). The connect timeout is short so an unreachable host fails
   fast; a slow transcription still gets `timeout_s`.
2. **Request contract** comes from the registry: multipart field for the audio
   (`file_field`), form field for the hotword context (`context_field`) and the JSON
   key of the transcript (`response_field`, dotted path allowed).
3. **Hotwords**: canonical hotwords are formatted into `hotword_prompt` and sent as
   context on `on` only.
4. **Resources**: inference is remote, so `weights_mb()` is `None` and VRAM is not
   measured; latency includes the network round-trip.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence

import numpy as np

from adapters.base import ModelUnavailable, Segment, check_boost_level

CONNECT_TIMEOUT_S = 5.0


def extract_text(payload, response_field: str) -> str:
    """Transcript from a JSON payload by dotted path; a plain string passes through."""
    value = payload
    for key in ([] if isinstance(payload, str) else response_field.split(".")):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"response has no field {response_field!r}: {str(payload)[:200]}")
        value = value[key]
    if not isinstance(value, str):
        raise TypeError(f"field {response_field!r} is not a string: {value!r}")
    return value.strip()


class Qwen3AsrApiSession:
    def __init__(self, adapter: "Qwen3AsrApiAdapter", context: str | None) -> None:
        self._adapter = adapter
        self._context = context

    def transcribe(self, segment: Segment) -> str:
        return self._adapter.post(segment.audio.read_bytes(), segment.audio.name, self._context)

    def close(self) -> None:
        pass


class Qwen3AsrApiAdapter:
    needs_ground_truth = False

    def load(self, config: dict) -> None:
        if not config.get("url"):
            raise ValueError("qwen3-asr-api needs `url`")
        self.config = dict(config)
        self.ping(int(config.get("ping_attempts") or 2))

    def post(self, audio: bytes, filename: str, context: str | None) -> str:
        import requests

        cfg = self.config
        context_field = cfg.get("context_field", "context")
        response = requests.post(
            cfg["url"],
            files={cfg.get("file_field") or "file": (filename, audio, "audio/wav")},
            data={context_field: context} if context and context_field else {},
            timeout=(CONNECT_TIMEOUT_S, float(cfg.get("timeout_s") or 60)),
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = response.text
        return extract_text(payload, cfg.get("response_field") or "text")

    def ping(self, attempts: int) -> None:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(16000, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
        error = None
        for _ in range(max(1, attempts)):
            try:
                self.post(buffer.getvalue(), "ping.wav", None)
                return
            except Exception as exc:  # noqa: BLE001 - any failure means "not usable here"
                error = f"{type(exc).__name__}: {exc}"
        raise ModelUnavailable(f"API {self.config['url']} unreachable after {attempts} attempt(s): {error}")

    def weights_mb(self) -> None:
        return None

    def open_session(self, hotwords: Sequence[str], boost_level: str) -> Qwen3AsrApiSession:
        template = self.config.get("hotword_prompt")
        on = check_boost_level(boost_level) == "on"
        context = template.format(words=", ".join(hotwords)) if on and hotwords and template else None
        return Qwen3AsrApiSession(self, context)

"""Adapter interface — docs/PIPELINE.md, *Interface adapter & Registry*.

The harness calls only the operations below. There is no `if model == ...` on the
harness side: every backend difference lives in that runtime's adapter file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

BOOST_OFF = "off"
BOOST_ON = "on"
BOOST_LEVELS = (BOOST_OFF, BOOST_ON)

#: Trailing silence the sherpa (transducer) adapters append: transducers need audio after
#: the speech to emit the last tokens. Runtimes that read the file themselves get no padding.
TAIL_PADDING_MS = 500


class ModelUnavailable(RuntimeError):
    """Raised from `load()` when the model cannot be used here (e.g. API unreachable).

    The harness skips the model and records the reason instead of aborting the run.
    """


@dataclass(frozen=True)
class Segment:
    """One 16 kHz mono utterance."""

    id: str
    audio: Path


class Session(Protocol):
    def transcribe(self, segment: Segment) -> str:
        """Verbatim model output; adapters never canonicalize it."""

    def close(self) -> None: ...


class Adapter(Protocol):
    #: only test control adapters set `True`; the harness gives labels to those alone
    needs_ground_truth: bool

    def load(self, config: dict) -> None: ...

    def open_session(self, hotwords: Sequence[str], boost_level: str) -> Session: ...

    def weights_mb(self) -> float | None: ...


def check_boost_level(boost_level: str) -> str:
    if boost_level not in BOOST_LEVELS:
        raise ValueError(f"boost_level {boost_level!r} not in {BOOST_LEVELS}")
    return boost_level


def read_waveform(path: Path, sample_rate: int = 16000):
    """Mono float32 waveform; rejects other sample rates."""
    import soundfile as sf

    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != sample_rate:
        raise ValueError(f"{Path(path).name}: {rate} Hz, expected {sample_rate}")
    return samples, rate


def pad_tail(samples, sample_rate: int):
    """Append `TAIL_PADDING_MS` of silence."""
    count = int(sample_rate * TAIL_PADDING_MS / 1000)
    return np.concatenate([samples, np.zeros(count, dtype=samples.dtype)])


def tensor_bytes_mb(module) -> float:
    """Bytes of parameters + buffers held by a PyTorch module, in MB."""
    tensors = {id(t): t for t in (*module.parameters(), *module.buffers())}.values()
    return sum(t.numel() * t.element_size() for t in tensors) / (1024 * 1024)


def weight_files_mb(*paths) -> float:
    """Size of weight files in MB (ONNX Runtime and CTranslate2 load whole files).

    A directory counts its `model.bin`.
    """
    total = 0
    for raw in filter(None, paths):
        path = Path(raw)
        if path.is_dir():
            path = path / "model.bin"
        if path.is_file():
            total += path.stat().st_size
    return total / (1024 * 1024)

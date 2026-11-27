"""Resource probes read from the OS, and the machine fingerprint."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _proc_status(field: str) -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(f"{field}:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def peak_rss_mb() -> float | None:
    """Peak host RAM of this process (VmHWM), in MB.

    VmHWM only grows, so it is the peak over the whole process lifetime — which is why
    every model runs in its own process (`harness/cli.py`).
    """
    return _proc_status("VmHWM") or _proc_status("VmRSS")


def total_ram_mb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _nvidia_smi(*query: str) -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_name() -> str | None:
    lines = _nvidia_smi("--query-gpu=name")
    return lines[0] if lines else None


def device_used_mb() -> float | None:
    lines = _nvidia_smi("--query-gpu=memory.used")
    try:
        return float(lines[0]) if lines else None
    except ValueError:
        return None


def process_used_mb(pid: int) -> float | None:
    """VRAM used by `pid` according to the driver; `None` when not listed.

    Inside containers (e.g. Kaggle) the driver often cannot map PIDs, so the process
    is missing from the list; the sampler then falls back to the device delta.
    """
    for line in _nvidia_smi("--query-compute-apps=pid,used_memory"):
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


class VramSampler:
    """Poll the driver for this process's peak VRAM while a model loads and runs.

    Preferred measure: per-process `used_memory` (covers torch, ONNX Runtime and
    CTranslate2 alike). Fallback when the process is not visible: peak device
    `memory.used` minus the baseline taken before the model was loaded — exact as
    long as nothing else uses the GPU meanwhile (true for one model per process).
    """

    POLL_S = 0.1

    def __init__(self, enabled: bool, interval: float = POLL_S) -> None:
        self.enabled = enabled
        self.interval = interval
        self.baseline_mb: float | None = None
        self._process_peak: float | None = None
        self._device_peak: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        own = process_used_mb(os.getpid())
        if own is not None:
            self._process_peak = max(self._process_peak or 0.0, own)
        device = device_used_mb()
        if device is not None:
            self._device_peak = max(self._device_peak or 0.0, device)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> "VramSampler":
        if not self.enabled:
            return self
        self.baseline_mb = device_used_mb()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5.0)
            self._sample()
            self._thread = None

    @property
    def peak_mb(self) -> float | None:
        if self._process_peak is not None:
            return self._process_peak
        if self._device_peak is not None and self.baseline_mb is not None:
            return max(0.0, self._device_peak - self.baseline_mb)
        return None

    @property
    def method(self) -> str | None:
        if self._process_peak is not None:
            return "per_process"
        if self._device_peak is not None and self.baseline_mb is not None:
            return "device_delta"
        return None


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
            timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "no-git"
    return result.stdout.strip()


def fingerprint(num_threads: int, extra: dict | None = None) -> dict:
    """Identity of a run: which machine, when, which versions."""
    data = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "gpu": gpu_name(),
        "cpu_count": os.cpu_count(),
        "num_threads": num_threads,
        "total_ram_mb": round(total_ram_mb(), 1),
        "python": sys.version.split()[0],
        "git_sha": git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        data.update(extra)
    return data

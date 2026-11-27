"""Set `LD_LIBRARY_PATH` for the process **before** it starts.

glibc reads `LD_LIBRARY_PATH` once at process start, so editing `os.environ` later
does not affect `dlopen()`. If the variable does not point at the CUDA libraries of
the environment yet, the process re-executes itself.

Without it sherpa-onnx prints `Failed to enable CUDA ... Fallback to cpu` and
CTranslate2 reports CPU compute types — neither raises, so the pipeline would silently
measure CPU numbers while believing it runs on the GPU.
"""

from __future__ import annotations

import importlib.util
import os
import site
import sys
from pathlib import Path


def _nvidia_roots() -> list[Path]:
    """Every `nvidia/` package directory Python can see.

    The pip wheels `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, ... (installed with torch)
    live under `<site dir>/nvidia/<name>/lib`. The site dir differs between layouts:
    conda/venv use `<prefix>/lib/pythonX.Y/site-packages`, Debian/Ubuntu images such as
    Kaggle use `/usr/local/lib/pythonX.Y/dist-packages`. Ask Python instead of guessing.
    """
    roots: list[Path] = []
    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations:
        roots += [Path(location) for location in spec.submodule_search_locations]
    candidates = [*site.getsitepackages(), site.getusersitepackages(), *sys.path]
    roots += [Path(entry) / "nvidia" for entry in candidates if entry]
    unique: list[Path] = []
    for root in roots:
        if root.is_dir() and root.resolve() not in {item.resolve() for item in unique}:
            unique.append(root)
    return unique


def cuda_library_path() -> str:
    """`nvidia/*/lib` of the pip CUDA wheels first, then a system CUDA toolkit if any."""
    dirs: list[str] = []
    for root in _nvidia_roots():
        dirs += sorted(str(path) for path in root.glob("*/lib") if path.is_dir())
    for extra in ("/usr/local/cuda/lib64",):
        if Path(extra).is_dir():
            dirs.append(extra)
    return os.pathsep.join(dict.fromkeys(dirs))


def ensure_process_env(prefix, marker: str = "CUDA_ENV_REEXEC") -> None:
    """Re-exec the program with the right `LD_LIBRARY_PATH`, at most once.

    `prefix` is what follows `python`, e.g. `["run.py"]` for the root script or
    `["-m", "harness.cli"]` for the module.
    """
    if os.environ.get(marker):
        return
    cuda = cuda_library_path()
    if not cuda:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(p for p in (cuda, current) if p)}
    # `expandable_segments` lets the PyTorch allocator grow non-contiguously, which
    # avoids fragmentation OOMs when a session is rebuilt between `off` and `on`.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env[marker] = "1"
    os.execve(sys.executable, [sys.executable, *prefix, *sys.argv[1:]], env)

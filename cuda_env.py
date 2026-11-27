"""Đặt `LD_LIBRARY_PATH` cho tiến trình **trước khi** nó khởi động.

glibc đọc `LD_LIBRARY_PATH` một lần lúc process khởi động, nên sửa `os.environ`
sau đó **không** ảnh hưởng tới `dlopen()`. Vì vậy nếu biến chưa trỏ tới các thư
viện CUDA của env thì phải exec lại chính mình.

Thiếu biến này thì sherpa-onnx in `Failed to enable CUDA ... Fallback to cpu` và
CTranslate2 trả về compute type CPU — cả hai đều **không báo lỗi**, nên pipeline sẽ
lặng lẽ cho số đo CPU trong khi tưởng là GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def cuda_library_path(prefix: str | None = None) -> str:
    prefix = prefix or sys.prefix
    site = (
        Path(prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    dirs = sorted(str(path) for path in (site / "nvidia").glob("*/lib"))
    for extra in ("/usr/local/cuda-12.5/lib64", "/usr/local/cuda/lib64"):
        if Path(extra).is_dir():
            dirs.append(extra)
    return os.pathsep.join(dirs)


def ensure_process_env(prefix, marker: str = "CUDA_ENV_REEXEC") -> None:
    """Exec lại chính chương trình với `LD_LIBRARY_PATH` đúng, tối đa một lần.

    `prefix` là phần đứng sau `python`, ví dụ `["run.py"]` khi chạy script ở gốc,
    hoặc `["-m", "harness.cli"]` khi chạy bằng module.
    """
    cuda = cuda_library_path()
    if not cuda:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    first = cuda.split(os.pathsep)[0]
    if (first in current and os.environ.get("PYTORCH_CUDA_ALLOC_CONF")) or os.environ.get(
        marker
    ):
        return
    env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(p for p in (cuda, current) if p)}
    # `expandable_segments` lets the PyTorch allocator grow non-contiguously, which
    # avoids fragmentation OOMs when a session is rebuilt between `off` and `on`.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env[marker] = "1"
    os.execve(sys.executable, [sys.executable, *prefix, *sys.argv[1:]], env)

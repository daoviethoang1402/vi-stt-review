"""Single entry point of the survey.

    python run.py                    # run every model of registry.yaml
    python run.py --model a b        # run only these models
    python run.py --list             # list the models
    python run.py --help             # every flag

Model parameters live in `registry.yaml`, never on the command line.
"""

from __future__ import annotations

import sys

from cuda_env import ensure_process_env
from harness.cli import main

if __name__ == "__main__":
    # Set `LD_LIBRARY_PATH` and re-exec this script before any CUDA library is loaded;
    # without it sherpa-onnx / CTranslate2 silently fall back to CPU.
    ensure_process_env([sys.argv[0]])
    sys.exit(main())

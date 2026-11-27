"""Download every local model into the paths `registry.yaml` expects.

    python scripts/download_models.py                  # all models
    python scripts/download_models.py --model zipformer-30m phowhisper-medium
    python scripts/download_models.py --list
    python scripts/download_models.py --output /kaggle/working/models

Files come from the Hugging Face repos listed in docs/OVERVIEW.md, pinned to a commit
so that a rebuild fetches exactly the same weights. Existing files with the expected
size are skipped, so the script can be re-run after an interruption.

`qwen3-asr-1.7b` is served by the company API and has nothing to download.
PhoWhisper is converted once to CTranslate2 float32 (faster-whisper cannot read the
Hugging Face checkpoint directly).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: registry model name → download recipe. `files` maps repo file → local file name.
MODELS: dict[str, dict] = {
    "zipformer-30m": {
        "repo": "hynt/Zipformer-30M-RNNT-6000h",
        "revision": "ad6f873b5f5b6ed083821253a88d81afed2505c4",
        "dir": "zipformer-30m",
        "files": {
            "encoder-epoch-20-avg-10.onnx": "encoder-epoch-20-avg-10.onnx",
            "decoder-epoch-20-avg-10.onnx": "decoder-epoch-20-avg-10.onnx",
            "joiner-epoch-20-avg-10.onnx": "joiner-epoch-20-avg-10.onnx",
            # this repo ships the sherpa-onnx token table under the name `config.json`
            "config.json": "tokens.txt",
            "bpe.model": "bpe.model",
        },
    },
    "zipformer-291m": {
        "repo": "stdo/PengChengStarling",
        "revision": "3afca1d0cdbda6b0cde440e31539be81df5d1301",
        "dir": "zipformer-291m",
        "files": {
            "encoder-epoch-75-avg-11-chunk-16-left-128.onnx": "encoder-epoch-75-avg-11-chunk-16-left-128.onnx",
            "decoder-epoch-75-avg-11-chunk-16-left-128.onnx": "decoder-epoch-75-avg-11-chunk-16-left-128.onnx",
            "joiner-epoch-75-avg-11-chunk-16-left-128.onnx": "joiner-epoch-75-avg-11-chunk-16-left-128.onnx",
            "tokens.txt": "tokens.txt",
            "bpe.model": "bpe.model",
        },
    },
    "chunkformer-ctc-large-vie": {
        "repo": "khanhld/chunkformer-ctc-large-vie",
        "revision": "311fc03558a895dc2b32957f2fb4236c7fb1455b",
        "dir": "chunkformer-ctc-large-vie",
        "files": {name: name for name in ("config.yaml", "global_cmvn", "pytorch_model.bin", "vocab.txt")},
    },
    "parakeet-ctc-0.6b-vi": {
        "repo": "nvidia/parakeet-ctc-0.6b-Vietnamese",
        "revision": "b0493142b49458810324e3db8be9e8e07b4ebc17",
        "dir": "parakeet-ctc-0.6b-vi",
        "files": {"parakeet-ctc-0.6b-vi.nemo": "parakeet-ctc-0.6b-vi.nemo"},
    },
    "phowhisper-medium": {
        "repo": "vinai/PhoWhisper-medium",
        "revision": "55a7e3eb6c906de891f8f06a107754427dd3be79",
        "dir": "phowhisper-medium",
        "files": {
            name: name
            for name in (
                "added_tokens.json", "config.json", "generation_config.json", "merges.txt",
                "normalizer.json", "preprocessor_config.json", "pytorch_model.bin",
                "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
            )
        },
        "ct2": {"dir": "phowhisper-medium-ct2-fp32", "quantization": "float32"},
    },
}

MODELS["qwen3-asr-0.6b"] = {
    "repo": "Qwen/Qwen3-ASR-0.6B-hf",  # Transformers-native copy of Qwen/Qwen3-ASR-0.6B
    "revision": "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
    "dir": "qwen3-asr-0.6b",
    "files": {
        name: name
        for name in (
            "chat_template.jinja", "config.json", "generation_config.json", "model.safetensors",
            "processor_config.json", "tokenizer.json", "tokenizer_config.json",
        )
    },
}

REMOTE = {"qwen3-asr-1.7b": "served by the company API (http://192.168.0.47:3979), nothing to download"}


def _retry(call, attempts: int = 4):
    """Retry transient Hugging Face errors (504s happen) with a growing pause."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                raise
            print(f"  retry   {type(exc).__name__} (attempt {attempt}/{attempts})", flush=True)
            time.sleep(5 * attempt)


def _remote_sizes(repo: str, revision: str) -> dict[str, int]:
    from huggingface_hub import HfApi

    info = _retry(lambda: HfApi().model_info(repo, revision=revision, files_metadata=True))
    return {item.rfilename: item.size for item in info.siblings if item.size is not None}


def download(name: str, output: Path) -> None:
    from huggingface_hub import hf_hub_download

    recipe = MODELS[name]
    target = output / recipe["dir"]
    target.mkdir(parents=True, exist_ok=True)
    sizes = _remote_sizes(recipe["repo"], recipe["revision"])
    for remote, local in recipe["files"].items():
        destination = target / local
        expected = sizes.get(remote)
        if destination.is_file() and expected is not None and destination.stat().st_size == expected:
            print(f"  ok      {destination.relative_to(output)}")
            continue
        print(f"  fetch   {recipe['repo']}/{remote} -> {destination.relative_to(output)}", flush=True)
        # download straight into the target folder: no second copy in ~/.cache
        fetched = Path(_retry(lambda: hf_hub_download(recipe["repo"], remote,
                                                      revision=recipe["revision"], local_dir=target)))
        if fetched.resolve() != destination.resolve():
            shutil.move(str(fetched), destination)
    if "ct2" in recipe:
        convert_ct2(target, output / recipe["ct2"]["dir"], recipe["ct2"]["quantization"])


def convert_ct2(source: Path, target: Path, quantization: str) -> None:
    if (target / "model.bin").is_file():
        print(f"  ok      {target.name}/model.bin (already converted)")
        return
    from ctranslate2.converters import TransformersConverter

    print(f"  convert {source.name} -> {target.name} ({quantization})", flush=True)
    converter = TransformersConverter(
        str(source), copy_files=["tokenizer.json", "preprocessor_config.json"]
    )
    converter.convert(str(target), quantization=quantization, force=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the survey models into models/.")
    parser.add_argument("--model", nargs="*", metavar="NAME",
                        help="download only these models (see --list)")
    parser.add_argument("--list", action="store_true", help="list downloadable models and exit")
    parser.add_argument("--output", type=Path, default=ROOT / "models",
                        help="destination folder (default: models/)")
    args = parser.parse_args(argv)

    if args.list:
        for name, recipe in MODELS.items():
            print(f"{name:28}{recipe['repo']}@{recipe['revision'][:8]}")
        for name, reason in REMOTE.items():
            print(f"{name:28}{reason}")
        return 0
    if args.model is not None and not args.model:
        parser.error("--model needs at least one model name (use --list to see them)")

    names = args.model or list(MODELS)
    unknown = [name for name in names if name not in MODELS and name not in REMOTE]
    if unknown:
        parser.error(f"unknown model(s): {unknown}")
    for name in names:
        print(f"[{name}]")
        if name in REMOTE:
            print(f"  skip    {REMOTE[name]}")
            continue
        download(name, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

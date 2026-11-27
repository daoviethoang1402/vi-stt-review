"""Command-line entry of the survey harness.

    python run.py                              # run every model of the registry
    python run.py --model a b                  # run only these models
    python run.py --list                       # list the models
    python run.py --hotword yes|no|both        # conditions (default: both)
    python run.py --warmup false               # skip the warm-up samples
    python run.py --manifest m.jsonl --hotwords h.yaml --output out/

Model parameters are never CLI flags: they all live in `registry.yaml`.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

from cuda_env import ensure_process_env
from harness import probe, report
from harness.registry import ROOT, Registry, load
from harness.runner import (
    HOTWORD_CHOICES,
    STATUS_OK,
    STATUS_RUN_FAILED,
    ModelResult,
    RunContext,
    load_canonicals,
    load_manifest,
    run_model,
)

DEFAULTS = {
    "registry": ROOT / "registry.yaml",
    "manifest": ROOT / "data" / "manifest.jsonl",
    "hotwords": ROOT / "data" / "hotwords.yaml",
    "output": ROOT / "report",
}


def _bool(text: str) -> bool:
    value = text.strip().lower()
    if value in ("true", "yes", "1"):
        return True
    if value in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Benchmark Vietnamese speech-to-text models on one shared dataset.",
        epilog="Without --model every model of the registry is run.",
    )
    parser.add_argument("--model", nargs="*", metavar="NAME",
                        help="run only these models (at least one name; see --list)")
    parser.add_argument("--list", action="store_true", help="list the models of the registry and exit")
    parser.add_argument("--hotword", choices=sorted(HOTWORD_CHOICES), default="both",
                        help="no: only without hotwords (WER only); yes: only with hotwords; "
                             "both: without then with (default)")
    parser.add_argument("--warmup", type=_bool, default=True, metavar="true|false",
                        help="run the first `run.warmup_samples` samples once before measuring "
                             "(default: true)")
    parser.add_argument("--registry", type=Path, default=DEFAULTS["registry"],
                        help="model registry (default: registry.yaml)")
    parser.add_argument("--manifest", type=Path, default=DEFAULTS["manifest"],
                        help="input dataset manifest; audio paths are relative to its folder "
                             "(default: data/manifest.jsonl)")
    parser.add_argument("--hotwords", type=Path, default=DEFAULTS["hotwords"],
                        help="keyword list, used both as decoder hotwords and for scoring "
                             "(default: data/hotwords.yaml)")
    parser.add_argument("--output", type=Path, default=DEFAULTS["output"],
                        help="output report directory (default: report/)")
    parser.add_argument("--threads", type=int, help="override `run.num_threads`")
    # internal: run exactly one model in this process and pickle the result
    parser.add_argument("--single", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    return parser


def build_context(reg: Registry, args) -> RunContext:
    return RunContext(
        records=load_manifest(args.manifest),
        data_dir=args.manifest.resolve().parent,
        canonicals=load_canonicals(args.hotwords),
        conditions=HOTWORD_CHOICES[args.hotword],
        warmup_samples=reg.warmup_samples if args.warmup else 0,
        num_threads=args.threads or reg.num_threads,
    )


def list_models(reg: Registry) -> None:
    print(f"{'model':30}{'adapter':16}{'device':8}{'precision':11}hotwords")
    for spec in reg.specs:
        hot = (f"on · case={spec.hotword_case} · score={spec.config.get('hotword_score')}"
               if spec.hotword_enabled else "off")
        print(f"{spec.name:30}{spec.adapter:16}{spec.device:8}{spec.dtype or '-':11}{hot}")
    print(f"\n{len(reg.specs)} model(s) in {reg.path.name}")


def _child_command(spec_name: str, out: Path, args) -> list[str]:
    command = [
        sys.executable, "-m", "harness.cli",
        "--single", spec_name, "--out", str(out),
        "--registry", str(args.registry), "--manifest", str(args.manifest),
        "--hotwords", str(args.hotwords), "--output", str(args.output),
        "--hotword", args.hotword, "--warmup", str(args.warmup).lower(),
    ]
    if args.threads:
        command += ["--threads", str(args.threads)]
    return command


def run_isolated(reg: Registry, args) -> list[ModelResult]:
    """Run each model in its own process.

    Needed for honest numbers: VmHWM only grows, and two GPU models in one process
    would also share (and pollute) the VRAM measurement.
    """
    results = []
    for index, spec in enumerate(reg.specs, 1):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as handle:
            out = Path(handle.name)
        print(f"[{index}/{len(reg.specs)}] {spec.name}: running in a separate process ...", flush=True)
        completed = subprocess.run(_child_command(spec.name, out, args), cwd=ROOT)
        if completed.returncode == 0 and out.is_file() and out.stat().st_size:
            result = pickle.loads(out.read_bytes())
        else:
            result = ModelResult(spec=spec, status=STATUS_RUN_FAILED,
                                 error=f"worker process exited with code {completed.returncode}")
        out.unlink(missing_ok=True)
        if result.status != STATUS_OK:
            print(f"[{index}/{len(reg.specs)}] {spec.name}: {result.status} — {result.error} "
                  "(skipping to the next model)", flush=True)
        results.append(result)
    return results


def _sha16(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        # Must happen before any CUDA library is loaded (see cuda_env.py).
        ensure_process_env(["-m", "harness.cli"])

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model is not None and not args.model:
        parser.error("--model needs at least one model name (use --list to see them)")

    registry = load(args.registry, only=args.model or ([args.single] if args.single else None))
    if args.list:
        list_models(registry)
        return 0

    if args.single:
        if not args.out:
            raise SystemExit("--single needs --out")
        result = run_model(registry.by_name(args.single), build_context(registry, args))
        Path(args.out).write_bytes(pickle.dumps(result))
        return 0

    context = build_context(registry, args)
    results = run_isolated(registry, args)
    fingerprint = probe.fingerprint(
        context.num_threads,
        extra={
            "models": [spec.name for spec in registry.specs],
            "hotword": args.hotword,
            "warmup_samples": context.warmup_samples,
            "samples": len(context.records),
            "manifest": str(args.manifest),
            "manifest_sha256_16": _sha16(args.manifest),
            "hotwords": str(args.hotwords),
            "hotwords_sha256_16": _sha16(args.hotwords),
        },
    )
    summary = report.write_all(results, fingerprint, args.output)

    print(f"\n{'model':30}{'status':14}detail")
    for result in results:
        print(f"{result.spec.name:30}{result.status:14}{result.error or ''}")
    print(f"-> {summary}")
    if all(result.status != STATUS_OK for result in results):
        print("No model could be measured.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

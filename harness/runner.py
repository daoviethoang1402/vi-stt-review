"""Run one model over the dataset and return per-sample scores and resource probes.

Per model (docs/OVERVIEW.md, *Quy trình tính các metric*):

- optional warm-up on the first few samples (not recorded);
- `off` (no hotwords): one pass, only WER is reported;
- `on` (hotwords): one pass, WER / Recall / Precision / FA/1k + resource footprint.

There are no gates: a model that loads is always measured and reported.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
import yaml

import adapters
import scoring
from adapters.base import BOOST_OFF, BOOST_ON, ModelUnavailable, Segment

from . import probe

#: `--hotword` choice → conditions run, in order.
HOTWORD_CHOICES = {"no": (BOOST_OFF,), "yes": (BOOST_ON,), "both": (BOOST_OFF, BOOST_ON)}
#: Casing applied to hotwords so they exist in the model's vocabulary.
HOTWORD_CASES = ("as_is", "upper", "lower")

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_LOAD_FAILED = "load_failed"
STATUS_RUN_FAILED = "run_failed"


def load_manifest(path: Path) -> list[dict]:
    """Read a `manifest.jsonl`; audio paths are relative to the manifest's folder."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_canonicals(path: Path) -> list[str]:
    """Every canonical keyword of `hotwords.yaml`, distractors included."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [entry["canonical"] for entry in doc["entries"]]


def cased(canonicals: list[str], case: str) -> list[str]:
    """Hotwords for the `on` condition, cased to match the model vocabulary."""
    if case not in HOTWORD_CASES:
        raise ValueError(f"hotword_case {case!r} not in {HOTWORD_CASES}")
    convert = {"upper": str.upper, "lower": str.lower}.get(case, str)
    return list(dict.fromkeys(convert(item) for item in canonicals))


def percentile(values: list[float], fraction: float) -> float | None:
    """Linear-interpolated percentile; `None` for an empty input."""
    items = sorted(values)
    if not items:
        return None
    position = fraction * (len(items) - 1)
    lower = int(position)
    upper = min(lower + 1, len(items) - 1)
    return items[lower] + (items[upper] - items[lower]) * (position - lower)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    adapter: str
    #: runtime precision (`fp32` / `fp16` / `bf16` / `int8`), reported as `precision`
    dtype: str = ""
    #: adapter parameters — the registry is the only place declaring them
    config: dict = field(default_factory=dict)

    @property
    def device(self) -> str:
        """`cpu` / `cuda` / `remote`, from the registry."""
        return str(self.config.get("provider", "cpu"))

    @property
    def hotword_enabled(self) -> bool:
        return bool(self.config.get("hotword_enabled", False))

    @property
    def hotword_case(self) -> str:
        return str(self.config.get("hotword_case") or "as_is")


@dataclass
class RunContext:
    records: list[dict]
    #: folder the manifest's relative audio paths are resolved against
    data_dir: Path
    #: canonical keywords: fed to decoders (`on`) and, normalized, used for scoring
    canonicals: list[str]
    conditions: tuple[str, ...] = HOTWORD_CHOICES["both"]
    warmup_samples: int = 0
    num_threads: int = 4

    def __post_init__(self) -> None:
        self.vocab = scoring.keyword_vocab(self.canonicals)


@dataclass
class ModelResult:
    spec: ModelSpec
    status: str = STATUS_OK
    #: condition → per-sample rows
    levels: dict[str, list[dict]] = field(default_factory=dict)
    resource: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def _adapter_config(adapter, ctx: RunContext, spec: ModelSpec) -> dict:
    config = {**spec.config, "num_threads": ctx.num_threads}
    if getattr(adapter, "needs_ground_truth", False):  # control adapters only
        config["references"] = {record["id"]: record["ref"] for record in ctx.records}
        config["acronyms"] = {record["id"]: list(record.get("spoken_form") or []) for record in ctx.records}
    return config


def _run_pass(adapter, ctx: RunContext, spec: ModelSpec, level: str, records: list[dict]) -> list[dict]:
    """One pass over `records` under one condition."""
    granted = cased(ctx.canonicals, spec.hotword_case) if level == BOOST_ON else []
    session = adapter.open_session(granted, level)
    rows: list[dict] = []
    try:
        for record in records:
            path = ctx.data_dir / record["audio"]
            info = sf.info(path)
            started = time.perf_counter()
            text = session.transcribe(Segment(id=record["id"], audio=path))
            latency_ms = (time.perf_counter() - started) * 1000.0
            scored = scoring.score_record(record, text, ctx.vocab)
            rows.append({
                "id": record["id"],
                "data_class": record["data_class"],
                "ref": record["ref"],
                "text_raw": text,
                "normalized_output": scored.normalized_output,
                "heard": sorted(scored.heard),
                "label": sorted(scored.label),
                **vars(scored.score),
                "fa": scored.score.fa,
                "audio_ms": info.frames / info.samplerate * 1000.0,
                "latency_ms": latency_ms,
            })
    finally:
        session.close()
    return rows


def _resource(spec: ModelSpec, adapter, rows: list[dict], condition: str, sampler: probe.VramSampler) -> dict:
    latencies = [row["latency_ms"] for row in rows]
    total_audio = sum(row["audio_ms"] for row in rows)
    total_time = sum(latencies)
    weights = adapter.weights_mb()
    remote = spec.device == "remote"
    rounded = lambda value, digits=1: None if value is None else round(value, digits)  # noqa: E731
    return {
        "condition": condition,
        "device": spec.device,
        "precision": spec.dtype or None,
        "model_size_mb": rounded(weights),
        # the client process of a remote model says nothing about the model itself
        "peak_rss_mb": None if remote else rounded(probe.peak_rss_mb()),
        "peak_vram_mb": rounded(sampler.peak_mb),
        "vram_method": sampler.method,
        "latency_p50_ms": rounded(percentile(latencies, 0.50)),
        "latency_p95_ms": rounded(percentile(latencies, 0.95)),
        # micro: total processing time / total audio, and its inverse
        "rtf": round(total_time / total_audio, 4) if total_audio else None,
        "throughput": round(total_audio / total_time, 2) if total_time else None,
    }


def run_model(spec: ModelSpec, ctx: RunContext) -> ModelResult:
    """Warm-up → `off` → `on` for one model; writes no file."""
    result = ModelResult(spec=spec)
    conditions = list(ctx.conditions)
    if BOOST_ON in conditions and not spec.hotword_enabled:
        conditions.remove(BOOST_ON)
        result.notes.append("hotword_enabled: false in the registry — `on` not run")
    if not conditions:
        result.status, result.error = STATUS_RUN_FAILED, "no condition left to run"
        return result

    sampler = probe.VramSampler(enabled=spec.device == "cuda").start()
    adapter = adapters.create(spec.adapter)
    try:
        adapter.load(_adapter_config(adapter, ctx, spec))
        if ctx.warmup_samples > 0:
            _run_pass(adapter, ctx, spec, conditions[0], ctx.records[: ctx.warmup_samples])
    except ModelUnavailable as exc:
        result.status, result.error = STATUS_UNAVAILABLE, str(exc)
    except Exception as exc:  # noqa: BLE001 - a load failure is a result, not a crash
        result.status, result.error = STATUS_LOAD_FAILED, f"{type(exc).__name__}: {exc}"
    if result.status != STATUS_OK:
        sampler.stop()
        return result

    try:
        for level in conditions:
            result.levels[level] = _run_pass(adapter, ctx, spec, level, ctx.records)
    except Exception as exc:  # noqa: BLE001
        result.status, result.error = STATUS_RUN_FAILED, f"{type(exc).__name__}: {exc}"
    finally:
        sampler.stop()

    measured = BOOST_ON if BOOST_ON in result.levels else next(iter(result.levels), None)
    if measured is not None:
        result.resource = _resource(spec, adapter, result.levels[measured], measured, sampler)
        if measured != BOOST_ON:
            result.notes.append("resource measured on `off` (`on` was not run)")
    if BOOST_ON in result.levels and scoring.Aggregate.of(result.levels[BOOST_ON]).a == 0:
        result.notes.append(
            "Σ|A| = 0: the model printed no canonical keyword at all — inspect `text_raw` in "
            "per_sample/ (model limitation vs adapter bug)"
        )
    return result

"""Read and validate `registry.yaml` — the only place that knows about models.

This module never runs a model. Every typo or contradiction is rejected here, before
any model is loaded. Each model has only `name` + `config`; the per-adapter key
contract lives in `adapters.CONTRACTS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

import adapters
from harness.runner import HOTWORD_CASES, ModelSpec

ROOT = Path(__file__).resolve().parent.parent

ALLOWED_TOP = {"version", "run", "models"}
ALLOWED_MODEL = {"name", "config"}
ALLOWED_RUN = {"num_threads", "warmup_samples"}
PROVIDERS = ("cpu", "cuda", "remote")


class RegistryError(SystemExit):
    """Invalid registry — stop before loading any model."""

    def __init__(self, path: Path, problems: list[str]):
        body = "\n".join(f"  - {item}" for item in problems)
        super().__init__(f"{path}: {len(problems)} configuration error(s)\n{body}")


@dataclass(frozen=True)
class Registry:
    path: Path
    num_threads: int
    warmup_samples: int
    specs: tuple[ModelSpec, ...] = field(default_factory=tuple)

    def by_name(self, name: str) -> ModelSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(f"no model {name!r} in {self.path.name}")

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]


def _check_keys(where: str, given, allowed, problems: list[str]) -> None:
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        problems.append(f"{where}: unknown key(s) {unknown}; allowed {sorted(allowed)}")


def _check_model(entry, seen: set[str], problems: list[str]) -> ModelSpec | None:
    if not isinstance(entry, dict):
        problems.append(f"models[]: must be a mapping, got {type(entry).__name__}")
        return None
    if "name" not in entry:
        problems.append("models[]: missing `name`")
        return None

    name = str(entry["name"]).strip()
    where = f"models[{name}]"
    _check_keys(where, entry, ALLOWED_MODEL, problems)
    if name in seen:
        problems.append(f"{where}: duplicate `name`")
    seen.add(name)
    config = entry.get("config")
    if not isinstance(config, dict):
        problems.append(f"{where}: `config` missing or not a mapping")
        return None

    adapter = config.get("adapter")
    if not adapter:
        problems.append(f"{where}.config: missing `adapter`")
        return None
    if adapter not in adapters.FACTORIES:
        problems.append(f"{where}.config.adapter: {adapter!r} unknown; known {sorted(adapters.FACTORIES)}")
        return None

    deal = adapters.CONTRACTS[adapter]
    _check_keys(f"{where}.config", config, adapters.COMMON_KEYS | deal["allowed"], problems)
    for key in sorted(deal["required"]):
        if not config.get(key) and config.get(key) != 0:
            problems.append(f"{where}.config: missing `{key}` (required by {adapter})")

    flag = deal.get("dtype_flag")
    if flag:
        key, dtypes = flag
        value = config.get(key)
        if not isinstance(value, bool):
            problems.append(f"{where}.config.{key}: must be true or false, got {value!r}")
        elif config.get("dtype") != dtypes[value]:
            problems.append(f"{where}.config.dtype: must be {dtypes[value]!r} when {key}: "
                            f"{str(value).lower()}, got {config.get('dtype')!r}")

    provider = config.get("provider", "cpu")
    if provider not in PROVIDERS:
        problems.append(f"{where}.config.provider: {provider!r} not in {list(PROVIDERS)}")

    if "hotword_enabled" not in config:
        problems.append(f"{where}.config: missing `hotword_enabled: true|false`")

    if config.get("hotword_enabled"):
        case = config.get("hotword_case", "as_is")
        if case not in HOTWORD_CASES:
            problems.append(f"{where}.config.hotword_case: {case!r} not in {list(HOTWORD_CASES)}")

        param = deal["hotword_param"]
        if param is not None:
            score = config.get("hotword_score")
            if not isinstance(score, (int, float)) or score <= 0:
                problems.append(
                    f"{where}.config.hotword_score: must be a number > 0 for {adapter}, "
                    f"got {score!r} (0 makes `on` identical to `off`)"
                )
            if config.get("hotword_score_param") != param:
                problems.append(
                    f"{where}.config.hotword_score_param: must be {param!r} (the runtime's "
                    f"name), got {config.get('hotword_score_param')!r}"
                )
        for key, expected in (deal.get("requires_with_hotword") or {}).items():
            if config.get(key) != expected:
                problems.append(
                    f"{where}.config.{key}: must be {expected!r} with hotwords, got "
                    f"{config.get(key)!r} (other values are silently ignored by the runtime)"
                )
        if deal["needs_prompt"] and not config.get("hotword_prompt"):
            problems.append(f"{where}.config.hotword_prompt: {adapter} only takes hotwords via a prompt")
        prompt = config.get("hotword_prompt")
        if prompt and "{words}" not in str(prompt):
            problems.append(f"{where}.config.hotword_prompt: must contain `{{words}}`")

    return ModelSpec(name=name, adapter=str(adapter), dtype=str(config.get("dtype") or ""),
                     config=dict(config))


def load(path: Path | str | None = None, only: list[str] | None = None) -> Registry:
    """Read and validate the registry. Any problem ⇒ `RegistryError` listing all of them."""
    target = Path(path) if path else ROOT / "registry.yaml"
    if not target.is_file():
        raise RegistryError(target, ["registry file not found"])

    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RegistryError(target, ["top level must be a mapping"])
    problems: list[str] = []
    _check_keys("registry", raw, ALLOWED_TOP, problems)
    run_cfg = raw.get("run") or {}
    _check_keys("run", run_cfg, ALLOWED_RUN, problems)

    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for entry in raw.get("models") or []:
        spec = _check_model(entry, seen, problems)
        if spec is not None and (not only or spec.name in only):
            specs.append(spec)

    if only:
        missing = sorted(set(only) - seen)
        if missing:
            problems.append(f"--model: not in the registry: {missing}")
        # keep the order the user asked for
        specs.sort(key=lambda spec: only.index(spec.name))
    if not specs and not problems:
        problems.append("models: the registry has no model")
    if problems:
        raise RegistryError(target, problems)

    return Registry(
        path=target,
        num_threads=int(run_cfg.get("num_threads") or 4),
        warmup_samples=int(run_cfg.get("warmup_samples", 3)),
        specs=tuple(specs),
    )

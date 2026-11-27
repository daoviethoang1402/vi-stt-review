"""Write the report artifacts of one run (docs/PIPELINE.md, *Artifact*).

    summary.md      the four tables to read
    accuracy.csv    one row per model × condition × group, raw counts + rates
    resource.csv    one row per model
    per_sample/     one JSONL per model: every output, A, L and latency
    fingerprint.json  (written by the CLI)

Cell conventions:
    —    undefined, needs a look (e.g. Precision when the model printed no keyword)
    n/a  not applicable (Recall/Precision of a group whose labels hold no keyword)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from adapters.base import BOOST_OFF, BOOST_ON

from scoring import ALL, Aggregate, by_group

from .runner import STATUS_OK, ModelResult

UNDEFINED = "—"
NOT_APPLICABLE = "n/a"


def _pct(value: float | None) -> str:
    return UNDEFINED if value is None else f"{value * 100:.1f}%"


def _num(value, digits: int = 2) -> str:
    return UNDEFINED if value is None else f"{value:.{digits}f}"


def _csv_rate(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _groups(results: list[ModelResult]) -> list[str]:
    names = {
        row["data_class"]
        for result in results
        for rows in result.levels.values()
        for row in rows
    }
    return [ALL, *sorted(names)]


def write_per_sample(results: list[ModelResult], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for result in results:
        if not result.levels:
            continue
        with (directory / f"{result.spec.name}.jsonl").open("w", encoding="utf-8") as handle:
            for condition, rows in result.levels.items():
                for row in rows:
                    handle.write(json.dumps({"condition": condition, **row}, ensure_ascii=False) + "\n")


def write_accuracy_csv(results: list[ModelResult], path: Path) -> None:
    columns = ["model", "condition", "group", "samples", "n", "e", "a", "l", "tp", "fa",
               "wer", "recall", "precision", "fa_per_1k"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            for condition, rows in result.levels.items():
                for group, agg in by_group(rows).items():
                    keyword = condition == BOOST_ON  # `off` reports WER only
                    writer.writerow({
                        "model": result.spec.name, "condition": condition, "group": group,
                        "samples": agg.samples, "n": agg.n, "e": agg.e, "a": agg.a,
                        "l": agg.l, "tp": agg.tp, "fa": agg.fa,
                        "wer": _csv_rate(agg.wer),
                        "recall": _csv_rate(agg.recall) if keyword else "",
                        "precision": _csv_rate(agg.precision) if keyword else "",
                        "fa_per_1k": _csv_rate(agg.fa_per_1k) if keyword else "",
                    })


RESOURCE_COLUMNS = ["condition", "device", "precision", "model_size_mb", "peak_rss_mb",
                    "peak_vram_mb", "vram_method", "latency_p50_ms", "latency_p95_ms",
                    "rtf", "throughput"]


def write_resource_csv(results: list[ModelResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "status", *RESOURCE_COLUMNS])
        writer.writeheader()
        for result in results:
            row = {"model": result.spec.name, "status": result.status}
            row.update({key: result.resource.get(key, "") for key in RESOURCE_COLUMNS})
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _keyword_cells(agg) -> tuple[str, str]:
    """Recall/Precision cells of one group; `n/a` when its labels hold no keyword."""
    if agg.l == 0:
        return NOT_APPLICABLE, NOT_APPLICABLE
    return _pct(agg.recall), _pct(agg.precision)


def write_summary_md(results: list[ModelResult], fingerprint: dict, path: Path) -> None:
    groups = _groups(results)
    classes = [group for group in groups if group != ALL]
    measured = [result for result in results if result.levels]
    lines = [
        "# STT survey results",
        "",
        f"Generated `{fingerprint['timestamp']}` · GPU `{fingerprint.get('gpu') or 'none'}` · "
        f"`{fingerprint['platform']}` · {fingerprint['num_threads']} threads · "
        f"{fingerprint.get('samples', '?')} samples · hotword `{fingerprint.get('hotword', '?')}`",
        "",
    ]

    off = [result for result in measured if BOOST_OFF in result.levels]
    if off:
        lines += [
            "## 1. WER without hotwords (`off`)",
            "",
            "| Model | " + " | ".join(f"`{group}`" for group in groups) + " |",
            "|---|" + "---:|" * len(groups),
        ]
        for result in off:
            aggs = by_group(result.levels[BOOST_OFF])
            cells = " | ".join(_pct(aggs[group].wer) if group in aggs else UNDEFINED for group in groups)
            lines.append(f"| `{result.spec.name}` | {cells} |")
        lines.append("")

    on = [result for result in measured if BOOST_ON in result.levels]
    if on:
        lines += [
            "## 2. Accuracy with hotwords (`on`) — whole dataset",
            "",
            "| Model | WER | Recall | Precision | FA/1k | FA | Σ\\|L\\| | Σ\\|A\\| |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for result in on:
            agg = Aggregate.of(result.levels[BOOST_ON])
            lines.append(
                f"| `{result.spec.name}` | {_pct(agg.wer)} | {_pct(agg.recall)} | "
                f"{_pct(agg.precision)} | {_num(agg.fa_per_1k)} | {agg.fa} | {agg.l} | {agg.a} |"
            )
        lines += [
            "",
            "## 3. Appendix — accuracy with hotwords (`on`) per data group",
            "",
            "| Model | Group | WER | Recall | Precision | FA/1k | FA |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for result in on:
            aggs = by_group(result.levels[BOOST_ON])
            for group in classes:
                if group not in aggs:
                    continue
                agg = aggs[group]
                recall, precision = _keyword_cells(agg)
                lines.append(
                    f"| `{result.spec.name}` | `{group}` | {_pct(agg.wer)} | {recall} | "
                    f"{precision} | {_num(agg.fa_per_1k)} | {agg.fa} |"
                )
        lines.append("")

    if measured:
        lines += [
            "## 4. Resource footprint",
            "",
            "| Model | Device | Precision | Model size | peak_rss | peak_vram | p50 latency | "
            "p95 latency | Throughput (×RT) | Measured on |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for result in measured:
            res = result.resource
            mb = lambda key: UNDEFINED if res.get(key) is None else f"{res[key]:.0f} MB"  # noqa: E731
            ms = lambda key: UNDEFINED if res.get(key) is None else f"{res[key]:.0f} ms"  # noqa: E731
            lines.append(
                f"| `{result.spec.name}` | {res.get('device', UNDEFINED)} | "
                f"{res.get('precision') or UNDEFINED} | {mb('model_size_mb')} | {mb('peak_rss_mb')} | "
                f"{mb('peak_vram_mb')} | {ms('latency_p50_ms')} | {ms('latency_p95_ms')} | "
                f"{_num(res.get('throughput'), 1)} | `{res.get('condition', UNDEFINED)}` |"
            )
        lines.append("")

    failed = [result for result in results if result.status != STATUS_OK]
    if failed:
        lines += ["## Models not measured / failed", "", "| Model | Status | Reason |", "|---|---|---|"]
        for result in failed:
            reason = (result.error or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{result.spec.name}` | {result.status} | {reason} |")
        lines.append("")

    noted = [result for result in results if result.notes]
    if noted:
        lines += ["## Notes", ""]
        for result in noted:
            for note in result.notes:
                lines.append(f"- `{result.spec.name}`: {note}")
        lines.append("")

    lines += [
        "## How to read",
        "",
        "- All rates are **micro**: raw counts are summed over samples, then divided.",
        "- `WER` = Σ(S+D+I) / ΣN on tokens after normalization (NFC, lowercase, punctuation "
        "removed). A spelled-out keyword (`a dê hát` for `ADH`) costs several tokens.",
        "- Keywords are matched on their **canonical** form only, as whole tokens, case-insensitive. "
        "`A` = keywords of `hotwords.yaml` (distractors included) present in the output; "
        "`L` = keywords annotated in the label; both are sets per sample.",
        "- `Recall` = Σ|A∩L| / Σ|L| · `Precision` = Σ|A∩L| / Σ|A| · `FA/1k` = Σ|A\\L| / ΣN × 1000; "
        "`FA` is the raw false-alarm count.",
        f"- `{UNDEFINED}` = undefined (e.g. Precision when Σ|A| = 0: the model printed no keyword — "
        f"inspect per_sample/). `{NOT_APPLICABLE}` = not applicable (the group's labels hold no keyword).",
        "- `peak_rss` = peak host RAM of the whole process (VmHWM). `peak_vram` = peak VRAM of the "
        "process (per-process driver figure, or device delta against the pre-load baseline).",
        "- Latency = wall time of one `transcribe()` call; `Throughput` = Σ audio / Σ processing time "
        "(inverse RTF). Resources are measured on the `on` pass. A remote model's latency includes "
        "the network round-trip and it has no local RAM/VRAM figures.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_all(results: list[ModelResult], fingerprint: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fingerprint.json").write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_accuracy_csv(results, directory / "accuracy.csv")
    write_resource_csv(results, directory / "resource.csv")
    write_per_sample(results, directory / "per_sample")
    summary = directory / "summary.md"
    write_summary_md(results, fingerprint, summary)
    return summary

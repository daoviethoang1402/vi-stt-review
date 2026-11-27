"""Harness tests on the control adapters (no real model, no GPU), plus the CLI.

    python -m unittest tests.test_harness -v
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from adapters.base import BOOST_OFF, BOOST_ON
from harness import cli, report
from harness.runner import (
    STATUS_OK,
    STATUS_UNAVAILABLE,
    ModelSpec,
    RunContext,
    load_canonicals,
    load_manifest,
    run_model,
)
from scoring import ALL, Aggregate, by_group
from tests import controls

controls.register()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RECORDS = load_manifest(DATA_DIR / "manifest.jsonl")
CANONICALS = load_canonicals(DATA_DIR / "hotwords.yaml")


def spec(adapter: str, hotword_enabled: bool = True, **config) -> ModelSpec:
    return ModelSpec(name=adapter, adapter=adapter, dtype="fp32",
                     config={"adapter": adapter, "hotword_enabled": hotword_enabled, **config})


def context(**overrides) -> RunContext:
    values = dict(records=RECORDS, data_dir=DATA_DIR, canonicals=CANONICALS, warmup_samples=1, num_threads=1)
    return RunContext(**{**values, **overrides})


class ControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = {name: run_model(spec(name), context()) for name in controls.CONTROLS}

    def test_oracle_is_perfect(self) -> None:
        result = self.results["oracle"]
        self.assertEqual(result.status, STATUS_OK)
        agg = Aggregate.of(result.levels[BOOST_ON])
        self.assertEqual((agg.wer, agg.recall, agg.precision, agg.fa), (0.0, 1.0, 1.0, 0))
        self.assertEqual(Aggregate.of(result.levels[BOOST_OFF]).wer, 0.0)

    def test_dummy_prints_no_keyword(self) -> None:
        result = self.results["dummy"]
        agg = Aggregate.of(result.levels[BOOST_ON])
        self.assertEqual((agg.wer, agg.a, agg.recall), (1.0, 0, 0.0))
        self.assertIsNone(agg.precision)
        self.assertTrue(any("Σ|A| = 0" in note for note in result.notes))

    def test_boostable_needs_hotwords_and_spelled_forms_never_count(self) -> None:
        result = self.results["boostable"]
        off, on = Aggregate.of(result.levels[BOOST_OFF]), Aggregate.of(result.levels[BOOST_ON])
        self.assertEqual((off.recall, on.recall), (0.0, 1.0))
        self.assertGreater(off.wer, on.wer)

    def test_only_the_acronym_group_has_label_keywords(self) -> None:
        groups = by_group(self.results["oracle"].levels[BOOST_ON])
        self.assertEqual(set(groups), {ALL, "vi", "vi-en", "vi-en-acro"})
        self.assertEqual((groups["vi"].l, groups["vi-en"].l), (0, 0))
        self.assertEqual(groups[ALL].l, groups["vi-en-acro"].l)

    def test_resource_is_measured_on_the_on_pass(self) -> None:
        resource = self.results["oracle"].resource
        self.assertEqual(resource["condition"], BOOST_ON)
        for key in ("precision", "latency_p50_ms", "latency_p95_ms", "throughput", "peak_rss_mb"):
            self.assertIn(key, resource)


class ConditionTest(unittest.TestCase):
    def test_hotword_choices(self) -> None:
        for conditions in ((BOOST_OFF,), (BOOST_ON,), (BOOST_OFF, BOOST_ON)):
            with self.subTest(conditions=conditions):
                result = run_model(spec("oracle"), context(conditions=conditions))
                self.assertEqual(tuple(result.levels), conditions)
                self.assertEqual(result.resource["condition"], conditions[-1])

    def test_disabled_hotword_skips_on(self) -> None:
        result = run_model(spec("oracle", hotword_enabled=False), context())
        self.assertEqual(list(result.levels), [BOOST_OFF])

    def test_unreachable_api_is_skipped_not_raised(self) -> None:
        model = spec("qwen3-asr-api", provider="remote", url="http://127.0.0.1:9/transcribe",
                     timeout_s=2, ping_attempts=2, hotword_prompt="{words}")
        result = run_model(model, context())
        self.assertEqual(result.status, STATUS_UNAVAILABLE)
        self.assertIn("unreachable after 2 attempt(s)", result.error)
        self.assertEqual(result.levels, {})


class ReportTest(unittest.TestCase):
    def test_report_files_and_tables(self) -> None:
        results = [run_model(spec(name), context()) for name in ("oracle", "dummy")]
        fingerprint = {"timestamp": "t", "platform": "p", "num_threads": 1, "gpu": None,
                       "samples": len(RECORDS), "hotword": "both"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = report.write_all(results, fingerprint, out).read_text(encoding="utf-8")
            for heading in ("## 1. WER without hotwords", "## 2. Accuracy with hotwords",
                            "## 3. Appendix", "## 4. Resource footprint"):
                self.assertIn(heading, summary)
            appendix = summary.split("## 3. Appendix")[1].split("## 4.")[0]
            vi_row = next(line for line in appendix.splitlines() if "| `oracle` | `vi` |" in line)
            self.assertEqual(vi_row.count("n/a"), 2)
            overall = summary.split("## 2.")[1].split("## 3.")[0]
            self.assertIn("—", next(line for line in overall.splitlines() if line.startswith("| `dummy`")))

            with (out / "accuracy.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["recall"] == "" for row in rows if row["condition"] == BOOST_OFF))
            self.assertEqual({row["group"] for row in rows}, {"all", "vi", "vi-en", "vi-en-acro"})
            self.assertTrue((out / "resource.csv").is_file())
            self.assertTrue((out / "per_sample" / "oracle.jsonl").is_file())


class CliTest(unittest.TestCase):
    def test_flags(self) -> None:
        args = cli.build_parser().parse_args([])
        self.assertEqual((args.model, args.hotword, args.warmup), (None, "both", True))
        args = cli.build_parser().parse_args(["--warmup", "false", "--hotword", "no", "--model", "a", "b"])
        self.assertEqual((args.model, args.hotword, args.warmup), (["a", "b"], "no", False))

    def test_model_flag_without_names_is_an_error(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            cli.main(["--model"])
        self.assertEqual(caught.exception.code, 2)

    def test_list(self) -> None:
        self.assertEqual(cli.main(["--list"]), 0)


if __name__ == "__main__":
    unittest.main()

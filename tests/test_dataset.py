"""Dataset tooling tests: manifest schema, the acronym-sequence rule, hotwords.yaml.

    python -m unittest tests.test_dataset -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from data import validate_dataset as vd
from data.prepare_dataset import canonicalize_reference
from scoring import normalize

ROOT = Path(__file__).resolve().parent.parent
RECORDS = vd.load_manifest()
HOTWORDS = yaml.safe_load((ROOT / "data" / "hotwords.yaml").read_text(encoding="utf-8"))
KEYWORDS = {normalize(entry["canonical"]) for entry in HOTWORDS["entries"]}


def record(ref: str, acronyms: list[str], **extra) -> dict:
    base = {"id": "x", "data_class": "vi-en-acro", "audio": RECORDS[60]["audio"], "ref": ref,
            "spoken_form": acronyms, "annotation_status": "reviewed"}
    return {**base, **extra}


class ManifestTest(unittest.TestCase):
    def test_real_dataset_is_valid(self) -> None:
        self.assertEqual(vd.validate(RECORDS, HOTWORDS), [])

    def test_scoring_normalizer_agrees_with_the_dataset_builder(self) -> None:
        for item in RECORDS:
            with self.subTest(sample=item["id"]):
                self.assertEqual(normalize(item["ref"]), canonicalize_reference(item["ref"]))

    def test_acronym_sequence_must_match_ref(self) -> None:
        cases = {
            "ok": (record("hdl và ldl", ["HDL", "LDL"]), False),
            "repeat kept": (record("nadh rồi nadh", ["NADH", "NADH"]), False),
            "out of order": (record("hdl và ldl", ["LDL", "HDL"]), True),
            "missing repeat": (record("nadh rồi nadh", ["NADH"]), True),
            "extra": (record("hdl thôi", ["HDL", "LDL"]), True),
            "unannotated keyword token": (record("hdl và ldl", ["HDL"]), True),
        }
        for name, (item, broken) in cases.items():
            with self.subTest(case=name):
                errors = [e for e in vd.check_record(item, KEYWORDS) if "spoken_form" in e]
                self.assertEqual(bool(errors), broken, errors)

    def test_keyword_equal_to_an_ordinary_word_is_caught(self) -> None:
        errors = vd.check_record(record("ai đang nói", [], data_class="vi"), KEYWORDS | {"ai"})
        self.assertTrue(any("spoken_form" in e for e in errors))

    def test_required_fields_come_first(self) -> None:
        item = record("hdl", ["HDL"])
        reordered = {"keywords": [], **item}
        self.assertTrue(any("must come first" in e for e in vd.check_record(reordered, KEYWORDS)))
        self.assertTrue(any("unknown field" in e for e in vd.check_record({**item, "ref_alt": []}, KEYWORDS)))


class HotwordsTest(unittest.TestCase):
    def test_wrong_distractor_flag_and_bad_format_are_reported(self) -> None:
        records = [record("mti", ["MTI"])]
        doc = {"entries": [{"canonical": "MTI", "distractor": True}, {"canonical": "a-b", "distractor": True}]}
        errors = " ".join(vd.check_hotwords(doc, records))
        self.assertIn("MTI", errors)
        self.assertIn("a-b", errors)


if __name__ == "__main__":
    unittest.main()

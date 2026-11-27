"""Scoring tests (docs/METRIC.md). Harness-level tests live in test_harness.py.

    python -m unittest tests.test_scoring -v
"""

from __future__ import annotations

import inspect
import unittest

import scoring

VOCAB = scoring.keyword_vocab(["MTI", "HDL", "MRI"])


def record(ref: str, canonicals: tuple[str, ...] = ()) -> dict:
    return {
        "id": "t_001",
        "ref": ref,
        "spoken_form": list(canonicals),
    }


def score(ref: str, output: str, canonicals: tuple[str, ...] = ()) -> scoring.SampleScore:
    return scoring.score_record(record(ref, canonicals), output, VOCAB).score


class WerTest(unittest.TestCase):
    def test_case_and_punctuation_are_normalized_on_both_sides(self) -> None:
        result = score("hôm nay em đi học", "HÔM NAY, em đi học!")
        self.assertEqual(result.e, 0)

    def test_tone_marks_are_kept(self) -> None:
        self.assertEqual(score("hóa học", "hoa học").e, 1)

    def test_letter_names_are_not_joined(self) -> None:
        self.assertEqual(scoring.normalize("M T I"), "m t i")
        self.assertEqual(scoring.normalize("N.M.N"), "n m n")

    def test_spelled_keyword_costs_several_tokens(self) -> None:
        result = score("bổ sung và mti nhé", "bổ sung và em ti ai nhé", ("MTI",))
        self.assertEqual(result.e, 3)

    def test_edit_distance(self) -> None:
        self.assertEqual(scoring.edit_distance(["a", "b"], ["a", "b"]), 0)
        self.assertEqual(scoring.edit_distance(["a", "b"], ["a"]), 1)
        self.assertEqual(scoring.edit_distance(["a", "b"], ["a", "c"]), 1)
        self.assertEqual(scoring.edit_distance(["a", "b"], []), 2)


class KeywordTest(unittest.TestCase):
    def test_canonical_counts_case_insensitively(self) -> None:
        for output in ("và MTI nhé", "và mti nhé", "và Mti, nhé"):
            with self.subTest(output=output):
                result = score("và mti nhé", output, ("MTI",))
                self.assertEqual((result.a, result.l, result.tp, result.fa), (1, 1, 1, 0))

    def test_spelled_or_phonetic_forms_never_count(self) -> None:
        for output in ("và M T I nhé", "và mờ tê i nhé", "và M-T-I nhé", "và em ti ai nhé"):
            with self.subTest(output=output):
                result = score("và mti nhé", output, ("MTI",))
                self.assertEqual((result.a, result.tp), (0, 0))

    def test_only_whole_tokens_match(self) -> None:
        self.assertEqual(score("và mti nhé", "và mtis nhé", ("MTI",)).a, 0)

    def test_set_semantics_per_sample(self) -> None:
        result = score("mti và mti", "mti mti mti", ("MTI", "MTI"))
        self.assertEqual((result.a, result.l, result.tp), (1, 1, 1))

    def test_wrong_keyword_is_a_miss_and_a_false_alarm(self) -> None:
        result = score("xét nghiệm hdl", "xét nghiệm mri", ("HDL",))
        self.assertEqual((result.a, result.l, result.tp, result.fa), (1, 1, 0, 1))

    def test_distractor_in_output_is_a_false_alarm(self) -> None:
        result = score("xét nghiệm hdl", "xét nghiệm HDL và MRI", ("HDL",))
        self.assertEqual((result.tp, result.fa), (1, 1))

    def test_label_keyword_outside_vocab_is_ignored(self) -> None:
        self.assertEqual(score("mti và xyz", "mti và xyz", ("MTI", "XYZ")).l, 1)

    def test_multi_token_canonical_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scoring.keyword_vocab(["A B"])


class AggregateTest(unittest.TestCase):
    def test_micro_average_sums_before_dividing(self) -> None:
        agg = scoring.Aggregate.of([
            scoring.SampleScore(n=10, e=1, a=1, l=1, tp=1),
            scoring.SampleScore(n=90, e=9, a=3, l=1, tp=0),
        ])
        self.assertAlmostEqual(agg.wer, 0.10)
        self.assertAlmostEqual(agg.recall, 0.5)
        self.assertAlmostEqual(agg.precision, 0.25)
        self.assertAlmostEqual(agg.fa_per_1k, 3 / 100 * 1000)
        self.assertEqual(agg.fa, 3)

    def test_undefined_rates_are_none(self) -> None:
        agg = scoring.Aggregate.of([scoring.SampleScore(n=5, e=0, a=0, l=0, tp=0)])
        self.assertIsNone(agg.recall)
        self.assertIsNone(agg.precision)
        self.assertEqual(agg.fa_per_1k, 0.0)


class BoundaryTest(unittest.TestCase):
    def test_scoring_imports_nothing_from_the_project(self) -> None:
        imported = {value.__name__ for value in vars(scoring).values() if inspect.ismodule(value)}
        self.assertEqual(imported, {"unicodedata"})

    def test_score_has_no_condition_parameter(self) -> None:
        self.assertEqual(
            set(inspect.signature(scoring.score_record).parameters),
            {"record", "output", "vocab"},
        )


if __name__ == "__main__":
    unittest.main()

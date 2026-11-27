"""Scoring: WER, keyword Recall, keyword Precision and FA/1k (docs/METRIC.md).

Rules (docs/OVERVIEW.md, *Benchmark*):

1. **Canonical only.** A keyword counts as heard only when the normalized output
   contains its canonical form as a whole token (`ADH` → token `adh`). Spelled or
   phonetic forms (`a dê hát`, `A D H`) never count.
2. **No post-processing of the output.** `normalize()` only brings both sides onto
   the same written surface (NFC, lowercase, punctuation → space). It never joins
   letter names or maps pronunciations to letters.
3. **Set semantics per sample.** `A` = keywords of the vocabulary present in the
   output, `L` = canonicals listed in the record's `spoken_form`. A keyword repeated in one sentence counts
   once; extra repetitions are still penalised by WER.
4. **Micro average.** Raw counts are summed over samples first, then divided.

This module imports nothing from the project: the keyword vocabulary is passed in
by the caller and is the same for every model.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: FA is reported per this many reference tokens.
FA_SCALE = 1000
#: Pseudo-group holding every sample.
ALL = "all"


def normalize(text: str) -> str:
    """NFC → lowercase → every Unicode punctuation char becomes a space → collapse spaces.

    Applied to both the label and the output. Tone marks are kept (`hóa` ≠ `hoa`).
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return " ".join(text.split())


@dataclass(frozen=True)
class SampleScore:
    """Raw counts of one sample. `n` and `l` always come from the label."""

    n: int  # reference tokens
    e: int  # token edit distance, S + D + I
    a: int  # |A|, keywords heard in the output
    l: int  # |L|, keywords in the label
    tp: int  # |A ∩ L|

    @property
    def fa(self) -> int:
        """False alarms, |A \\ L|."""
        return self.a - self.tp


@dataclass(frozen=True)
class Aggregate:
    """Raw counts summed over samples (micro average)."""

    samples: int
    n: int
    e: int
    a: int
    l: int
    tp: int

    @classmethod
    def of(cls, items: Iterable) -> Aggregate:
        """Sum `SampleScore`s or per-sample row dicts with keys `n e a l tp`."""
        rows = [vars(item) if isinstance(item, SampleScore) else item for item in items]
        return cls(len(rows), *(sum(row[key] for row in rows) for key in ("n", "e", "a", "l", "tp")))

    @property
    def fa(self) -> int:
        return self.a - self.tp

    @property
    def wer(self) -> float | None:
        return self.e / self.n if self.n else None

    @property
    def recall(self) -> float | None:
        """`None` when the labels hold no keyword (Σ|L| = 0)."""
        return self.tp / self.l if self.l else None

    @property
    def precision(self) -> float | None:
        """`None` when the model printed no keyword at all (Σ|A| = 0)."""
        return self.tp / self.a if self.a else None

    @property
    def fa_per_1k(self) -> float | None:
        return self.fa / self.n * FA_SCALE if self.n else None


def by_group(rows: Iterable[dict]) -> dict[str, Aggregate]:
    """`{"all": ..., <data_class>: ...}` — the 4 columns of the report."""
    items = list(rows)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in items:
        groups[row["data_class"]].append(row)
    return {ALL: Aggregate.of(items), **{name: Aggregate.of(group) for name, group in sorted(groups.items())}}


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Token-level Levenshtein distance; equals `S + D + I`."""
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, 1):
        current = [row]
        for column, hyp_token in enumerate(hypothesis, 1):
            current.append(min(
                previous[column] + 1,  # deletion
                current[column - 1] + 1,  # insertion
                previous[column - 1] + (ref_token != hyp_token),  # substitution
            ))
        previous = current
    return previous[-1]


def keyword_vocab(canonicals: Iterable[str]) -> frozenset[str]:
    """Canonical forms → the set of normalized tokens used for matching.

    A canonical that does not normalize to exactly one token could never match.
    """
    vocab: set[str] = set()
    for canonical in canonicals:
        surface = normalize(str(canonical))
        if len(surface.split()) != 1:
            raise ValueError(f"canonical {canonical!r} normalizes to {surface!r}, not a single token")
        vocab.add(surface)
    return frozenset(vocab)


@dataclass(frozen=True)
class Scored:
    """A sample score plus the sets it was computed from (for per-sample audit)."""

    score: SampleScore
    heard: frozenset[str]
    label: frozenset[str]
    normalized_output: str


def score_record(record: dict, output: str, vocab: frozenset[str]) -> Scored:
    """Score one model output against one manifest record."""
    hyp = normalize(output)
    ref = normalize(record["ref"])
    label = {normalize(canonical) for canonical in record.get("spoken_form") or []} & vocab
    heard = {token for token in hyp.split() if token in vocab}
    return Scored(
        score=SampleScore(
            n=len(ref.split()),
            e=edit_distance(ref.split(), hyp.split()),
            a=len(heard),
            l=len(label),
            tp=len(heard & label),
        ),
        heard=frozenset(heard),
        label=frozenset(label),
        normalized_output=hyp,
    )

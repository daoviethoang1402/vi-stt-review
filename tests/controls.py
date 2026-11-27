"""Control adapters for the tests — fake models that check the harness, not models.

- `oracle`: returns the exact label → WER 0, Recall 100 %, Precision 100 %, FA 0.
- `dummy`: returns "" → WER 100 %, Σ|A| = 0 so Precision is undefined.
- `boostable`: prints each annotated keyword in canonical form only when it was
  granted as a hotword, and spelled out letter by letter (last letter dropped)
  otherwise → Recall 0 % at `off`, 100 % at `on`; spelled forms are never credited.

`register()` adds them to `adapters.FACTORIES` / `CONTRACTS` for the test run only.
"""

from __future__ import annotations

import adapters
from adapters.base import check_boost_level


class _Session:
    def __init__(self, render) -> None:
        self._render = render

    def transcribe(self, segment) -> str:
        return self._render(segment.id)

    def close(self) -> None:
        pass


class OracleAdapter:
    needs_ground_truth = True

    def load(self, config: dict) -> None:
        self.references = config["references"]

    def weights_mb(self) -> float:
        return 0.0

    def open_session(self, hotwords, boost_level):
        check_boost_level(boost_level)
        return _Session(lambda sample_id: self.references[sample_id])


class DummyAdapter(OracleAdapter):
    needs_ground_truth = False

    def load(self, config: dict) -> None:
        pass

    def open_session(self, hotwords, boost_level):
        return _Session(lambda sample_id: "")


class BoostableAdapter(OracleAdapter):
    def load(self, config: dict) -> None:
        super().load(config)
        self.acronyms = config["acronyms"]

    def open_session(self, hotwords, boost_level):
        boosted = {item.upper() for item in hotwords} if check_boost_level(boost_level) == "on" else set()

        def render(sample_id: str) -> str:
            tokens = self.references[sample_id].split()
            for canonical in {item.lower() for item in self.acronyms.get(sample_id) or []}:
                if canonical.upper() not in boosted:
                    spelled = " ".join(canonical[:-1]) if len(canonical) > 1 else canonical
                    tokens = [spelled if token == canonical else token for token in tokens]
            return " ".join(tokens)

        return _Session(render)


CONTROLS = {"oracle": OracleAdapter, "dummy": DummyAdapter, "boostable": BoostableAdapter}


def register() -> None:
    for name, factory in CONTROLS.items():
        adapters.FACTORIES[name] = factory
        adapters.CONTRACTS[name] = {"allowed": frozenset(), "required": frozenset(),
                                    "hotword_param": None, "needs_prompt": False}

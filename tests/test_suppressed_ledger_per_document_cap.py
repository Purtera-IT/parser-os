"""The suppression ledger is capped per document, never globally, and says
what it cut.

Live 010353: the ledger held 477 drops and the envelope carried the first 300.
The 177 cut were whatever the later stages dropped, which included 26 SOW
lines that therefore never appeared as Dropped cards in the labelling view.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core import orbitbrief_envelope as env


class _Atom:
    def __init__(self, text: str, artifact_id: str) -> None:
        self.raw_text = text
        self.artifact_id = artifact_id
        self.review_flags = ["suppressed:semantic_dedup"]


@pytest.fixture
def carrying_the_ledger(monkeypatch):
    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")


def test_a_noisy_document_does_not_starve_the_sow_of_its_drops(carrying_the_ledger):
    cap = env._SUPPRESSED_MAX
    # One mail thread drops more than the cap on its own, ahead of the SOW.
    noisy = [_Atom(f"footer {i}", "art_mail") for i in range(cap + 151)]
    sow = [_Atom(f"SOW line {i}", "art_sow") for i in range(26)]
    result = SimpleNamespace(suppressed_atoms=noisy + sow, project_id="d1")
    shown = env._suppressed_for_review(result, [])
    sow_shown = [r for r in shown if r["artifact_id"] == "art_sow"]
    assert len(sow_shown) == 26, "every dropped SOW line must reach the view"
    assert len([r for r in shown if r["artifact_id"] == "art_mail"]) == cap
    trunc = env._suppressed_truncated(result)
    assert trunc["count"] == 151
    assert trunc["by_artifact"] == {"art_mail": 151}
    assert trunc["per_document_cap"] == cap
    assert env._suppressed_total(result) == len(noisy) + len(sow)


def test_nothing_cut_reports_zero(carrying_the_ledger):
    result = SimpleNamespace(
        suppressed_atoms=[_Atom("a", "x"), _Atom("b", "y")], project_id="d1"
    )
    assert env._suppressed_truncated(result)["count"] == 0
    assert len(env._suppressed_for_review(result, [])) == 2

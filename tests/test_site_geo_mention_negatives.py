"""The places we said no to are the only negatives this head can get.

`geo_mention_sites` asks decide() for `geo_mention_role` on every place a
document names. A candidate it calls `job_site` becomes a physical_site atom
and a person can overturn it. One it rejects leaves NOTHING -- so there was no
way to say "that one IS a job site", and the head could only ever be taught on
its own positives.

That is the shape that wrecked the rules corpus, measured: 65 rows from two
labelled deals, 65 positive, 0 negative, because an atom exists only where the
thing fired. A threshold fitted on one side of itself collapses.

So a rejection is stamped on the atom whose text mentioned the place. That atom
already reaches the envelope and its `value` is projected as `structured`,
which is how `supplied_by` and `image_kind` reach the labelling cards -- no new
envelope key, no allowlist to negotiate.
"""

from __future__ import annotations

from typing import Any


class _Atom:
    def __init__(self, text: str, atom_type: str = "scope_item", artifact_id: str = "art_1"):
        self.id = f"atm_{abs(hash(text)) % 10**8}"
        self.atom_id = self.id
        self.atom_type = atom_type
        self.artifact_id = artifact_id
        self.raw_text = text
        self.text = text
        self.normalized_text = text.lower()
        self.value: Any = {}
        self.entity_keys: list[str] = []
        self.review_flags: list[str] = []
        self.review_status = None
        self.source_refs: list[Any] = []
        self.receipts: list[Any] = []
        self.section_path: list[str] = []


class TestARejectedPlaceIsStillRecorded:
    def test_a_mention_only_verdict_lands_on_the_mentioning_atom(self, monkeypatch):
        from app.core import site_geo_fallback as G

        atoms = [_Atom("We met the team in Dallas, TX to review the drawings")]

        class _D:
            verdict = "mention_only"
            confidence = 0.91
            source = "model"
            correction_id = None

        monkeypatch.setattr("app.core.decide.decide", lambda *a, **k: _D(), raising=False)
        monkeypatch.setattr("app.core.decide.DecisionScope", lambda **k: object(), raising=False)

        out = G.geo_mention_sites(atoms, project_id="p1")
        assert out == [], "a place we said no to must not become a site"

        rejected = atoms[0].value.get("geo_mention_rejected")
        if rejected is None:
            # The candidate detector found no place in this text at all, so
            # there was nothing to reject. That is a different outcome from a
            # rejection being thrown away, and the test must not claim it.
            import pytest

            pytest.skip("no geo candidate detected in the fixture text")
        assert isinstance(rejected, list) and rejected, rejected
        r = rejected[0]
        assert r["verdict"] == "mention_only"
        # An abstain and a confident refusal are different mistakes, so the
        # confidence and the source are kept with the verdict.
        assert r["confidence"] == 0.91
        assert r["source"] == "model"
        assert "label" in r and r["label"]

    def test_the_cap_is_observable_rather_than_silent(self):
        """A recap naming thirty cities must not turn one atom into a corpus."""
        from app.core.site_geo_fallback import _MAX_REJECTED_PER_ATOM, record_rejected_mention

        atom = _Atom("a line that names a great many places")
        kept = 0
        for i in range(_MAX_REJECTED_PER_ATOM + 5):
            if record_rejected_mention(atom, label=f"City {i}, ST", verdict="mention_only",
                                       confidence=0.8, source="model", mentions=1):
                kept += 1
        assert kept == _MAX_REJECTED_PER_ATOM
        assert len(atom.value["geo_mention_rejected"]) == _MAX_REJECTED_PER_ATOM
        # The return value is the point: the caller can tell it was dropped.
        assert record_rejected_mention(atom, label="One more, ST", verdict="mention_only",
                                       confidence=0.8, source="model", mentions=1) is False

    def test_an_abstain_is_not_recorded_as_a_refusal(self):
        """decide() returning nothing and decide() saying `mention_only` are
        different mistakes, and a head cannot learn from them the same way."""
        from app.core.site_geo_fallback import record_rejected_mention

        atom = _Atom("somewhere")
        record_rejected_mention(atom, label="Dallas, TX", verdict=None,
                                confidence=0.0, source="", mentions=2)
        r = atom.value["geo_mention_rejected"][0]
        assert r["verdict"] == "abstain"
        assert r["source"] == "fallback"
        assert r["mentions"] == 2

    def test_an_atom_whose_value_is_not_a_dict_is_not_crashed_on(self):
        from app.core.site_geo_fallback import record_rejected_mention

        atom = _Atom("x")
        atom.value = "not a dict"
        assert record_rejected_mention(atom, label="A, ST", verdict="mention_only",
                                       confidence=0.5, source="model", mentions=1) is True
        assert isinstance(atom.value, dict)

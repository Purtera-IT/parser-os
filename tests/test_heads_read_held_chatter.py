"""Lines an admission regex held aside are read by the heads; the regex's
small_talk is a hint that yields to a sure head reading (overrides.md point 4).
Made-up atoms; the 010358 demarc wording is the audit's own example."""
from __future__ import annotations

import app.core.c3_type_base as c3
from app.core.admission_chatter import mark_admission_chatter
from app.core.deal_chatter import retype_small_talk
from app.core.schemas import AtomType


class _A:
    def __init__(self, atom_id, text):
        self.id = atom_id
        self.raw_text = text
        self.atom_type = AtomType.deal_metadata
        self.value = {}
        self.review_flags = []
        self.source_refs = []
        self.artifact_id = "mail"
        self.project_id = "deal-test"


DEMARC = "Physically locates and documents the carrier demarc and associated equipment at each site."


def _held():
    return [mark_admission_chatter(_A("d", DEMARC), "signature"),
            mark_admission_chatter(_A("hi", "Hi Trent,"), "greeting"),
            mark_admission_chatter(_A("bye", "Thanks so much,"), "signoff")]


def test_the_heads_sure_type_stands_over_the_regex_small_talk(monkeypatch):
    monkeypatch.setattr(c3, "_DEAL", {"types": {"d": ["dependency", 0.83], "hi": ["small_talk", 0.99],
                                                 "bye": ["dependency", 0.31]}})
    monkeypatch.setenv("SOWSMITH_C3_SURE", "0.5")
    atoms = _held()
    assert retype_small_talk(atoms) == 2
    by = {a.id: a.atom_type.value for a in atoms}
    assert by == {"d": "deal_metadata", "hi": "small_talk", "bye": "small_talk"}  # unsure head: the hint stands


def test_without_a_reading_the_regex_hint_types_the_line(monkeypatch):
    monkeypatch.setattr(c3, "_DEAL", {})
    atoms = _held()
    assert retype_small_talk(atoms) == 3


def test_held_lines_are_in_the_deals_read_and_authority_types_them(monkeypatch):
    monkeypatch.setattr(c3, "_DEAL", {})
    c3.set_deal([])
    held = _held()
    kept = [_A("k", "Install four access points on the second floor.")]
    c3.also_read(held)
    _deal, rows, keys = c3._rows(kept, {})
    assert keys == ["k", "d", "hi", "bye"]
    monkeypatch.setenv("SOWSMITH_C3_AUTHORITY", "on")
    c3._DEAL["types"] = {"hi": ["small_talk", 0.99], "d": ["dependency", 0.95]}
    assert c3.apply_authority(held) == 2
    assert {a.id: a.atom_type.value for a in held} == {"d": "dependency", "hi": "small_talk", "bye": "deal_metadata"}

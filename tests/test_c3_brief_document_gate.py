"""Which documents feed the brief, and what decided it, visibly (audit 2026-10-28, wave 12 W5). Synthetic."""
from types import SimpleNamespace as NS

import pytest

from app.core import c3_brief, c3_type_base

TIMELINE = {"created_at": "2026-07-01T00:00:00Z", "transitions": [
    {"ts": "2026-07-01T00:00:00Z", "label": "Submitted for Quoting", "order": 1},
    {"ts": "2026-09-01T00:00:00Z", "label": "Closed Lost", "order": 4}]}


@pytest.fixture
def deal(monkeypatch):
    monkeypatch.setitem(c3_type_base._DEAL, "types", {})
    monkeypatch.setitem(c3_type_base._DEAL, "stage_timeline", TIMELINE)
    monkeypatch.setitem(c3_type_base._DEAL, "documents", [
        {"artifact_id": "art_req", "filename": "request.eml", "authored_at": "2026-07-02T00:00:00Z", "direction": "inbound"},
        {"artifact_id": "art_late", "filename": "late-call.json", "authored_at": "2026-09-10T00:00:00Z"},
        {"artifact_id": "art_other", "filename": "kiosk.eml", "authored_at": "2026-07-03T00:00:00Z", "direction": "inbound"}])
    monkeypatch.setitem(c3_type_base._DEAL, "document_job", [
        {"filenames": ["request.eml"], "verdict": "this_deal", "source": "c3", "c3_verdict": "this_deal",
         "c3_confidence": 0.97, "confidence": 0.97},
        {"filenames": ["kiosk.eml"], "verdict": "this_deal", "source": "c3", "c3_verdict": "other_job",
         "c3_confidence": 0.99, "confidence": 0.99, "share_rule_spared": True}])


def _atom(i, art, text):
    return NS(id=f"a{i}", atom_type="scope_item", raw_text=text, review_status="approved", artifact_id=art)


def test_a_document_after_the_decision_does_not_feed_the_brief_and_says_so(deal):
    atoms = [_atom(1, "art_req", "Install two displays."), _atom(2, "art_late", "Run cable on floor nine.")]
    reads = {"a1": {"col:admission": ["keep", 0.95]}, "a2": {"col:admission": ["keep", 0.95]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    body = [x["id"] for v in b["sections"].values() for x in v]
    assert body == ["a1"] and "a2" in b["left_out"]["ids"]
    docs = {d["filename"]: d for d in b["documents"]}
    assert docs["late-call.json"]["feeds"] is False
    assert docs["late-call.json"]["deal_stage"]["admissible_for"] == "neither"
    assert docs["request.eml"]["feeds"] is True
    assert docs["request.eml"]["document_job"]["by"] == "head:jdg:document_job"
    # a rule that kept a document the head called another job is named
    assert docs["kiosk.eml"]["document_job"]["kept_by_rule"] == ["share_rule_spared"]
    assert docs["kiosk.eml"]["document_job"]["head"] == ["other_job", 0.99]


def test_the_admissibility_gate_can_be_switched_off(deal, monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_ADMISSIBILITY_GATE", "0")
    atoms = [_atom(2, "art_late", "Run cable on floor nine.")]
    b = c3_brief.build(atoms, readings={"reads": {"a2": {"col:admission": ["keep", 0.95]}}, "links": [], "deal": {}})
    assert [x["id"] for v in b["sections"].values() for x in v] == ["a2"]

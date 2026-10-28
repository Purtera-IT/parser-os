"""The brief's keep decision: col:admission decides, co_action only drops as a company action, the type
head's sure small_talk is final (audit 2026-10-28, wave 3 #1 and wave 12 W6). Synthetic lines."""
from types import SimpleNamespace as NS

import pytest

from app.core import c3_brief, c3_type_base


def _atom(i, t, text="x"):
    return NS(id=f"a{i}", atom_type=t, raw_text=text, review_status="approved")


@pytest.fixture
def types(monkeypatch):
    t: dict = {}
    monkeypatch.setitem(c3_type_base._DEAL, "types", t)
    return t


def _ids(items):
    return [x["id"] for x in items]


def test_admission_decides_keep_and_co_action_only_drops_as_company_action(types):
    atoms = [_atom(1, "scope_item", "Mount four displays in the lobby."),
             _atom(2, "scope_item", "Run new cable to the server room."),
             _atom(3, "scope_item", "Thanks, talk soon."),
             _atom(4, "scope_item", "Customer will paint the backboard.")]
    reads = {
        "a1": {"col:admission": ["keep", 0.97], "read:co_action": ["keep", 0.4]},
        "a2": {"col:admission": ["keep", 0.62], "read:co_action": ["keep", 0.99]},
        "a3": {"col:admission": ["drop", 0.91], "read:co_action": ["keep", 0.96]},
        "a4": {"col:admission": ["keep", 0.95], "read:co_action": ["ignore", 0.8],
               "read:co_reason": ["not_our_scope", 0.7]},
    }
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    body = [x for v in b["sections"].values() for x in v]
    assert _ids(body) == ["a1"] and body[0]["by"] == "base:col:admission"
    assert _ids(b["also_mentioned"]) == ["a2"], "admission 0.62 is 'also mentioned', co_action 0.99 no longer lifts it"
    assert "a3" in b["left_out"]["ids"], "co_action keep does not re-keep a line admission dropped"
    assert "a4" in b["left_out"]["ids"], "co_action ignore stays a company action"
    assert b["decided_by"]["keep_head"] == "col:admission"
    assert b["decided_by"]["keep"]["admission:drop"] == 1
    assert b["decided_by"]["keep"]["head:company_ignore"] == 1


def test_sure_small_talk_is_final(types):
    atoms = [_atom(1, "scope_item", "You're on mute, buddy."), _atom(2, "small_talk", "the three weeks after the holiday")]
    types.update({"a1": ["small_talk", 0.99], "a2": ["small_talk", 0.6]})
    reads = {"a1": {"col:admission": ["keep", 0.95], "read:co_action": ["keep", 0.957]},
             "a2": {"col:admission": ["keep", 0.93], "read:co_action": ["keep", 0.64]}}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    assert "a1" in b["left_out"]["ids"]
    assert b["decided_by"]["keep"]["type_head:small_talk"] == 1
    kept = [x["id"] for v in b["sections"].values() for x in v] + _ids(b["also_mentioned"])
    assert "a2" in kept, "an unsure small_talk is kept when the keep head keeps it"


def test_without_admission_co_action_still_decides(types):
    atoms = [_atom(1, "scope_item", "Mount four displays.")]
    b = c3_brief.build(atoms, readings={"reads": {"a1": {"read:co_action": ["keep", 0.95]}}, "links": [], "deal": {}})
    assert b["decided_by"]["keep_head"] == "read:co_action"
    assert [x["by"] for v in b["sections"].values() for x in v] == ["purtera:read:co_action"]


def test_picture_readings_are_parser_derived_and_go_to_figures(types):
    pic = NS(id="a1", atom_type="exclusion", raw_text="The image shows a sensor bracket on a tank wall.",
             review_status="approved", value={"via": "pdf_image_vision"},
             source_refs=[NS(extraction_method="pdf_image_vision_v1", locator={"region_ref": "p3"})])
    line = _atom(2, "scope_item", "Install two sensors per tank.")
    reads = {"a1": {"col:admission": ["keep", 0.99], "read:co_action": ["keep", 0.99]},
             "a2": {"col:admission": ["keep", 0.97]}}
    b = c3_brief.build([pic, line], readings={"reads": reads, "links": [], "deal": {}})
    assert [x["id"] for v in b["sections"].values() for x in v] == ["a2"]
    assert b["left_out"]["by_reason"].get("picture_reading") == 1


def test_also_mentioned_overflow_is_counted_not_hidden(types, monkeypatch):
    monkeypatch.setattr(c3_brief, "ALSO_MENTIONED_SHOWN", 2)
    atoms = [_atom(i, "scope_item", f"line {i}") for i in range(5)]
    reads = {f"a{i}": {"col:admission": ["keep", 0.6 + i / 100]} for i in range(5)}
    b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
    assert _ids(b["also_mentioned"]) == ["a4", "a3"]
    assert b["also_mentioned_total"] == 5
    assert b["also_mentioned_more_ids"] == ["a2", "a1", "a0"]
    from app.core.c3_type_base import quote_snapshot

    snap = quote_snapshot(b)
    assert snap["also_mentioned_more_ids"] == ["a2", "a1", "a0"] and len(snap["also_mentioned"]) == 2

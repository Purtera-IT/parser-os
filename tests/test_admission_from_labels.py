"""Should this text have been an atom at all?

The one axis a labeler cannot teach by judging what is on screen. An atom
exists only where the parser admitted it, so every ordinary label is a positive
and the boundary has one side only. Measured 2026-09-27, before this existed:
65 rule decisions joined to labels across two deals gave 65 positives and 0
negatives, and a threshold fitted on that collapses to zero and fires on
everything.

Two rows fix it, and both come from things a labeler already does:

    origin=labeler   they highlighted text the parser made no atom of, so the
                     admission decision was WRONG to skip it   -> keep
    _keep            the parser made an atom and a person said it is not a
                     fact (boilerplate, a header, table scaffolding), so it was
                     wrong to admit it                          -> drop

`origin` was written on every label row since the schema was drawn and read by
nothing. `_keep` reached `atom_type` as a class and taught the type head about
a class that is really an admission decision.
"""
from __future__ import annotations

import json

from app.learning.human_labels import rows_for_deal

DEAL = "c79db726-323e-41f8-899d-1d8ca29a579a"


def _doc(labels):
    return {"deal_id": DEAL, "labels": labels, "links": [], "judgments": [], "answers": {}}


def _label(**kw):
    base = {
        "label_key": kw.get("label_key", "lbl_x"),
        "text": kw.get("text", "Two drops per workstation."),
        "label_type": kw.get("label_type", "quantity"),
        "labeler": "developer@purtera-it.com",
        "filename": "notes.eml",
        "hints": [],
        "hint_refs": [],
        "entity_keys": [],
        "note": None,
        "is_new_type": False,
        "labeled_at": "2026-09-27T09:00:00Z",
    }
    base.update(kw)
    return base


def _admission(rows):
    return [r for r in rows if r.get("relation") == "admission"]


def _prov(row):
    """Provenance ships as a JSON string on the row."""
    p = row.get("provenance")
    return json.loads(p) if isinstance(p, str) else (p or {})


def test_a_hand_added_atom_says_the_parser_should_have_admitted_it():
    rows = rows_for_deal(_doc([
        _label(label_key="lbl_a", origin="labeler",
               text="111 Great Neck Rd #314, Great Neck, NY 11021",
               label_type="physical_site"),
    ]))
    adm = _admission(rows)
    assert [r["label"] for r in adm] == ["keep"]
    assert "Great Neck" in adm[0]["raw_text"]
    assert _prov(adm[0])["parser_missed"] is True


def test_keep_says_the_parser_should_not_have():
    rows = rows_for_deal(_doc([
        _label(label_key="lbl_b", label_type="_keep", origin="parser",
               text="Bell Works | 101 Crawfords Corner Road"),
    ]))
    adm = _admission(rows)
    assert [r["label"] for r in adm] == ["drop"]
    assert _prov(adm[0])["parser_admitted_a_non_fact"] is True


def test_an_ordinary_label_teaches_admission_nothing():
    """A kept atom with a real type says the parser was right, but it says so
    about every atom on the screen. Emitting a positive for each would drown
    the few rows that carry a boundary."""
    rows = rows_for_deal(_doc([
        _label(label_key="lbl_c", label_type="quantity", origin="parser"),
    ]))
    assert _admission(rows) == []


def test_both_sides_reach_the_table_together():
    rows = rows_for_deal(_doc([
        _label(label_key="lbl_d", origin="labeler", text="Cabling to all rooms.",
               label_type="scope_item"),
        _label(label_key="lbl_e", label_type="_keep", origin="parser",
               text="Passcode: jz7o5CE9"),
        _label(label_key="lbl_f", label_type="quantity", origin="parser"),
    ]))
    assert sorted(r["label"] for r in _admission(rows)) == ["drop", "keep"]


def test_a_hand_added_keep_counts_as_a_miss_not_a_non_fact():
    """If somebody highlighted it by hand, they thought it was worth an atom.
    The admission verdict follows what they DID, not the type they settled on
    afterwards."""
    rows = rows_for_deal(_doc([
        _label(label_key="lbl_g", origin="labeler", label_type="_keep",
               text="Executive Summary"),
    ]))
    assert [r["label"] for r in _admission(rows)] == ["keep"]

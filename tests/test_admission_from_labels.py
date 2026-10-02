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


# --- small talk is an admission reject --------------------------------------
#
# "Hi Trent,", "Hope you had a great 4th of July!", "Thank you,". The registry
# says small_talk "carries no fact about the work"; the admission head must
# see it as `drop` or it never learns what a greeting looks like.

def test_small_talk_says_the_parser_should_not_have():
    rows = rows_for_deal(_doc([
        _label(label_key="lbl_h", label_type="small_talk", origin="parser",
               text="Hope you had a great 4th of July!"),
    ]))
    adm = _admission(rows)
    assert [r["label"] for r in adm] == ["drop"]
    assert _prov(adm[0])["rejected_as"] == "small_talk"
    # The type head still learns the class itself.
    assert any(r["relation"] == "atom_type" and r["label"] == "small_talk" for r in rows)


def test_hand_highlighted_small_talk_is_a_drop_not_a_miss():
    """Greetings and sign-offs are cut by a regex before they are atoms, so a
    labeler can only show one by highlighting it. That is a negative."""
    rows = rows_for_deal(_doc([
        _label(label_key="lbl_i", label_type="small_talk", origin="labeler",
               text="Hi Trent,"),
        _label(label_key="lbl_j", label_type="small_talk", origin="labeler",
               text="Thank you,"),
    ]))
    adm = _admission(rows)
    assert [r["label"] for r in adm] == ["drop", "drop"]
    assert all(not _prov(r).get("parser_missed") for r in adm)


def test_a_reject_flag_is_not_a_contrastive_type():
    """The labeling page writes rejected="true" on a reject. That is a flag,
    not the type the labeler ruled out, so it must not mint a `rejected` row
    whose class is the word "true"."""
    rows = rows_for_deal(_doc([
        _label(label_key="lbl_k", label_type="_keep", origin="parser",
               rejected="true", text="Stephanie Hechsel"),
        _label(label_key="lbl_l", label_type="small_talk", origin="parser",
               rejected=True, text="Thank you,"),
    ]))
    assert [r for r in rows if r["relation"] == "rejected"] == []
    assert sorted(r["label"] for r in _admission(rows)) == ["drop", "drop"]


def test_a_rejected_type_name_still_teaches_the_contrast():
    rows = rows_for_deal(_doc([
        _label(label_key="lbl_m", label_type="quantity", rejected="scope_item"),
    ]))
    assert [r["label"] for r in rows if r["relation"] == "rejected"] == ["scope_item"]


def test_admission_drop_rows_survive_the_training_blob_filter():
    """`is_placeholder_label` drops `_keep` as a TYPE label (right: the type
    head must not learn a placeholder). The admission row's label is `drop`,
    so the negative reaches the admission head."""
    from app.core.training_row_blob import is_placeholder_label, jsonl_to_rows

    rows = rows_for_deal(_doc([
        _label(label_key="lbl_n", label_type="_keep", origin="parser",
               text="Stephanie Hechsel"),
        _label(label_key="lbl_o", label_type="small_talk", origin="labeler",
               text="Hi Trent,"),
    ]))
    lines = []
    for r in rows:
        d = {k: r[k] for k in ("relation", "label", "raw_text", "masked_text", "label_kind",
                               "teacher", "weight", "deal_id", "split")}
        d["provenance"] = _prov(r)
        lines.append(json.dumps(d))
    back = jsonl_to_rows("\n".join(lines))
    assert sorted(r.label for r in back if r.relation == "admission") == ["drop", "drop"]
    assert not any(r.label == "_keep" for r in back)
    assert is_placeholder_label("_keep") and not is_placeholder_label("drop")

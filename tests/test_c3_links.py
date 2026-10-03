"""The evidence links the labeling page stores at the top of the training blob
(``doc.links``) reach ml/c3, on invented rows.

* links resolve by label_key, atom id or text (a highlighted Missed span);
* an ``answers`` link runs answer -> question whichever card drew it;
* a contradiction runs from the later line to the earlier one, whichever card drew it;
* the labeler's note is the teacher's WHY for that relation, with its
  [purtera] part going to the company pass;
* a hint_ref that names a line only by its text (or "<file tail>: <text>") points at that line;
* drafts written by a model, and links touching a row set aside, never train.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from ml.c3.brain import told_why  # noqa: E402
from ml.c3.data import DealExample, featurize  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402


def _atom(k, text, t):
    return {"label_key": k, "atom_id": f"at-{k}", "text": text, "doc_id": "email-1", "doc_kind": "email",
            "entered_at": f"2026-05-01T10:{t:02d}:00Z", "speaker_role": "customer", "speaker_side": "customer"}


ATOMS = [
    _atom("k1", "Do we know what kind of lock the door uses?", 0),
    _atom("k2", "Install window is Saturdays.", 1),
    _atom("k3", "They plan to use a maglock on that door.", 2),
    _atom("k4", "Correction: the install window is weekday evenings.", 3),
    _atom("k5", "Old kit row: 2 doors, 1 day.", 4),
]
LABELS = [
    {"label_key": "k1", "label_type": "open_question", "note": "Open question about the lock type."},
    {"label_key": "k2", "label_type": "schedule", "note": "First stated window; later replaced."},
    {"label_key": "k3", "label_type": "requirement", "note": "Names the lock type."},
    {"label_key": "k4", "label_type": "schedule", "note": "The window that governs now.",
     # Real rows point at text with no atom id; a heading and the line's own words point nowhere.
     "hint_refs": [{"hint": "other_doc", "kind": "text", "atomId": None, "text": "ved..txt: Install window is Saturdays."},
                   {"hint": "section", "kind": "heading", "atomId": None, "text": ATOMS[2]["text"]},
                   {"hint": "own_words", "kind": "text", "atomId": None, "text": ATOMS[3]["text"]}]},
    {"label_key": "k5", "label_type": "_keep", "note": "[EXCLUDE_FROM_TRAINING: old manual Deal Kit] Old row."},
    {"label_key": "k9", "origin": "labeler", "label_type": "requirement",
     "text": "Door frames are steel.", "entered_at": "2026-05-01T10:05:00Z", "doc_id": "email-1",
     "note": "Missed by the parser: frame material."},
]
PM = "pm@example.com"
LINKS = [
    # Drawn on the Questions card: question -> answer; must become answer -> question.
    {"from_head": "gap", "from_key": "gap:abc", "from_text": ATOMS[0]["text"], "to_atom_id": "at-k3",
     "relation": "answers", "note": "The maglock line closes the lock-type question.", "labeler": PM},
    # Drawn on the earlier line's card: must run later -> earlier.
    {"from_head": "atom", "from_key": "k2", "to_label_key": "k4", "relation": "contradicts",
     "note": "Weekday evenings replaces Saturdays; both cannot hold.\n[purtera] Our crews quote evenings at night rates.",
     "labeler": PM},
    # To a highlighted span (a Missed row), found by its text; the card's default note says nothing.
    {"from_head": "atom", "from_key": "k3", "to_kind": "text", "to_text": "Door frames are steel.",
     "relation": "context", "note": "drawn while labelling the whole deal", "labeler": PM},
    # Never trains: a draft, and a link touching an excluded row.
    {"from_head": "atom", "from_key": "k3", "to_label_key": "k4", "relation": "supports",
     "note": "Draft.", "labeler": "Claude (assistant)"},
    {"from_head": "atom", "from_key": "k5", "to_label_key": "k4", "relation": "supports",
     "note": "Old kit agrees.", "labeler": PM},
]


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def batch(schema):
    deal = DealExample.from_training_blob({"labels": LABELS, "links": LINKS}, ATOMS, deal_id="synthetic-links")
    return featurize(deal, schema)


def _i(batch, word):
    return next(i for i, t in enumerate(batch.texts) if word in t)


def test_blob_links_become_edges(batch):
    q, sat, mag, eve, steel = (_i(batch, w) for w in ("kind of lock", "Saturdays", "maglock", "evenings", "steel"))
    assert batch.edges["answers"] == [(mag, q)]
    assert batch.edges["contradicts"] == [(eve, sat)]
    assert batch.edges["context"] == [(mag, steel)]
    assert "supports" not in batch.edges or batch.edges["supports"] == []


def test_link_notes_teach_the_relation(schema, batch):
    mag, eve = _i(batch, "maglock"), _i(batch, "evenings")
    assert batch.field_notes[mag]["rel:answers"].startswith("The maglock line closes")
    assert "rel:context" not in batch.field_notes[mag]          # the card's default note
    note = batch.field_notes[eve]["rel:contradicts"]
    assert "both cannot hold" in note and "night rates" not in note
    assert "night rates" in batch.policy_note[eve]
    assert "night rates" not in (told_why(batch, schema)[eve] or "")


def test_text_only_hint_refs_point_at_their_line(batch):
    assert batch.hint_lines[_i(batch, "evenings")] == [_i(batch, "Saturdays")]

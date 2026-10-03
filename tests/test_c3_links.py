"""The evidence links the labeling page stores at the top of the training blob
(``doc.links``) reach ml/c3, on invented rows.

* links resolve by label_key, atom id or text (a highlighted Missed span);
* an ``answers`` link runs answer -> question whichever card drew it;
* a contradiction runs from the later line to the earlier one, whichever card drew it;
* the labeler's note is the teacher's WHY for that relation, with its
  [purtera] part going to the company pass;
* a hint_ref that names a line only by its text (or "<file tail>: <text>") points at that line;
* the parser's guess a person overruled is a known-wrong type;
* a heading pointer finds the heading line; a link to a whole document points
  at its lines; links from an older parse resolve by text or drop quietly;
  machine-draft markers are stripped;
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
    _atom("k6", "Customer responsibilities", 6),
    _atom("k7", "Review the site access notes before scheduling.", 7),
    {**_atom("k8", "SOW: provider installs the camera.", 1), "doc_id": "art-sow"},
    {**_atom("k10", "SOW: customer provides the lift.", 1), "doc_id": "art-sow"},
]
LABELS = [
    {"label_key": "k1", "label_type": "open_question", "note": "Open question about the lock type."},
    {"label_key": "k2", "label_type": "schedule", "note": "First stated window; later replaced."},
    {"label_key": "k3", "label_type": "requirement", "note": "Names the lock type."},
    {"label_key": "k4", "label_type": "schedule", "note": "The window that governs now.",
     # Real rows point at text with no atom id; a heading with no line and the line's own words point nowhere.
     "hint_refs": [{"hint": "other_doc", "kind": "text", "atomId": None, "text": "ved..txt: Install window is Saturdays."},
                   {"hint": "section", "kind": "heading", "atomId": None, "text": "Scheduling"},
                   {"hint": "own_words", "kind": "text", "atomId": None, "text": ATOMS[3]["text"]}]},
    {"label_key": "k5", "label_type": "_keep", "note": "[EXCLUDE_FROM_TRAINING: old manual Deal Kit] Old row."},
    {"label_key": "k6", "label_type": "list_header", "parser_type": "deal_metadata",
     "note": "Introduces the lines under it."},
    {"label_key": "k7", "label_type": "action_item", "parser_type": "action_item", "rejected": "requirement",
     "hint_refs": [{"hint": "section", "kind": "heading", "atomId": None, "text": "Kickoff email > Customer responsibilities"}],
     "note": "Accepted from claude-code (assistant)'s proposal: A to-do for us before scheduling.",
     "reads_set": {"sow_coverage": "missing_from_sow",
                   "sow_coverage_note": "From claude-code (assistant)'s proposal: The SOW never mentions it."}},
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
    # A link to a whole document: a pointer to every line of it.
    {"from_head": "atom", "from_key": "k7", "to_kind": "text", "to_artifact_id": "art-sow",
     "relation": "context", "note": "[points_at_artifact] the file this line announces", "labeler": PM},
    # Drawn on an earlier parse: stale artifact, no ids; still found by its text.
    {"from_head": "atom", "from_key": "k7", "to_kind": "text", "to_artifact_id": "art-old-parse",
     "to_text": "Install window is Saturdays.", "relation": "supports", "note": "Same window.", "labeler": PM},
    # Keys from an older parse that this one no longer has: dropped quietly.
    {"from_head": "atom", "from_key": "lbl_old", "to_label_key": "lbl_gone", "relation": "supports", "labeler": PM},
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
    assert batch.edges["supports"] == [(_i(batch, "access notes"), sat)]       # the stale-parse link only


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


def test_parser_guess_and_old_ruled_out_types_are_known_wrong(schema, batch):
    t = schema.by_key()["col:label_type"]
    neg = batch.negatives["col:label_type"]
    assert neg[_i(batch, "Customer responsibilities")] == [t.index("deal_metadata")]
    # The parser was right here ("Parser's right"); the old ruled-out type still counts.
    assert neg[_i(batch, "access notes")] == [t.index("requirement")]
    assert neg[_i(batch, "kind of lock")] == []


def test_heading_pointers_document_pointers_and_draft_markers(batch):
    i = _i(batch, "access notes")
    assert batch.hint_lines[i] == sorted([_i(batch, "Customer responsibilities"),
                                          _i(batch, "installs the camera"), _i(batch, "provides the lift")])
    assert batch.why[i] == "A to-do for us before scheduling."
    assert batch.field_notes[i]["read:sow_coverage"] == "The SOW never mentions it."

"""ml/c3: a judgment's note splits like an atom's (invented rows only).

The universal WHY goes to the teacher; a ``[purtera]`` line goes to the
company layer; the page's accepted-proposal opener goes nowhere; a reason
code is the verdict's class, not part of the WHY, while a sentence typed in
the reason field is kept.
"""
from __future__ import annotations

from ml.c3.data import DealExample, featurize
from ml.c3.notes import judgment_note
from ml.c3.schema import load_schema

ATOMS = [
    {"label_key": f"k{n}", "atom_id": f"at-k{n}", "text": t, "doc_id": "email-1", "doc_kind": "email",
     "section": "body", "entered_at": f"2026-05-01T10:0{n}:00Z", "speaker_role": "customer",
     "speaker_side": "customer"}
    for n, t in enumerate(["Send the invoice to our head office on Main Avenue.",
                           "Work window is Saturday only.",
                           "Work window is weekdays after 6pm.",
                           "Who is the on-site contact?"])
]
WHY = "The billing address is not where the work happens."
POLICY = "we never schedule a crew to a billing address."
PERSON = "pm@example.com"


def _batch(*judgments):
    deal = DealExample.from_training_blob({"labels": [], "judgments": list(judgments)}, ATOMS,
                                          deal_id="synthetic-judgment-notes")
    return featurize(deal, load_schema())


def test_judgment_note_parts():
    assert judgment_note("wrong_shape", f"{WHY}\n[purtera] {POLICY}") == (WHY, POLICY)
    assert judgment_note("", f"Accepted in bulk from claude-code (assistant)'s proposal: {WHY}") == (WHY, "")
    assert judgment_note("", f"Accepted from Some Reviewer's proposal: [purtera] {POLICY}") == ("", POLICY)
    assert judgment_note("", f"{WHY}\n[parser] SHOULD SPLIT: two lines.") == (WHY, "")
    # A sentence in the reason field is the labeler's WHY; a code is not.
    sentence = "Asked already in the first email."
    assert judgment_note(sentence, WHY) == (f"{sentence}\n{WHY}", "")
    assert judgment_note("kept", "") == ("", "")


def test_a_line_verdict_sends_its_company_line_to_the_company_layer():
    b = _batch({"head": "site_role", "verdict": "vendor_or_billing_address", "labeler": PERSON,
                "target": {"site": {"evidence": {"atomId": "at-k0"}}}, "reason": "answered",
                "note": f"Accepted from claude-code (assistant)'s proposal: {WHY}\n[purtera] {POLICY}"})
    notes = b.field_notes[0]["jdg:site_role"]
    assert "billing address is not where" in notes
    for gone in ("[purtera]", "never schedule", "proposal", "answered"):
        assert gone not in notes, gone
    assert POLICY in b.policy_note[0]


def test_a_pair_verdict_sends_its_company_line_to_both_lines():
    b = _batch({"head": "conflict", "verdict": "contradicts", "labeler": PERSON,
                "target": {"a": {"atomId": "at-k1"}, "b": {"atomId": "at-k2"}}, "reason": "different_window",
                "note": f"Saturday only and weekdays after 6pm cannot both hold.\n[purtera] {POLICY}"})
    (j,) = [j for j in b.judged if j.key == "jdg:conflict"]
    assert "[purtera]" not in j.note and "never schedule" not in j.note
    assert not j.note.startswith("different_window")
    assert j.policy == POLICY
    assert POLICY in b.policy_note[1] and POLICY in b.policy_note[2]

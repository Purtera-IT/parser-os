"""An exclusion marker behind an accepted-proposal opener still excludes.

A draft accepted on the labelling page is saved with the page's provenance
first ("Accepted [in bulk] from <x>'s proposal: ..."), which pushes an
EXCLUDE_FROM_TRAINING marker off the start of the note. Invented rows only.
"""
from __future__ import annotations

from app.learning.human_labels import IngestReport, rows_for_deal

PERSON = "labeler@example.com"

NOTES = [
    "Accepted in bulk from claude-code (assistant)'s proposal: EXCLUDE_FROM_TRAINING: old manual Deal Kit. "
    "A hand-built line.",
    "Accepted from Some Reviewer's proposal: [EXCLUDE_FROM_TRAINING: old manual Deal Kit] A hand-built line.",
    "accepted from a.person@example.com’s proposal:\nEXCLUDE_FROM_TRAINING: old manual Deal Kit.",
]


def _label(key, text, **kw):
    return {"label_key": key, "atom_id": f"a_{key}", "labeler": PERSON,
            "label_type": "commercial_term", "decide_text": text, "text": text, **kw}


KEEP = _label("lbl_keep", "Install 12 displays in the lobby")


def _texts(rows):
    return " ".join(str(r.get("raw_text") or "") + " " + str(r.get("masked_text") or "") for r in rows)


def test_a_label_link_or_judgment_marked_behind_the_opener_is_excluded():
    for note in NOTES:
        gone = _label("lbl_gone", "Old kit line, marked behind the opener", note=note)
        link = {"labeler": PERSON, "relation": "context", "from_key": "lbl_keep", "to_key": "lbl_x",
                "from_text": "Install 12 displays in the lobby", "to_text": "A linked line, marked", "note": note}
        judgment = {"labeler": PERSON, "head": "gap", "verdict": "valid", "target_key": "gap:abc",
                    "text": "A judged question, marked", "note": note}
        report = IngestReport()
        rows = rows_for_deal({"deal_id": "999002", "labels": [KEEP, gone], "links": [link],
                              "judgments": [judgment]}, report)
        text = _texts(rows)
        assert "Install 12 displays" in text, note
        for marked in ("Old kit line", "A linked line", "A judged question"):
            assert marked not in text, (note, marked)
        assert report.skipped["excluded from training"] == 1


def test_an_opener_without_the_marker_still_trains():
    note = "Accepted in bulk from claude-code (assistant)'s proposal: a real argument about the line."
    lb = _label("lbl_real", "A real line accepted from a draft", note=note)
    assert "A real line accepted" in _texts(rows_for_deal({"deal_id": "999002", "labels": [lb]}))
    # The marker later in the note is the labeler's words, not a marker.
    lb = _label("lbl_mid", "A line that mentions the marker", note="Kept; not EXCLUDE_FROM_TRAINING.")
    assert "mentions the marker" in _texts(rows_for_deal({"deal_id": "999002", "labels": [lb]}))


def test_c3_excluded_reads_past_the_opener():
    from ml.c3.data import excluded

    for note in NOTES:
        assert excluded({"note": note}), note
    assert not excluded({"note": "Accepted in bulk from claude-code (assistant)'s proposal: a real argument."})
    assert not excluded({"note": "Accepted in bulk from claude-code (assistant)'s proposal"})

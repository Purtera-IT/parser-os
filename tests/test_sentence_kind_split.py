"""A paragraph of mixed kinds splits by kind; banter is a chatter atom.

The user's case: one email paragraph holding a cheer, a dependency (the PO
gates scheduling) and a housekeeping note. As one atom the cheer buried the
fact. Same-kind paragraphs stay whole.
"""
from __future__ import annotations

from pathlib import Path

from app.core.sentences import sentence_kind, split_by_kind
from app.parsers.email_parser import ADMISSION_REJECT_FLAG, EmailParser

PARA = (
    "Woohoo! Let's go Sarah! Famous words of D Khaled...Another one! We are all good over here. "
    "Just need PO from you/customer and we can start scheduling and getting the ball rolling on install. "
    "Also, just adding the opportunity number on subject line for tracking purposes."
)
BANTER = "Woohoo! Let's go Sarah! Famous words of D Khaled...Another one! We are all good over here."
PO = "Just need PO from you/customer and we can start scheduling and getting the ball rolling on install."
HOUSEKEEPING = "Also, just adding the opportunity number on subject line for tracking purposes."


def test_split_by_kind_on_the_user_paragraph():
    assert split_by_kind(PARA) == [BANTER, PO, HOUSEKEEPING]
    assert sentence_kind(BANTER) == "banter"
    assert sentence_kind(PO) == "work"


def test_same_kind_paragraph_stays_whole():
    assert split_by_kind("Please quote two racks. Remove the West Wing from scope.") == []
    assert split_by_kind("Thanks so much! Have a great weekend!") == []
    # An ellipsis inside a sentence is not a break.
    assert split_by_kind("Famous words of D Khaled...Another one!") == []


def test_kind_is_conservative():
    for work in ("Approved!", "Let's go with option B!", "We'll ship Monday!",
                 "Need 40 drops by Friday!", "Can we start Monday?"):
        assert sentence_kind(work) == "work", work
    # "well" is not "we'll": a pleasantry stays a pleasantry.
    assert sentence_kind("Hope you are well!") == "banter"


def test_email_paragraph_becomes_banter_chatter_plus_its_own_fact(tmp_path: Path):
    p = tmp_path / "po.eml"
    p.write_text(
        "From: Bob Smith <bob@acme.com>\nTo: Sarah <sarah@purtera-it.com>\nSubject: PO\n"
        "Date: Mon, 07 Jul 2026 09:00:00 -0400\nContent-Type: text/plain; charset=utf-8\n\n"
        f"Hi Sarah,\n\n{PARA}\n\nThanks,\nBob Smith\n",
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=p).atoms
    by_text = {a.raw_text: a for a in atoms}
    banter = by_text[BANTER]
    assert ADMISSION_REJECT_FLAG in banter.review_flags and "chatter" in banter.review_flags
    assert banter.value["admission_regex"] == "banter"
    po = by_text[PO]
    assert "chatter" not in (po.review_flags or [])
    assert HOUSEKEEPING in by_text
    assert PARA not in by_text

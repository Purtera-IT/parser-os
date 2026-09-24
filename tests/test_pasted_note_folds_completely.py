"""A note that is a copy of an email must not outlive it.

010288's "The Ask w diagram link" note is a paste of Alec's email. Everything
in it folded away except two lines: the drawing link, and the CRM's own header
saying who pasted it. The note stayed in the deal on the strength of those two
-- and 22 readings of the job's wiring diagram attached to that copy instead
of to the message that delivered it.
"""
from __future__ import annotations

from app.core.pasted_note_dedup import collapse_pasted_note_duplicates

PNG = "https://huzzard.com/wp-content/uploads/2025/11/BPW061725-Rev-1.png"
SAFELINK = (
    "https://nam13.safelinks.protection.outlook.com/?url="
    "https%3A%2F%2Fhuzzard.com%2Fwp-content%2Fuploads%2F2025%2F11%2F"
    "BPW061725-Rev-1.png&data=05%7C02%7Caj%40x.com%7C5dd"
)


class Atom:
    def __init__(self, artifact_id, text, kind=None, flags=None):
        self.artifact_id = artifact_id
        self.raw_text = text
        self.normalized_text = text
        self.atom_type = "deal_metadata"
        self.value = {"kind": kind} if kind else {}
        self.review_flags = list(flags or [])
        self.source_refs = []


def email(text, kind="email_body"):
    a = Atom("art_email", text, kind=kind)
    a.value["message_index"] = 0
    return a


def note(text, kind="hubspot_note_body"):
    return Atom("art_note", text, kind=kind, flags=["hubspot_note_parser"])


BODY = [
    "Here are the details for the small job I was discussing earlier.",
    "The club/installer will need to source anything beyond the Relay.",
    "Additionally I would note that the connectivity kit limits our power to 40-50ft.",
]


def test_a_note_that_copies_an_email_folds_away_completely():
    """The note's own header is not a paste of anything, so it must not sit in
    the denominator keeping the note alive. Before this, a fully duplicated
    note scored 1/2 against the 0.8 bar and survived."""
    atoms = [email(t) for t in BODY]
    atoms += [note(t) for t in BODY]
    atoms.append(note("note_id=116539976562 | author=AJ Evans | author_affiliation=internal",
                      kind="hubspot_note_meta"))

    kept, dropped = collapse_pasted_note_duplicates(atoms)
    survivors = {a.artifact_id for a in kept if a.value.get("kind") != "hubspot_note_meta"}
    assert survivors == {"art_email"}, [a.raw_text[:40] for a in kept]
    assert len(dropped) == len(BODY)


def test_the_same_picture_through_a_safelink_is_the_same_picture():
    """The email carries the bare vendor URL; the note carries it wrapped in
    700 characters of Outlook and Proofpoint. By text they are two unrelated
    strings; by destination they are one fact."""
    atoms = [email(t) for t in BODY]
    atoms.append(email(f"Diagram: {PNG}"))
    atoms += [note(t) for t in BODY]
    atoms.append(note(f"Diagram: {SAFELINK}", kind="note_field_image"))
    atoms.append(note("note_id=116539976562 | author=AJ Evans", kind="hubspot_note_meta"))

    kept, dropped = collapse_pasted_note_duplicates(atoms)
    note_left = [a for a in kept
                 if a.artifact_id == "art_note" and a.value.get("kind") != "hubspot_note_meta"]
    assert note_left == [], [a.raw_text[:60] for a in note_left]
    # The email's own copy of the drawing survives and records the paste.
    surviving = [a for a in kept if a.raw_text.startswith("Diagram:")]
    assert len(surviving) == 1
    assert surviving[0].artifact_id == "art_email"
    assert "art_note" in (surviving[0].value.get("also_in_note") or [])


def test_a_note_that_says_something_new_is_still_kept():
    """The guard against over-folding: a note carrying a fact the email never
    had is not a copy, and folding it would lose the only record of it."""
    atoms = [email(t) for t in BODY]
    atoms += [note(t) for t in BODY]
    atoms.append(note("Client confirmed the front door is the only access point in scope."))
    atoms.append(note("They have asked us to hold the quote until the new year, per Friday's call."))
    atoms.append(note("The budget for this phase has not been approved by the owner yet."))

    kept, _ = collapse_pasted_note_duplicates(atoms)
    left = [a.raw_text for a in kept if a.artifact_id == "art_note"]
    assert any("front door is the only access point" in t for t in left)


def test_scaffolding_alone_never_folds_a_document():
    """A note that is ONLY its own header has nothing to compare, and must not
    be declared a copy of anything."""
    atoms = [email(t) for t in BODY]
    atoms.append(note("note_id=1 | author=AJ Evans", kind="hubspot_note_meta"))
    kept, dropped = collapse_pasted_note_duplicates(atoms)
    assert dropped == []
    assert any(a.artifact_id == "art_note" for a in kept)

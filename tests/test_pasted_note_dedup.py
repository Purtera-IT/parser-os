"""A HubSpot note that is a pasted email is one message, not two sources.
Live 010289: 6 of 49 atoms were the same sentences twice."""
from __future__ import annotations

from app.core.pasted_note_dedup import collapse_pasted_note_duplicates
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _atom(text: str, artifact: str, value: dict, flags: list[str] | None = None):
    return EvidenceAtom(
        id=f"atm_{artifact}_{abs(hash(text)) % 10**6}", project_id="p", artifact_id=artifact,
        atom_type=AtomType.scope_item, raw_text=text, normalized_text=text.lower(), value=value,
        entity_keys=[],
        source_refs=[SourceRef(id=f"src_{artifact}", artifact_id=artifact, artifact_type=ArtifactType.txt,
                               filename=f"{artifact}.txt", locator={}, extraction_method="test", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.7, review_flags=flags or [], review_status=ReviewStatus.auto_accepted, parser_version="test",
    )


LINES = [
    "The club/installer will need to source anything beyond the Relay.",
    "Here are the details for the small job I was discussing earlier.",
    "If this is a successful implementation, it could lead to many more.",
    "Note the diagram is labeled by the vendor and is not accurate.",
    "Cat5e/6 (6 or better recommended)",
]


def _thread(lines):
    return [_atom(t, "art_mail", {"kind": "email_body_line", "message_index": 0}) for t in lines]


def _paste(lines):
    return [_atom(t, "art_note", {"kind": "hubspot_note_body"}, ["hubspot_note_parser"]) for t in lines]


def test_a_note_that_is_a_copy_folds_whole():
    # Copy-of is a fact about a DOCUMENT. All of it folds, or none of it --
    # a half-folded note is what left three of Alec's sentences alone in a
    # file of their own, attributed to the man who pasted them.
    mail, note = _thread(LINES), _paste(LINES)
    kept, dropped = collapse_pasted_note_duplicates(mail + note)
    assert len(dropped) == len(LINES), "every line of the copy, not some of them"
    assert all(a.artifact_id == "art_note" for a in dropped)
    assert kept == mail
    # the paste is not lost: the mail records where else it appeared
    assert mail[0].value["also_in_note"] == ["art_note"]
    assert any(r.artifact_id == "art_note" for r in mail[0].source_refs)


def test_a_line_the_copy_ADDED_stays_and_says_it_was_added():
    # Whoever typed it said it, at the note's time. It must not be folded
    # into the original's content and must not silently disappear.
    mail = _thread(LINES)
    note = _paste(LINES + ["AJ: flagged to Trent, he knows the CA installer."])
    kept, dropped = collapse_pasted_note_duplicates(mail + note)
    extra = [a for a in kept if a.artifact_id == "art_note"]
    assert len(extra) == 1
    assert extra[0].value["added_when_filed"] is True
    assert extra[0].value["copy_of_document"] == "art_mail"
    assert len(dropped) == len(LINES)


def test_a_document_that_only_half_matches_is_reported_not_guessed():
    # Neither one document nor two. Splitting it is the disorganised outcome:
    # some atoms from one file, some from the other, decided by nothing.
    mail = _thread(LINES)
    note = _paste(LINES[:2] + ["Unrelated line one here.", "Unrelated line two here.",
                               "Unrelated line three here."])
    kept, dropped = collapse_pasted_note_duplicates(mail + note)
    assert dropped == [], "nothing folds while it is ambiguous"
    flagged = [a for a in kept if a.value.get("maybe_copy_of")]
    assert len(flagged) == len(note)
    assert flagged[0].value["maybe_copy_of"]["document"] == "art_mail"
    assert 0.3 < flagged[0].value["maybe_copy_of"]["share"] < 0.6


def test_the_original_is_ranked_never_chosen_by_similarity():
    # An email has a sender, a timestamp and a thread. A note has whoever
    # pasted it. The mail survives whichever order they arrive in.
    mail, note = _thread(LINES), _paste(LINES)
    for order in (mail + note, note + mail):
        for a in order:
            a.value.pop("also_in_note", None)
        kept, dropped = collapse_pasted_note_duplicates(list(order))
        assert all(a.artifact_id == "art_mail" for a in kept)
        assert all(a.artifact_id == "art_note" for a in dropped)


def test_two_emails_saying_the_same_thing_are_still_two_sources():
    line = "The club provides the mag lock."
    a = _atom(line, "art_m1", {"kind": "email_body_line", "message_index": 0})
    b = _atom(line, "art_m2", {"kind": "email_body_line", "message_index": 0})
    kept, dropped = collapse_pasted_note_duplicates([a, b])
    assert not dropped and len(kept) == 2


def test_a_note_that_glues_its_title_onto_the_mail_is_still_the_same_line():
    """Live 010288: the note pasted the ask under its own title, so the line
    read "The Ask w diagram link Hey AJ, Here are the details…" -- the same
    sentence with a prefix, and it showed as a second card."""
    line = "Here are the details for the small job I was discussing earlier."
    mail = _thread(LINES)
    note = _paste([f"The Ask w diagram link Hey AJ, {LINES[1]}"] + LINES[2:])
    kept, dropped = collapse_pasted_note_duplicates(mail + note)
    assert len(dropped) == len(note), "the glued title does not make it a different sentence"
    assert all(a.artifact_id == "art_mail" for a in kept)


def test_a_short_line_inside_a_longer_one_is_not_collapsed():
    # "Relay" appears inside plenty of sentences; only a substantial line is
    # safe to treat as the same statement.
    mail = _atom("Relay", "art_mail", {"kind": "email_body_line", "message_index": 0})
    note = _atom("The club supplies the Relay and the lock", "art_note", {"kind": "hubspot_note_body"},
                 ["hubspot_note_parser"])
    kept, dropped = collapse_pasted_note_duplicates([mail, note])
    assert dropped == [] and len(kept) == 2

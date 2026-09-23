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


def test_the_note_copy_folds_onto_the_email_it_was_pasted_from():
    line = "The club/installer will need to source anything beyond the Relay."
    mail = _atom(line, "art_mail", {"kind": "email_body_line", "message_index": 0})
    note = _atom(line, "art_note", {"kind": "hubspot_note_body"}, ["hubspot_note_parser"])
    other = _atom("Cat5e/6 (6 or better recommended)", "art_note", {"kind": "note_field_item"}, ["hubspot_note_parser"])
    kept, dropped = collapse_pasted_note_duplicates([mail, note, other])
    assert [a.artifact_id for a in dropped] == ["art_note"]
    assert mail in kept and other in kept
    # the paste is not lost: the mail records where else it appeared
    assert mail.value["also_in_note"] == ["art_note"]
    assert any(r.artifact_id == "art_note" for r in mail.source_refs)


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
    mail = _atom(line, "art_mail", {"kind": "email_body_line", "message_index": 0})
    note = _atom(f"The Ask w diagram link Hey AJ, {line}", "art_note", {"kind": "hubspot_note_body"},
                 ["hubspot_note_parser"])
    kept, dropped = collapse_pasted_note_duplicates([mail, note])
    assert dropped == [note] and kept == [mail]
    assert mail.value["also_in_note"] == ["art_note"]


def test_a_short_line_inside_a_longer_one_is_not_collapsed():
    # "Relay" appears inside plenty of sentences; only a substantial line is
    # safe to treat as the same statement.
    mail = _atom("Relay", "art_mail", {"kind": "email_body_line", "message_index": 0})
    note = _atom("The club supplies the Relay and the lock", "art_note", {"kind": "hubspot_note_body"},
                 ["hubspot_note_parser"])
    kept, dropped = collapse_pasted_note_duplicates([mail, note])
    assert dropped == [] and len(kept) == 2

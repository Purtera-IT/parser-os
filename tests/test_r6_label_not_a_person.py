"""A field label in a rate table or a clause is not a person (010353, 010003).

A SOW's premium-rate tiers read "Business Hours: 8:00 AM to 5:00 PM (17:00)
local time at the Stated Rate." The "PM" of the clock time passed for a
Project Manager role cue, so every capitalised phrase beside it became a
stakeholder ("Business Hours", "After Hours", "Stated Rate"), and "The Buyer
and Provider Contact Persons shall ... approve" made "The Buyer" one. A
determiner-led phrase, or a "Label:" a figure follows, names no one, and a
clock time is not a role. Real names in the same document stay people.
"""
from __future__ import annotations

from app.core.entity_extraction import _emit_stakeholders, enrich_atoms
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.domain import load_domain_pack

_PACK = load_domain_pack("default")

TIERS = [
    "o Weekday Shift: 7:00 AM to 4:00 PM (16:00) local time at the Posted Rate. "
    "o Evening Shift: 4:00 PM to 7:00 AM Monday through Friday: 30% premium over Posted Rate.",
    "Evening Shift: 4:00 PM (16:00) to 7:00 AM: 30% premium over Posted Rate.",
    "The Purchaser and Vendor Contact Persons shall be authorized to approve changes.",
]
PEOPLE = [
    "FULL NAME: Dana Whitfield | JOB TITLE: Director of Operations | EMAIL ADDRESS: dana@example-provider.com",
    "Make the aiming adjustments requested by Project Manager Riley Okafor during the onsite window.",
]


def _atom(i: int, text: str) -> EvidenceAtom:
    return EvidenceAtom(
        id=f"atm_{i}", project_id="p", artifact_id="doc", atom_type=AtomType.assumption,
        raw_text=text, normalized_text=text.lower(), value={},
        authority_class=AuthorityClass.contractual_scope, confidence=0.8,
        review_status=ReviewStatus.auto_accepted, entity_keys=[],
        source_refs=[SourceRef(id=f"src_{i}", artifact_id="doc", artifact_type=ArtifactType.pdf,
                               filename="sow.pdf", locator={"page": 0}, extraction_method="test",
                               parser_version="test")],
        parser_version="test",
    )


def test_rate_labels_and_a_determiner_phrase_are_not_people() -> None:
    atoms = [_atom(i, t) for i, t in enumerate(TIERS + PEOPLE)]
    n = len(atoms)
    enrich_atoms(atoms, _PACK)
    derived = {(a.value or {}).get("name") for a in atoms[n:]
               if str(getattr(a.atom_type, "value", a.atom_type)) == "stakeholder"}
    for label in ("Weekday Shift", "Evening Shift", "Posted Rate", "The Purchaser"):
        assert label not in derived, derived
    keys = {k for a in atoms[:n] for k in (a.entity_keys or [])}
    assert not {"stakeholder:weekday_shift", "stakeholder:evening_shift", "stakeholder:posted_rate",
                "stakeholder:the_purchaser"} & keys, keys
    assert "Dana Whitfield" in derived, derived
    assert "stakeholder:riley_okafor" in keys, keys


def test_a_name_before_a_meeting_time_is_still_a_person() -> None:
    keys = _emit_stakeholders("Riley Okafor approved the change at the 3:00 PM review.")
    assert "stakeholder:riley_okafor" in keys

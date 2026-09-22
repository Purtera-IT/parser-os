"""Who said it, to whom, for which company. Without it a PM reading "we
provide the parts" cannot tell whether "we" is us or the reseller."""
from __future__ import annotations

from app.core.deal_parties import parties_for_message, party, stamp_parties
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _atom(value):
    return EvidenceAtom(
        id="atm_1", project_id="p", artifact_id="art_m", atom_type=AtomType.scope_item,
        raw_text="We provide the parts that connect the PC to the relay.",
        normalized_text="we provide the parts", value=value, entity_keys=[],
        source_refs=[SourceRef(id="s1", artifact_id="art_m", artifact_type=ArtifactType.email, filename="m.eml",
                               locator={}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t",
    )


def test_a_person_resolves_to_a_company_and_a_side():
    them = party("Alec Burns <alecbur@cdw.com>")
    assert them == {"email": "alecbur@cdw.com", "name": "Alec Burns", "company": "cdw.com",
                    "side": "theirs", "role_guess": "reseller"}
    us = party('"AJ Evans" <aj@purtera-it.com>')
    assert us["side"] == "ours" and us["role_guess"] == "internal"
    # a company we have never met gets no invented role
    assert party("albert@rd-systems.com")["role_guess"] == "unknown"
    assert party("not an address") is None


def test_said_to_separates_a_promise_from_a_thought():
    outward = parties_for_message({"sender": "aj@purtera-it.com", "to": ["Alec <alecbur@cdw.com>"]})
    assert outward["said_to"][0]["company"] == "cdw.com"
    assert outward["internal_only"] is False
    internal = parties_for_message({"sender": "aj@purtera-it.com", "to": ["chase@purtera-it.com"], "cc": []})
    assert internal["internal_only"] is True


def test_every_email_atom_is_stamped_once():
    a = _atom({"kind": "email_body_line",
               "email_thread": {"thread_id": "T1", "sender": "Alec <alecbur@cdw.com>", "to": ["aj@purtera-it.com"]}})
    assert stamp_parties([a]) == 1
    assert a.value["said_by"]["company"] == "cdw.com"
    assert a.value["said_to"][0]["side"] == "ours"
    assert stamp_parties([a]) == 0  # idempotent
    # an atom from a PDF has no message and is left alone
    assert stamp_parties([_atom({"kind": "pdf_line"})]) == 0

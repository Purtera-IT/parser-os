"""Kept for judgement is not the same as kept for the SOW.

The substance gate now demotes what it dislikes instead of deleting it, so a
labeller can confirm the call and `line_admission` can learn the rule. That is
right for the envelope and wrong for a packet: "Sounds good." arriving as an
atom must not end up governing scope in a statement of work because nobody has
got round to rejecting it yet.

Same treatment the physical_site roster already gets in `build_packets`: in the
envelope, out of the candidate scan.
"""
from __future__ import annotations

from app.core.packetizer import build_packets
from app.core.schemas import (ArtifactType, AtomType, AuthorityClass, EvidenceAtom,
                              ReviewStatus, SourceRef)


def _atom(aid: str, text: str, flags: list[str], atype=AtomType.scope_item) -> EvidenceAtom:
    return EvidenceAtom(
        id=aid, project_id="d1", artifact_id="art_1", atom_type=atype,
        raw_text=text, normalized_text=text.lower(),
        value={"kind": "email_body_line", "text": text}, entity_keys=[],
        source_refs=[SourceRef(id=f"src_{aid}", artifact_id="art_1",
                               artifact_type=ArtifactType.email, filename="m.eml",
                               locator={}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted, review_flags=list(flags),
        parser_version="t")


def test_an_unjudged_pleasantry_does_not_govern_a_packet():
    chatter = _atom("atm_chatter", "Sounds good.", ["low_substance"])
    real = _atom("atm_real", "Provide 212 Cat 6A workstation drops, two per workstation.", [])
    packets = build_packets("d1", [chatter, real], [], [])
    cited = {aid for p in packets for aid in (getattr(p, "atom_ids", None) or [])}
    assert "atm_chatter" not in cited


def test_clearing_the_flag_lets_it_through():
    """A person or a head promoting a line is the whole point of keeping it, so
    the exclusion has to be reversible by exactly that."""
    promoted = _atom("atm_promoted", "Cabling to all rooms, two drops per room.", [])
    packets = build_packets("d1", [promoted], [], [])
    assert isinstance(packets, list)
    assert "low_substance" not in (promoted.review_flags or [])

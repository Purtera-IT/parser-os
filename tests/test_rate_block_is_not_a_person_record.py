"""A passage that names a person is not that person's record.

Live 010353: the SOW premium-rate block named who approves premium-rate work.
The structural people bridge minted a stakeholder atom whose text was the
WHOLE block (``raw[:500]``), so the person record and the block were the same
text; pre_classify_dedup then folded one into the other -- the block came out
typed stakeholder, or a "stakeholder" copy of it was dropped.
"""

from __future__ import annotations

from app.core.entity_extraction import _structural_people_atoms
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.semantic_dedup import cross_type_dedup_atoms

_RATE_BLOCK = (
    "Work performed outside normal business hours is billed at 1.5x the "
    "standard rate on weekdays after 5:00 PM and 2.0x on weekends and "
    "holidays. All premium-rate work must be approved in advance by "
    "Dana Whitfield, Project Manager."
)


def _block(atom_type: AtomType) -> EvidenceAtom:
    return EvidenceAtom(
        id="atm_rate_block", project_id="p", artifact_id="sow",
        atom_type=atom_type, raw_text=_RATE_BLOCK,
        normalized_text=_RATE_BLOCK.lower(), value={},
        authority_class=AuthorityClass.contractual_scope, confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
        entity_keys=["stakeholder:dana_whitfield"],
        source_refs=[SourceRef(
            id="src_rate", artifact_id="sow", artifact_type=ArtifactType.docx,
            filename="010353 SOW.docx", locator={"paragraph_index": 7},
            extraction_method="test", parser_version="test",
        )],
        parser_version="test",
    )


def test_the_person_record_carries_the_person_not_the_passage():
    people = _structural_people_atoms([_block(AtomType.scope_item)], project_id="p")
    dana = [a for a in people if a.value.get("name") == "Dana Whitfield"]
    assert dana, "the person is still read"
    assert dana[0].raw_text == "Dana Whitfield, Project Manager"
    assert dana[0].raw_text in _RATE_BLOCK  # verbatim, so replay stays exact
    assert dana[0].value.get("context") == _RATE_BLOCK


def test_the_block_survives_pre_classify_dedup_under_its_own_type():
    for typ in (AtomType.scope_item, AtomType.pricing_assumption):
        block = _block(typ)
        people = _structural_people_atoms([block], project_id="p")
        kept = cross_type_dedup_atoms([block, *people])
        assert block in kept, typ
        # Nothing typed stakeholder carries the block's words.
        assert not any(
            a.atom_type == AtomType.stakeholder and a.raw_text == _RATE_BLOCK for a in kept
        )
        assert len(kept) == 1 + len(people)

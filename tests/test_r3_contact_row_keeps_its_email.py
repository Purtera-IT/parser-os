"""pre_classify_dedup keeps the contact row that HAS the email.

Deal 010246 (re-run on #268): a contact table row was read twice -- once
with its email, once as a bare "name | title" twin typed higher. The cross-
type key cuts the text at 80 characters, so both keyed alike and the bare
copy won: the row with the address was folded away and the deal lost its
contacts' emails. The address is now protected like a figure: the fuller
row wins when it contains the bare one's words, and otherwise survives.
"""
from __future__ import annotations

from app.core import fold_invariants
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.semantic_dedup import cross_type_dedup_atoms

TITLE = "Senior Project Manager, Facilities and Physical Security Operations, North Campus"


def _atom(aid: str, text: str, atype: AtomType, value: dict | None = None) -> EvidenceAtom:
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id="a", atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value=value or {}, entity_keys=[],
        source_refs=[SourceRef(id=f"s{aid}", artifact_id="a", artifact_type=ArtifactType.docx,
                               filename="SOW.docx",
                               locator={"table_index": 2, "row": 3, "cell": None},
                               extraction_method="docx_table_row_v1", parser_version="t")],
        authority_class=AuthorityClass.contractual_scope, confidence=0.8,
        review_status=ReviewStatus.needs_review, review_flags=[], parser_version="t",
    )


def test_row_with_email_wins_over_bare_twin() -> None:
    bare = _atom("bare", f"Jane Roe | {TITLE}", AtomType.deal_metadata, {"name": "Jane Roe"})
    full = _atom("full", f"Jane Roe | {TITLE} | jane.roe@acme.com | 555-201-3344", AtomType.raw_table_row)
    out = cross_type_dedup_atoms([bare, full])
    texts = [a.raw_text for a in out]
    assert any("jane.roe@acme.com" in t for t in texts), texts


def test_email_survives_when_not_a_containment() -> None:
    a = _atom("a", f"Jane Roe | {TITLE} | Owner", AtomType.deal_metadata)
    b = _atom("b", f"Jane Roe | {TITLE} | jane.roe@acme.com", AtomType.raw_table_row)
    out = cross_type_dedup_atoms([a, b])
    assert any("jane.roe@acme.com" in x.raw_text for x in out)


def test_refuse_fold_names_the_address() -> None:
    w = _atom("w", "Jane Roe", AtomType.deal_metadata)
    l = _atom("l", "Jane Roe jane.roe@acme.com", AtomType.raw_table_row)
    assert "email" in (fold_invariants.refuse_fold(w, l) or "")

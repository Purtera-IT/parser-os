"""Protected email communication kinds must not be span-admitted away."""
from __future__ import annotations

from app.core.schemas import (
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.span_admission import _is_protected_email_atom, _readmit_via_store


def _mk(atom_type: str, text: str, *, value=None) -> EvidenceAtom:
    return EvidenceAtom(
        id="atm_" + str(abs(hash((atom_type, text))))[:12],
        project_id="p",
        artifact_id="art_t",
        atom_type=AtomType(atom_type),
        raw_text=text,
        normalized_text=text.lower(),
        value=value if value is not None else {"text": text},
        entity_keys=[],
        source_refs=[
            SourceRef(
                id="src_t",
                artifact_id="art_t",
                artifact_type="email",
                filename="t.eml",
                locator={},
                extraction_method="t",
                parser_version="t",
            )
        ],
        authority_class=AuthorityClass.customer_current_authored,
        confidence=0.6,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="t",
    )


def test_email_addressee_is_protected() -> None:
    a = _mk(
        "deal_metadata",
        "Eddie,",
        value={"text": "Eddie,", "kind": "email_addressee", "role": "to_greeting"},
    )
    assert _is_protected_email_atom(a)


def test_email_body_context_is_protected() -> None:
    a = _mk(
        "deal_metadata",
        "Appreciate you hopping on in such short notice.",
        value={
            "text": "Appreciate you hopping on in such short notice.",
            "kind": "email_body_context",
            "role": "intro",
        },
    )
    assert _is_protected_email_atom(a)


def test_readmit_store_skips_protected_email_kinds() -> None:
    """Even if store would admit 'Eddie,' as stakeholder, kind lock wins."""
    a = _mk(
        "deal_metadata",
        "Eddie,",
        value={"text": "Eddie,", "kind": "email_addressee", "role": "to_greeting"},
    )
    n = _readmit_via_store(
        [a],
        weak=frozenset({"deal_metadata"}),
        cand=["stakeholder", "requirement"],
        scope=None,
    )
    assert n == 0
    assert a.atom_type == AtomType.deal_metadata

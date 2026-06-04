"""Retained-suppression ledger: a dropped atom is captured + reason-stamped,
never lost. This is the Phase-1 prerequisite for omission-learning — a PM can
only say "you missed X" if X still exists, flagged, after the stage that
removed it.
"""

from __future__ import annotations

from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.suppression_ledger import (
    SUPPRESSION_FLAG_PREFIX,
    capture_suppressed,
    merge_suppressed,
)


def _atom(atom_id: str, *, raw_text: str = "x", value: dict | None = None) -> EvidenceAtom:
    src = SourceRef(
        id=f"src_{atom_id}",
        artifact_id="art",
        artifact_type=ArtifactType.docx,
        filename="sow.docx",
        locator={"extraction": "test"},
        extraction_method="test",
        parser_version="test",
    )
    return EvidenceAtom(
        id=atom_id,
        project_id="p",
        artifact_id="art",
        atom_type=AtomType.scope_item,
        raw_text=raw_text,
        normalized_text=raw_text.lower(),
        value={"kind": "scope_item"} if value is None else value,
        entity_keys=[],
        source_refs=[src],
        receipts=[],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def test_captures_only_the_dropped_atoms():
    a, b, c = _atom("a"), _atom("b"), _atom("c")
    before = [a, b, c]
    after = [a, c]  # b dropped
    dropped = capture_suppressed(before, after, stage="semantic_dedup", reason="dup")
    assert [d.id for d in dropped] == ["b"]


def test_stamps_flag_and_marker_on_dropped_atom():
    a, b = _atom("a"), _atom("b")
    dropped = capture_suppressed([a, b], [a], stage="semantic_dedup", reason="dup of a")
    (only,) = dropped
    assert f"{SUPPRESSION_FLAG_PREFIX}semantic_dedup" in only.review_flags
    assert only.value["_suppression"] == {"stage": "semantic_dedup", "reason": "dup of a"}


def test_does_not_touch_kept_atoms():
    a, b = _atom("a"), _atom("b")
    capture_suppressed([a, b], [a], stage="semantic_dedup", reason="dup")
    # a survived — no suppression flag, no marker.
    assert not any(f.startswith(SUPPRESSION_FLAG_PREFIX) for f in a.review_flags)
    assert "_suppression" not in a.value


def test_preserves_existing_value_keys_and_flags():
    b = _atom("b", value={"kind": "scope_item", "important": 42})
    b.review_flags = ["low_confidence_floor"]
    capture_suppressed([_atom("a"), b], [_atom("a")], stage="execution_boilerplate_drop", reason="r")
    assert b.value["important"] == 42  # existing key untouched
    assert b.value["_suppression"]["stage"] == "execution_boilerplate_drop"
    assert "low_confidence_floor" in b.review_flags  # existing flag retained
    assert f"{SUPPRESSION_FLAG_PREFIX}execution_boilerplate_drop" in b.review_flags


def test_empty_when_nothing_dropped():
    a, b = _atom("a"), _atom("b")
    assert capture_suppressed([a, b], [a, b], stage="dedup", reason="r") == []


def test_merge_dedupes_by_id():
    ledger: list = []
    a, b = _atom("a"), _atom("b")
    merge_suppressed(ledger, [a, b])
    merge_suppressed(ledger, [b])  # b already present
    assert [x.id for x in ledger] == ["a", "b"]


def test_merge_preserves_order_across_stages():
    ledger: list = []
    merge_suppressed(ledger, [_atom("dup1")])
    merge_suppressed(ledger, [_atom("boiler1"), _atom("boiler2")])
    assert [x.id for x in ledger] == ["dup1", "boiler1", "boiler2"]

"""Unticked checkbox rows are facts: semantic_dedup never folds them away.

Deal 010003: in the SOW's service-type checkbox block the unticked rows
("☐ Staff Augmentation ...", "☐ Knowledge Transfer ...") keyed alike with the
ticked "☒ Installation ..." row (same site, same column) and were dropped with
no surviving copy of their own text.
"""
from __future__ import annotations

from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.semantic_dedup import cross_type_dedup_atoms, semantic_dedup_atoms
from app.parsers.checkbox_cells import checkbox_atom

ROWS = [
    "☒ Installation and Configuration of Hardware",
    "☐ Staff Augmentation (on-site resources)",
    "☐ Knowledge Transfer sessions for IT staff",
]


def _cells(rows, *, art="sow", start=0):
    return [
        checkbox_atom(project_id="p", artifact_id=art, artifact_type=ArtifactType.docx, filename=f"{art}.docx",
                      text=t, column="Service Type", subject="Main Campus",
                      locator={"table_index": 3, "row": start + i}, extraction_method="t", parser_version="t",
                      site_row=True)
        for i, t in enumerate(rows)
    ]


def test_every_checkbox_row_survives_semantic_dedup() -> None:
    out = semantic_dedup_atoms(_cells(ROWS))
    assert [a.raw_text for a in out] == ROWS


def test_the_same_checkbox_row_twice_still_folds() -> None:
    out = semantic_dedup_atoms(_cells([ROWS[1]]) + _cells([ROWS[1]], start=5))
    assert [a.raw_text for a in out] == [ROWS[1]]


def _box(aid, text, atype, value):
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id="sow", atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value=value, entity_keys=[],
        source_refs=[SourceRef(id=f"s{aid}", artifact_id="sow", artifact_type=ArtifactType.docx,
                               filename="sow.docx", locator={"paragraph_index": len(aid)},
                               extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.contractual_scope, confidence=0.8,
        review_status=ReviewStatus.needs_review, review_flags=[], parser_version="t",
    )


def test_an_unticked_box_is_not_retyped_into_a_ticked_one() -> None:
    ticked = _box("c1", "☒ Staff Augmentation", AtomType.scope_item,
                  {"kind": "checkbox", "label": "Staff Augmentation", "checked": True})
    unticked = _box("c22", "☐ Staff Augmentation", AtomType.form_option_state,
                    {"kind": "checkbox", "label": "Staff Augmentation", "checked": False})
    out = cross_type_dedup_atoms([ticked, unticked])
    assert sorted(a.raw_text for a in out) == ["☐ Staff Augmentation", "☒ Staff Augmentation"]

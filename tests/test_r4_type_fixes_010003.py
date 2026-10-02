"""010003 round 4: a conflict needs two real values; Gantt labour rows and
the customer PO line are not totals.

1. A CDW quote footer with the copyright year beside a copy without it
   raised "Documents disagree on one clause: 2026 vs (no figure)". A copy
   that lacks the figure is another rendering of the line, not a second value.
2. A Deal Kit / Gantt row "Task | Hours | Tech Level | Cost" (no dates) and
   "Customer PO | 4500123456" came out commercial_total / pricing_assumption.
   A labour row is a task, a PO line is deal_metadata; only Total rows are
   totals.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.cross_document_conflicts import find_cross_document_conflicts
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _atom(aid: str, doc: str, atype: AtomType, text: str) -> EvidenceAtom:
    src = SourceRef(id=f"s_{aid}", artifact_id=doc, artifact_type=ArtifactType.xlsx, filename=f"{doc}.xlsx",
                    locator={}, extraction_method="x", parser_version="t")
    return EvidenceAtom(id=aid, project_id="p", artifact_id=doc, atom_type=atype, raw_text=text,
                        normalized_text=text.lower(), value={"text": text}, entity_keys=[], source_refs=[src],
                        authority_class=AuthorityClass.vendor_quote, confidence=0.8,
                        review_status=ReviewStatus.needs_review, parser_version="t")


def test_a_copy_without_the_figure_is_not_a_conflict() -> None:
    # Phrased so no boilerplate band catches it: only the two-values rule can.
    a = _atom("q1", "quote1", AtomType.scope_item, "CDW quote 2026 prepared for the customer by the account team at 200 Milwaukee Avenue")
    b = _atom("q2", "quote2", AtomType.scope_item, "CDW quote prepared for the customer by the account team at 200 Milwaukee Avenue")
    assert find_cross_document_conflicts([a, b], project_id="p") == []
    # Two real values still conflict, and the copy without a figure is left out.
    c = _atom("c1", "sow", AtomType.scope_item, "A cancellation fee of $500 applies to visits cancelled late by the customer")
    d = _atom("c2", "draft", AtomType.scope_item, "A cancellation fee of $300 applies to visits cancelled late by the customer")
    e = _atom("c3", "notes", AtomType.scope_item, "A cancellation fee of applies to visits cancelled late by the customer")
    qs = find_cross_document_conflicts([c, d, e], project_id="p")
    assert len(qs) == 1, [q.raw_text for q in qs]
    assert "no figure" not in qs[0].raw_text and "$500" in qs[0].raw_text and "$300" in qs[0].raw_text
    assert sorted(qs[0].value["artifact_ids"]) == ["draft", "sow"]


def test_labor_rows_and_po_lines_are_not_totals() -> None:
    from app.core.atom_type_sanity import retype_schedule_reference_rows

    rows = [
        ("g1", AtomType.commercial_total, "Task: Install displays | Hours: 24 | Tech Level: L2 Tech | Cost: 4200"),
        ("g2", AtomType.commercial_total, "Cable and terminate | 8 | L2 Tech | 760"),
        ("g3", AtomType.pricing_assumption, "Project management | Est. Hrs: 6 | PM | $720.00"),
        ("po", AtomType.pricing_assumption, "Customer PO | 4500123456"),
        ("pn", AtomType.commercial_total, "PO Number | PO-88213 | 18207.48"),
        ("pl", AtomType.commercial_total, "Customer PO #4500123456 - $18,207.48"),
        ("tt", AtomType.commercial_total, "Task: Total | Hours: 34 | Cost: 5210"),
        # A rate card row is a rate, not a task.
        ("rc", AtomType.pricing_assumption, "Install | L2 Tech | 95"),
        ("lv", AtomType.commercial_total, "L2 Tech | 24 | 95"),
    ]
    atoms = [_atom(i, "dk", t, x) for i, t, x in rows]
    retype_schedule_reference_rows(atoms)
    got = {a.id: a.atom_type.value for a in atoms}
    assert got == {
        "g1": "task", "g2": "task", "g3": "task",
        "po": "deal_metadata", "pn": "deal_metadata", "pl": "deal_metadata",
        "tt": "commercial_total", "rc": "pricing_assumption", "lv": "commercial_total",
    }, got


def test_deal_kit_labor_rows_after_compile(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from app.core.compiler import compile_project

    d = tmp_path / "deal"
    d.mkdir()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Gantt"
    for r in (["Oppty #", "010003"], ["Sales Rep", "Pat Doe"], ["Total Deal Revenue", 18207.48],
              ["Margin %", 0.31], ["Customer PO", 4500123456], [],
              ["Task", "Hours", "Tech Level", "Cost"], ["Project kickoff", 2, "PM", 250],
              ["Install displays", 24, "L2 Tech", 4200], ["Cable and terminate", 8, "Level 2", 760],
              ["Total", 34, None, 5210]):
        ws.append(r)
    wb.save(d / "Deal Kit.xlsx")
    r = compile_project(d, project_id="p", allow_errors=True, use_cache=False)
    for name in ("Project kickoff", "Install displays", "Cable and terminate", "Customer PO"):
        hits = [a for a in r.atoms if name in (a.raw_text or "")]
        assert hits, (name, [a.raw_text for a in r.atoms])
        for a in hits:
            assert a.atom_type.value not in ("commercial_total", "pricing_assumption"), (a.raw_text, a.atom_type)
    totals = [a for a in r.atoms if (a.raw_text or "").startswith("Task: Total")]
    assert totals and all(a.atom_type == AtomType.commercial_total for a in totals)


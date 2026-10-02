"""Every PDF line is accounted for one line at a time (010003 audit, O6).

A line counted as read when any atom merely touched it: one short atom
inside a long line claimed the whole line, and a PDF was never diffed at
all. So "f. Complete billing tasks" and the PO's "CDW PO's are not
transferrable." produced no atom and still never showed as unread. Now a
PDF is read back off its text layer and a line is claimed only when the
atoms between them carry all of it (an enumerator aside). And the PO clause
that sat under a priced cell row is its own atom again.
"""
from __future__ import annotations

from pathlib import Path

import fitz

from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.text_coverage import build_text_coverage, coverage_for_artifact

SOW_LINES = [
    "a. Coordinate resources and site access with the customer.",
    "b. Develop schedule and share it weekly.",
    "f. Complete billing tasks",
    "Technicians must submit signed timesheets within 48 hours of each visit.",
    "NYC Office 40 10th Ave Fl 4, New York, NY 10014",
]
PO_LINES = [
    "Vendor: PurTera LLC",
    "Ship To: CDW, 200 N Milwaukee Ave, Vernon Hills, IL 60061",
    "Line 1 | Onsite installation services | 1 | $4,500.00",
    "CDW PO's are not transferrable.",
    "Terms: Net 30",
]


def _pdf(path: Path, lines: list[str]) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i, t in enumerate(lines):
        page.insert_text((72, 60 + 15 * i), t, fontsize=10, fontname="helv")
    doc.save(path)
    return path


def _atom(text: str) -> EvidenceAtom:
    return EvidenceAtom(
        id=f"a{abs(hash(text)) % 10**8}", project_id="p", artifact_id="art", atom_type=AtomType.scope_item,
        raw_text=text, normalized_text=text.lower(), value={}, entity_keys=[],
        source_refs=[SourceRef(id="s", artifact_id="art", artifact_type=ArtifactType.pdf, filename="s.pdf",
                               locator={"page": 0}, extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.contractual_scope, confidence=0.8,
        review_status=ReviewStatus.needs_review, review_flags=[], parser_version="t",
    )


def test_a_pdf_line_with_no_atom_is_unread_even_when_an_atom_touches_it(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "s.pdf", SOW_LINES)
    kept = [
        _atom("Coordinate resources and site access with the customer."),   # enumerator aside
        _atom("Develop schedule and share it weekly."),
        _atom("billing"),                                                    # touches, does not carry
        _atom("Technicians must submit signed timesheets within 48 hours of each visit."),
    ]
    row = coverage_for_artifact(pdf, "art", kept)
    unread = {x["text"]: x for x in row["unclaimed"] if x["state"] == "unread"}
    assert set(unread) == {"f. Complete billing tasks", "NYC Office 40 10th Ave Fl 4, New York, NY 10014"}
    assert all(x["page"] == 1 for x in unread.values())
    assert row["lines_claimed"] == 3 and row["unread_count"] == 2


def test_two_atoms_that_share_a_line_claim_it_together(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "s.pdf", ["NYC Office 40 10th Ave Fl 4, New York, NY 10014"])
    row = coverage_for_artifact(pdf, "art", [_atom("NYC Office"), _atom("40 10th Ave Fl 4, New York, NY 10014")])
    assert row["unread_count"] == 0 and row["lines_claimed"] == 1


def test_build_text_coverage_reads_pdfs(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "s.pdf", SOW_LINES)
    rows = build_text_coverage({"art": pdf}, [])
    assert rows and rows[0]["unread_count"] == len(SOW_LINES)


def test_po_clause_under_a_priced_row_is_its_own_atom(tmp_path: Path) -> None:
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    pdf = _pdf(tmp_path / "PO 4410023.pdf", PO_LINES)
    atoms = OrbitBriefPdfParser().parse_artifact("p", "art", pdf)
    atoms = atoms if isinstance(atoms, list) else atoms.atoms
    texts = [a.raw_text for a in atoms]
    assert "CDW PO's are not transferrable." in texts, texts
    assert not any("Ship To" in t and "Line 1" in t for t in texts), texts
    row = coverage_for_artifact(pdf, "art", atoms)
    assert row["unread_count"] == 0, row["unclaimed"]

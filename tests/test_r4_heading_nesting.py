"""Headings open sibling sections, and a bare heading is not an atom (010087).

(a) The signed SOW is a Docusign PDF saved with a ".docx" name. Its cover
    title is a logo, so the first text line under the envelope stamp is
    "Executive Summary"; it was crowned the document title, the root of every
    atom's section_path, so Introduction, Scope, Responsibilities and Fees
    all read as nested under it. And its Title-Case headings are set bold,
    a weight the text layer drops: "Introduction" read as a prose line (its
    own atom) and its paragraph stayed under the heading above.
(b) The SOW drafts (docx) emitted "INTRODUCTION" and every other heading as a
    standalone atom. A heading is the section_path of the lines under it; a
    bare one waits in the suppressed sidecar, one that says more than its
    title ("Fees: $24,500") stays a line.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.core.compiler import compile_project  # noqa: E402

ENVELOPE = "Docusign Envelope ID: 3F2A9C1E-1B2C-4D5E-9F00-ABCDEF123456"

SECTIONS = [
    ("Executive Summary", ["Purtera will deploy network equipment at 12 retail locations for the Customer.",
                           "This document defines the work, the responsibilities and the fees."]),
    ("Introduction", ["The Customer operates restaurants across the region and requires new access points."]),
    ("Scope of Work", ["Provider will install two wireless access points per site.",
                       "Provider will mount the rack and dress all patch cables."]),
    ("Customer Responsibilities", ["Customer will provide site access during business hours."]),
    ("Fees", ["The total fee for the services is $24,500 billed on completion."]),
]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _signed_sow(path: Path, heading=lambda i, h: h) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 20), ENVELOPE, fontsize=7)
    y = 70
    for i, (h, paras) in enumerate(SECTIONS, 1):
        page.insert_text((36, y), heading(i, h), fontsize=11, fontname="hebo")
        y += 20
        for p in paras:
            page.insert_text((36, y), p, fontsize=10)
            y += 14
        y += 14
    doc.save(str(path))
    doc.close()


def _pdf_atoms(path: Path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    out = OrbitBriefPdfParser().parse(path)
    return list(getattr(out, "atoms", out))


def _path(a) -> list[str]:
    return list((a.source_refs[0].locator or {}).get("section_path") or [])


@pytest.mark.parametrize("style", [
    lambda i, h: h,                    # bold Title Case
    lambda i, h: h.upper(),            # capitals
    lambda i, h: f"{i}. {h.upper()}",  # numbered
])
def test_signed_sow_headings_are_siblings_not_children_of_executive_summary(tmp_path, style):
    pdf = tmp_path / "Signed SOW.docx"  # a PDF under a Word name, as delivered
    _signed_sow(pdf, style)
    atoms = _pdf_atoms(pdf)
    by_text = {a.raw_text: a for a in atoms}
    for i, (h, paras) in enumerate(SECTIONS, 1):
        for p in paras:
            a = by_text.get(p)
            assert a is not None, (p, list(by_text))
            path = _path(a)
            assert path, p
            # its own heading is the last step of the path ...
            assert path[-1].lower().endswith(h.lower()), (p, path)
            # ... and no other section heading sits above it
            others = {x.lower() for x, _ in SECTIONS if x != h}
            assert not any(step.lower().split(". ")[-1] in others for step in path), (p, path)
    # A heading is the path, never its own atom.
    heads = {style(i, h) for i, (h, _) in enumerate(SECTIONS, 1)}
    assert not [a.raw_text for a in atoms if a.raw_text in heads]


def test_a_bold_caption_over_its_short_value_is_not_a_heading():
    from app.parsers.orbitbrief_pdf import _text_rich_sections

    text = "Shipping Method\nUPS Ground\n\nIntroduction\nThe Customer operates restaurants across the region.\n"
    secs = _text_rich_sections(text, heading_hints={"Shipping Method", "Introduction"})
    heads = [s["heading"] for s in secs]
    assert "Introduction" in heads
    assert "Shipping Method" not in heads


def test_a_page_title_that_is_a_summary_label_is_not_the_document_root(tmp_path):
    """Even with a real title above it, "Executive Summary" stays a sibling."""
    pdf = tmp_path / "sow.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 40), "Executive Summary", fontsize=16, fontname="hebo")
    page.insert_text((36, 70), "Purtera will deploy network equipment at 12 retail locations.", fontsize=10)
    page.insert_text((36, 110), "Introduction", fontsize=11, fontname="hebo")
    page.insert_text((36, 130), "The Customer operates restaurants across the whole region.", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    for a in _pdf_atoms(pdf):
        if "restaurants" in a.raw_text:
            assert _path(a) == ["Introduction"], _path(a)


# ── (b) docx: a bare heading is section metadata ─────────────────────────

def _draft(path: Path) -> None:
    from docx import Document

    d = Document()
    d.add_heading("Statement of Work", 0)
    d.add_heading("INTRODUCTION", 1)
    d.add_paragraph("The Customer operates restaurants across the region and requires new access points.")
    d.add_heading("3. Customer Responsibilities", 1)
    d.add_paragraph("Customer will provide site access during business hours.")
    d.add_heading("Fees: $24,500 fixed", 1)
    d.add_paragraph("The fee is billed on completion of the last site.")
    d.save(str(path))


def test_docx_bare_headings_are_section_paths_not_atoms(tmp_path):
    deal = tmp_path / "deal"
    deal.mkdir()
    _draft(deal / "SOW draft.docx")
    r = compile_project(deal, project_id="p", allow_errors=True, use_cache=False)
    texts = {a.raw_text: a for a in r.atoms}
    for h in ("Statement of Work", "INTRODUCTION", "3. Customer Responsibilities"):
        assert h not in texts, h
        held = [a for a in r.suppressed_atoms if a.raw_text == h]
        assert held and "suppressed:section_heading" in held[0].review_flags, h
    intro = texts["The Customer operates restaurants across the region and requires new access points."]
    assert _path(intro)[-1] == "INTRODUCTION"
    cust = texts["Customer will provide site access during business hours."]
    assert _path(cust)[-1] == "3. Customer Responsibilities"
    # A heading that carries a figure says something: it stays a line.
    assert "Fees: $24,500 fixed" in texts
    # Read, not missed: the heading is the path of the lines under it.
    unread = [row["text"] for cov in (r.text_coverage or []) for row in cov.get("unclaimed") or []
              if row.get("state") == "unread"]
    assert "INTRODUCTION" not in unread and "3. Customer Responsibilities" not in unread, unread


def test_heading_carries_content():
    from app.parsers.sow_sections import heading_carries_content as f

    for bare in ("INTRODUCTION", "3. Customer Responsibilities", "A. IT Infrastructure Support",
                 "1.1 Project Overview", "Section 4: Fees", "(b) Site Access"):
        assert not f(bare), bare
    for content in ("Fees: $24,500", "Term: 12 months", "Scope. Provider will install the access points"):
        assert f(content), content

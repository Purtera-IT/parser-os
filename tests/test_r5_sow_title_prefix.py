"""A body heading is not the document title (010087 signed SOW).

The signed SOW is a Docusign PDF whose cover is a logo: page 0 holds only the
envelope stamp and the footer, page 1 opens on INTRODUCTION, EXECUTIVE SUMMARY,
PURTERA SALES CONTACTS, CUSTOMER CONTACTS, SOW LOCATION, page 2 on PROJECT
OVERVIEW and SCOPE OF WORK. The title picker rejected INTRODUCTION and
EXECUTIVE SUMMARY as section labels, then fell back to the first other
heading on the page, "PURTERA SALES CONTACTS", and prefixed it onto every
atom's section_path, the page-0 footer and INTRODUCTION included. The layout
mirrors the real pdftotext -layout of the deal; all text here is synthetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

ENV = "Docusign Envelope ID: 00000000-1111-2222-3333-444444444444"
FOOTER = "WWW.EXAMPLE-VENDOR.COM"


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


class _Page:
    def __init__(self, doc):
        self.p = doc.new_page(width=612, height=792)
        self.p.insert_text((36, 20), ENV, fontsize=7)
        self.y = 70

    def head(self, t):
        self.p.insert_text((90, self.y), t, fontsize=12, fontname="hebo")
        self.y += 22

    def text(self, t):
        self.p.insert_text((90, self.y), t, fontsize=9)
        self.y += 24

    def row(self, cells, xs=(95, 250, 400, 500)):
        for c, x in zip(cells, xs):
            self.p.insert_text((x + 3, self.y), c, fontsize=9)
        self.p.draw_rect(fitz.Rect(xs[0] - 2, self.y - 11, 560, self.y + 5), color=(0, 0, 0), width=0.5)
        for x in xs[1:len(cells)]:
            self.p.draw_line((x - 2, self.y - 11), (x - 2, self.y + 5), color=(0, 0, 0), width=0.5)
        self.y += 16


def _signed_sow(path: Path, cover_title: str | None = None) -> None:
    doc = fitz.open()
    p0 = _Page(doc)
    if cover_title:
        p0.p.insert_text((90, 300), cover_title, fontsize=20, fontname="hebo")
    p0.p.insert_text((90, 770), FOOTER, fontsize=8)
    p = _Page(doc)
    p.head("INTRODUCTION")
    p.row(["SOW VERSION", "QUOTED BY", "DATE", "REVISION HISTORY"])
    p.row(["v.1", "Alex Rivera", "07/09/2026", "First"])
    p.row(["v.2", "Jordan Lee", "07/16/2026", "Second"])
    p.y += 10
    p.head("EXECUTIVE SUMMARY")
    p.text('This Project Services Statement of Work ("SOW") is made by and between Acme Corp (the "Customer").')
    xs = (95, 250, 400)
    p.head("PURTERA SALES CONTACTS")
    p.row(["FULL NAME", "JOB TITLE", "EMAIL ADDRESS"], xs)
    p.row(["Sam Carter", "Executive VP Sales", "sam@example-vendor.com"], xs)
    p.row(["Alex Rivera", "Solution Architect", "alex@example-vendor.com"], xs)
    p.y += 10
    p.head("CUSTOMER CONTACTS")
    p.row(["FULL NAME", "JOB TITLE", "EMAIL ADDRESS"], xs)
    p.row(["Morgan Diaz", "Client Executive", "morgan.diaz@example-client.com"], xs)
    p.p.insert_text((90, 770), FOOTER, fontsize=8)
    p = _Page(doc)
    p.head("PROJECT OVERVIEW")
    p.text("The client requires seasonal laptop redeployment across all school locations.")
    p.head("SCOPE OF WORK")
    p.text("Provide on-site technical support for laptop redeployment at each location.")
    doc.save(str(path))
    doc.close()


def _atoms(path: Path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    out = OrbitBriefPdfParser().parse(path)
    return list(getattr(out, "atoms", out))


def _path(a) -> list[str]:
    return list((a.source_refs[0].locator or {}).get("section_path") or [])


def test_a_body_heading_is_not_prefixed_onto_every_path(tmp_path):
    pdf = tmp_path / "Signed SOW.pdf"
    _signed_sow(pdf)
    atoms = _atoms(pdf)
    assert atoms
    for a in atoms:
        assert "PURTERA SALES CONTACTS" not in _path(a)[:-1], (a.raw_text, _path(a))
        assert len(_path(a)) <= 1, (a.raw_text, _path(a))

    def path_of(fragment):
        hits = [a for a in atoms if fragment in a.raw_text]
        assert hits, (fragment, [a.raw_text for a in atoms])
        return [x.upper() for x in _path(hits[0])]

    assert path_of("Alex Rivera | DATE") == ["INTRODUCTION"]
    assert path_of("is made by and between") == ["EXECUTIVE SUMMARY"]
    assert path_of("Sam Carter") == ["PURTERA SALES CONTACTS"]
    assert path_of("Morgan Diaz") == ["CUSTOMER CONTACTS"]
    assert path_of("seasonal laptop") == ["PROJECT OVERVIEW"]
    assert path_of("on-site technical support") == ["SCOPE OF WORK"]
    assert path_of(FOOTER) == []


def test_a_real_cover_title_is_still_the_root(tmp_path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser  # noqa: F401

    pdf = tmp_path / "Signed SOW.pdf"
    _signed_sow(pdf, cover_title="Acme Laptop Refresh Program")
    atoms = _atoms(pdf)
    scope = [a for a in atoms if "on-site technical support" in a.raw_text]
    assert scope and _path(scope[0]) == ["Acme Laptop Refresh Program", "SCOPE OF WORK"], _path(scope[0])

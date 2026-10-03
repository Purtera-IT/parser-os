"""R8 (010353, 010087): a PDF heading atom reads before the line it leads.

#341 gave a heading that leads child lines its own atom, but stamped it with
the reading-order index of its first child. The pair tied on block_index and
line_start, the tie fell to the atom id (a hash), and the heading read after
its first item about half the time: a section opening with a plain intro
paragraph, and one opening with a bullet after a page break, both did it.

A heading now takes the half slot just before its first line (n - 0.5).
Every other atom keeps its integer index, id and text.
"""

from __future__ import annotations

import random

import pytest

fitz = pytest.importorskip("fitz")


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _put(page, x, top, text, size, font="helv"):
    page.insert_text((x, top + size * 0.8), text, fontsize=size, fontname=font)


def _sow(path):
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    _put(p, 72, 90, "STATEMENT OF WORK", 16, "hebo")
    _put(p, 72, 120, "This statement covers a device refresh for the customer.", 12)
    # Section opening with an intro paragraph at the heading's own x.
    _put(p, 72, 272, "WORK PLAN", 12.96, "hebo")
    _put(p, 72, 295, "Provide on-site support for device refresh across all listed sites.", 12)
    for k, top in enumerate((320, 338)):
        _put(p, 90, top, "•", 12)
        _put(p, 108, top + 0.4, f"Field task number {k + 1} for the refresh at each site.", 12)
    _put(p, 72, 744, "WWW.EXAMPLE.COM", 12, "hebo")
    p = doc.new_page(width=612, height=792)
    _put(p, 90, 103, "•", 12)
    _put(p, 108, 103.4, "Field task number 3 carried over the page break.", 12)
    # Section opening with a bullet.
    _put(p, 72, 243, "OUTPUTS", 12.96, "hebo")
    for k, top in enumerate((265, 283)):
        _put(p, 90, top, "•", 12)
        _put(p, 108, top + 0.4, f"Output document number {k + 1} for each site.", 12)
    _put(p, 72, 744, "WWW.EXAMPLE.COM", 12, "hebo")
    doc.save(str(path))
    doc.close()


def _loc(a) -> dict:
    return a.source_refs[0].locator or {}


def _first_child(atoms, head):
    kids = [a for a in atoms if _loc(a).get("block_kind") != "heading"
            and _loc(a).get("section_path") == _loc(head).get("section_path")]
    return min(kids, key=lambda a: _loc(a)["block_index"])


def test_pdf_heading_takes_the_slot_before_its_first_line(tmp_path):
    from app.core.orbitbrief_envelope import _in_reading_order
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    pdf = tmp_path / "sow.pdf"
    _sow(pdf)
    atoms = list(OrbitBriefPdfParser().parse(pdf).atoms)
    heads = [a for a in atoms if _loc(a).get("block_kind") == "heading"]
    assert [a.raw_text for a in heads] == ["WORK PLAN", "OUTPUTS"]

    rest = [a for a in atoms if _loc(a).get("block_kind") != "heading"]
    # every other atom keeps its integer reading index, 0..n-1
    assert [_loc(a)["block_index"] for a in rest] == list(range(len(rest)))
    assert all(type(_loc(a)["block_index"]) is int for a in rest)

    for h in heads:
        first = _first_child(atoms, h)
        n = _loc(first)["block_index"]
        assert _loc(h)["block_index"] == n - 0.5
        assert _loc(h)["line_start"] == _loc(h)["line_end"] == n - 0.5
        assert _loc(first)["line_start"] == n

    for seed in range(12):
        shuffled = list(atoms)
        random.Random(seed).shuffle(shuffled)
        for order in (
            sorted(shuffled, key=lambda a: _loc(a)["block_index"]),
            sorted(shuffled, key=lambda a: _loc(a)["line_start"]),
            sorted(shuffled, key=lambda a: (_loc(a).get("page") or 0, _loc(a)["block_index"],
                                            _loc(a)["line_start"], _loc(a).get("sentence_index") or 0)),
            _in_reading_order(shuffled, []),
        ):
            for h in heads:
                i = order.index(h)
                assert order.index(_first_child(atoms, h)) == i + 1, (seed, h.raw_text)


def test_heading_slot_leaves_ids_and_text_alone(tmp_path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    pdf = tmp_path / "sow.pdf"
    _sow(pdf)
    a1 = list(OrbitBriefPdfParser().parse(pdf).atoms)
    a2 = list(OrbitBriefPdfParser().parse(pdf).atoms)
    assert [(a.id, a.raw_text) for a in a1] == [(a.id, a.raw_text) for a in a2]
    assert len({a.id for a in a1}) == len(a1)

"""A count surfaced from a PO box carries the whole box (010003 PO Bill To).

A PO's "Bill To" box reads as one field, "Bill To: <address lines>", its
lines joined with spaces. The first value line ends in an abbreviation
("Accounts Payable Dept."), so the sentence splitter ended a sentence there,
and the headline quantity surfaced from the box's ZIP code took only the
"sentence" after it: a second atom on the same block and line, with the same
cell fragments, holding the box minus its first line. Its text differed from
the box's, so nothing folded it. The "Ship To" box has no period in it and
its copy was the whole box. A boxed field is one value, not prose: the
surfaced count's context is the whole field.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.core.atom_type_sanity import surface_headline_quantities  # noqa: E402
from app.core.schemas import AtomType  # noqa: E402
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser  # noqa: E402

GREY = (0.50196, 0.50196, 0.50196)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _box(p, x0, x1, top, label, values, bottom):
    """A Times 8.3 label on an unstroked grey cell over a line-ruled box of
    Helvetica 10 value lines, as on the real PO."""
    bar = top + 13.0
    p.draw_rect(fitz.Rect(x0, top, x1, bar), color=None, fill=GREY)
    p.insert_text((x0 + 2.5, bar - 3.0), label, fontsize=8.3, fontname="tiro")
    for x in (x0 + 0.4, x1 - 0.4):
        p.draw_line((x, top), (x, bottom), width=0.6)
    for y in (top + 0.4, bar + 0.4, bottom - 0.4):
        p.draw_line((x0, y), (x1, y), width=0.6)
    for k, v in enumerate(values):
        p.insert_text((x0 + 2.5, bar + 12 + 11.6 * k), v, fontsize=10, fontname="helv")


def _po(path: Path) -> None:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    _box(p, 348.2, 583.4, 218.0, "Ship To",
         ["Pat Lee (5501)", "12 Harbor Street", "Suite 300", "Boston, MA 02110", "United States of America"],
         bottom=315.4)
    _box(p, 348.6, 584.4, 365.0, "Bill To",
         ["Accounts Payable Dept.", "Pat Lee (5501)", "40 River Road", "Albany, NY 12207",
          "United States of America"],
         bottom=450.4)
    doc.save(str(path))
    doc.close()


def _surface(pdf: Path):
    atoms = list(getattr(OrbitBriefPdfParser().parse(pdf), "atoms", []))
    for a in atoms:
        a.atom_type = AtomType.scope_item
    return atoms, surface_headline_quantities(atoms, project_id="p")


def test_count_in_a_box_carries_the_whole_box(tmp_path):
    pdf = tmp_path / "PO-55012.pdf"
    _po(pdf)
    atoms, surfaced = _surface(pdf)
    bill = next(a for a in atoms if a.raw_text.startswith("Bill To:"))
    assert bill.source_refs[0].locator.get("cell_fragments"), bill.source_refs[0].locator
    on_bill = [q for q in surfaced if q.source_refs[0].locator.get("block_id")
               == bill.source_refs[0].locator.get("block_id")]
    assert on_bill, [q.raw_text for q in surfaced]
    for q in on_bill:
        # The whole box, never the box minus "Bill To: Accounts Payable Dept.".
        assert q.raw_text == bill.raw_text, q.raw_text
        assert q.value["context"] == bill.raw_text


def test_count_in_prose_keeps_its_sentence():
    """Prose with no cell fragments still carries just the count's sentence."""
    from app.core.schemas import ArtifactType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef

    text = "Kickoff is on Monday. We will install 24 cameras across the warehouse."
    atom = EvidenceAtom(
        id="atm_prose", project_id="p", artifact_id="art_x", atom_type=AtomType.scope_item,
        raw_text=text, normalized_text=text.lower(), value={}, entity_keys=[],
        source_refs=[SourceRef(id="src_1", artifact_id="art_x", artifact_type=ArtifactType.pdf,
                               filename="f.pdf", locator={"page": 0, "block_id": "blk_1"},
                               extraction_method="test", parser_version="t")],
        receipts=[], authority_class=AuthorityClass.machine_extractor, confidence=0.9,
        review_status=ReviewStatus.auto_accepted, parser_version="t",
    )
    surfaced = surface_headline_quantities([atom], project_id="p")
    assert [q.raw_text for q in surfaced] == ["We will install 24 cameras across the warehouse."]

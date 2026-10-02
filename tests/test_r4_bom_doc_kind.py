"""Only a vendor quote / BOM is a vendor_quote_bom (010003 round 4).

Live 010003 stamped document_kind "vendor_quote_bom" on every atom of CDW's
signed SOW PDF and of CDW's PO PDF, not only on the BOM: both price a line
from a QTY column, which was the whole test. And "Complete billing tasks",
a short duty on page 2 of the signed SOW, came out deal_metadata /
conversation_meta: the transcript-turn rule read any short line past page 1
as call filler by its word count, while the draft .docx copy stayed scope.
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_r3_vendor_quote_line_items import _quote  # noqa: E402

from app.core.schemas import AtomType  # noqa: E402

DUTIES = ["Coordinate resources and site access with Customer", "Validate items/services provided with Customer",
          "Complete billing tasks"]


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _priced_row(p, y: int) -> None:
    for x, h in ((36, "ITEM"), (330, "QTY"), (450, "UNIT PRICE"), (530, "EXT. PRICE")):
        p.insert_text((x, y), h, fontsize=8, fontname="hebo")
    for x, v in ((36, "Onsite installation services"), (330, "1"), (450, "$4,500.00"), (530, "$4,500.00")):
        p.insert_text((x, y + 16), v, fontsize=8)


def _sow(path: Path) -> None:
    """A CDW-generated SOW: priced like the quote, its duty list on page 2."""
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((36, 60), "PROJECT DESCRIPTION", fontsize=12, fontname="hebo")
    p.insert_text((36, 84), "CDW will install twelve TVs at the Customer's NYC office as described below.", fontsize=10)
    _priced_row(p, 140)
    p = doc.new_page(width=612, height=792)
    p.insert_text((36, 60), "PROVIDER RESPONSIBILITIES", fontsize=12, fontname="hebo")
    p.insert_text((36, 84), "CDW will provide project management for the duration of the project. Provider will:",
                  fontsize=10)
    y = 104
    for i, t in enumerate(DUTIES):
        p.insert_text((48, y), f"{'abc'[i]}. {t}", fontsize=10)
        y += 16
    p.insert_text((36, y + 20), "CUSTOMER RESPONSIBILITIES", fontsize=12, fontname="hebo")
    p.insert_text((36, y + 44), "Customer is responsible for providing site access during business hours.", fontsize=10)
    doc.save(str(path))
    doc.close()


def _po(path: Path) -> None:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((36, 60), "PURCHASE ORDER", fontsize=16, fontname="hebo")
    p.insert_text((36, 84), "Vendor: PurTera LLC", fontsize=10)
    _priced_row(p, 120)
    p.insert_text((36, 180), "CDW PO's are not transferrable.", fontsize=9)
    doc.save(str(path))
    doc.close()


def _kinds(atoms, name: str) -> set:
    return {(a.value or {}).get("document_kind") for a in atoms if a.source_refs[0].filename == name}


def test_only_the_quote_is_a_bom_and_sow_duties_stay_scope(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    d = tmp_path / "deal"
    d.mkdir()
    _quote(d / "CDW Quote PSNV676.pdf", right_aligned=True, ruled=True)
    sow = "SOW_198950 198584 for Project 4 TV Install NYC is ready for signature.pdf"
    _sow(d / sow)
    _po(d / "PO 4500123456.pdf")
    r = compile_project(d, project_id="p", allow_errors=True, use_cache=False)
    # The quote's own atoms are a BOM (a site the compiler derives from its
    # ship-to address is not one of them).
    q = [a for a in r.atoms if a.source_refs[0].filename == "CDW Quote PSNV676.pdf"
         and a.atom_type != AtomType.physical_site]
    assert q and all(a.value.get("document_kind") == "vendor_quote_bom" for a in q)
    assert _kinds(r.atoms, sow) == {None}, _kinds(r.atoms, sow)
    assert _kinds(r.atoms, "PO 4500123456.pdf") == {None}
    for duty in DUTIES:
        hits = [a for a in r.atoms if (a.raw_text or "").strip() == duty]
        assert hits, (duty, [a.raw_text for a in r.atoms])
        for a in hits:
            assert a.atom_type == AtomType.scope_item, (duty, a.atom_type, a.value, a.review_flags)
            assert "conversation_meta" not in a.review_flags and "chatter" not in a.review_flags


def test_a_priced_document_is_judged_by_what_it_says_it_is() -> None:
    from app.core.schemas import ArtifactType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
    from app.parsers.orbitbrief_pdf import _tag_vendor_quote_document

    def _a(aid, text, filename, section):
        return EvidenceAtom(
            id=aid, project_id="p", artifact_id="x", atom_type=AtomType.vendor_line_item, raw_text=text,
            normalized_text=text.lower(), value={"line_item": True, "unit_price": "$4,500.00"} if aid == "l" else {},
            entity_keys=[], authority_class=AuthorityClass.vendor_quote, confidence=0.8,
            review_status=ReviewStatus.needs_review, parser_version="t",
            source_refs=[SourceRef(id="s" + aid, artifact_id="x", artifact_type=ArtifactType.pdf, filename=filename,
                                   locator={"page": 0, "section_path": section}, extraction_method="t",
                                   parser_version="t")])

    def _doc(filename, sections):
        return [_a("l", "Onsite installation | 1 | $4,500.00", filename, sections[0]),
                _a("d", "Complete billing tasks", filename, sections[-1])]

    # A SOW by its sections, even under a neutral file name.
    sow = _doc("ready for signature.pdf", [["PROJECT DESCRIPTION"], ["PROVIDER RESPONSIBILITIES"]])
    _tag_vendor_quote_document(sow)
    assert not any(a.value.get("document_kind") for a in sow)
    # A purchase order by its title; a quote that mentions a PO is still a quote.
    po = _doc("order.pdf", [["PURCHASE ORDER"]])
    _tag_vendor_quote_document(po)
    assert not any(a.value.get("document_kind") for a in po)
    quote = _doc("CDW Quote for PO 4500.pdf", [["QUOTE CONFIRMATION"]])
    _tag_vendor_quote_document(quote)
    assert all(a.value.get("document_kind") == "vendor_quote_bom" for a in quote)


def test_list_item_is_never_a_conversational_turn() -> None:
    from app.core.hybrid_summary_transcript import retag_conversational_to_meta
    from app.core.schemas import ArtifactType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef

    def _a(aid, text, value, block_kind):
        return EvidenceAtom(
            id=aid, project_id="p", artifact_id="x", atom_type=AtomType.scope_item, raw_text=text,
            normalized_text=text.lower(), value=value, entity_keys=[], authority_class=AuthorityClass.vendor_quote,
            confidence=0.8, review_status=ReviewStatus.needs_review, parser_version="t",
            source_refs=[SourceRef(id="s" + aid, artifact_id="x", artifact_type=ArtifactType.pdf, filename="x.pdf",
                                   locator={"page": 1, "block_kind": block_kind}, extraction_method="t",
                                   parser_version="t")])

    item = _a("li", "Complete billing tasks", {"kind": "bullet", "depth": 1}, "bullet_list")
    turn = _a("tt", "Hey, how you doing? Been a while.", {"kind": "paragraph"}, "paragraph")
    retag_conversational_to_meta([item, turn])
    assert item.atom_type == AtomType.scope_item
    assert turn.atom_type == AtomType.deal_metadata

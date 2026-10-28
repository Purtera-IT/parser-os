"""Wave-5 wiring gaps on the final dev build (audit 2026-10-28). Synthetic deals only."""
from types import SimpleNamespace as NS

import pytest

from app.core import c3_brief, c3_type_base
from c3_record import estimating_calc_by_content


def _ref(art, filename, sheet=None, artifact_type="xlsx"):
    loc = {"sheet": sheet, "row": 1} if sheet else {}
    return NS(artifact_id=art, filename=filename, artifact_type=artifact_type, locator=loc)


def _line(i, art, text, kind="scope_item", filename="", sheet=None, value=None, artifact_type="xlsx", keys=()):
    return NS(id=f"a{i}", atom_type=kind, raw_text=text, review_status="approved", artifact_id=art,
              entity_keys=list(keys), review_flags=[], value=value or {},
              source_refs=[_ref(art, filename, sheet, artifact_type)])


# ---- 4. our own estimating calculator is cut by what it says, not only by its name

CALC = ["LABOR GM: Install 1D loc | 0.35: 220 | HRS: 1.3", "UNIT: COST |", "TOTAL: COST |",
        "LABOR GM: Install 2 Post Data Rack | 0.35: 16", "Materials inflated by 10% for bidding | Markup: 0.25",
        "Material cost | 1,200.00", "Labor cost | 4,000.00", "Install 220 data locations"]
VENDOR = ["Item | Description | Qty | Unit Price | Extended Price", "Cat6 cable 1000ft | 4 | $210.00 | $840.00",
          "Patch panel 24 port | 2 | $95.00 | $190.00", "Subtotal | $1,030.00", "Freight | $45.00"]


def test_a_calculator_is_known_by_its_margin_and_cost_lines():
    assert estimating_calc_by_content(CALC, "The Job Pricing Sheet ADJ 09-14-26.xls")
    assert not estimating_calc_by_content(VENDOR, "Vendor Pricing Sheet.xlsx")
    # The same words in a Word document are a SOW talking about pricing, not a workbook.
    assert not estimating_calc_by_content(CALC, "Statement of Work v2.docx")
    assert estimating_calc_by_content(CALC, "copy-without-extension", spreadsheet=True)


def test_a_calculator_not_named_deal_kit_is_cut_from_the_brief_and_says_why(monkeypatch):
    calc = [_line(i, "art_calc", t, filename="Job Pricing Sheet ADJ.xls", sheet="Labor") for i, t in enumerate(CALC)]
    sow = [_line(100, "art_sow", "Install 220 data locations and 433 WAP locations.", filename="Job SOW v2.docx",
                 artifact_type="docx")]
    fps = {"art_sow": {"filename": "Job SOW v2.docx"}, "art_calc": {"filename": "Job Pricing Sheet ADJ.xls"}}
    c3_type_base.set_deal(None)
    docs = c3_type_base.deal_documents(calc + sow, fps)
    kinds = {d["doc_id"]: (d["kind"], d.get("kind_by")) for d in docs}
    assert kinds == {"art_calc": ("estimating_calc", "content"), "art_sow": ("file", None)}
    c3_type_base.set_deal(docs)
    try:
        assert c3_type_base.skipped(calc[0], {}) and not c3_type_base.skipped(sow[0], {})
        assert c3_type_base.doc_kind(calc[0]) == "estimating_calc"
        monkeypatch.setitem(c3_type_base._DEAL, "document_job", [])
        atoms = calc + sow
        reads = {a.id: {"col:admission": ["keep", 0.95]} for a in atoms}
        b = c3_brief.build(atoms, readings={"reads": reads, "links": [], "deal": {}})
        assert [x["id"] for v in b["sections"].values() for x in v] == ["a100"]
        gate = {d["artifact_id"]: d for d in b["documents"]}
        assert gate["art_calc"]["feeds"] is False and gate["art_calc"]["kind_by"] == "content"
        assert "our pricing workbook: estimating_calc (by its content)" == gate["art_calc"]["why_not"]
        reasons = {r[0]: r[1] for r in b["line_decisions"]["rows"]}
        assert {reasons[a.id] for a in calc} == {"our_estimating_calc_by_content"}
    finally:
        c3_type_base.set_deal(None)


def test_a_workbook_the_customer_sent_is_their_paper(monkeypatch):
    calc = [_line(i, "art_calc", t, filename="Their Budget.xlsx", sheet="Budget") for i, t in enumerate(CALC)]
    monkeypatch.setattr(c3_type_base, "_origin", lambda *a, **k: {"direction": "customer_inbound"})
    c3_type_base.set_deal(None)
    docs = c3_type_base.deal_documents(calc, {"art_calc": {"filename": "Their Budget.xlsx"}})
    assert docs[0]["kind"] == "file" and "kind_by" not in docs[0]

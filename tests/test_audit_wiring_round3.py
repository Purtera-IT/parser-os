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


# ---- 5. the quote's after-hours / lift wording reads the job's work, not its pricing or exclusions

def test_pricing_terms_rate_rows_and_exclusions_leave_the_quotes_work_set(monkeypatch):
    seen = {}
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")
    monkeypatch.setattr(c3_type_base, "_DEAL", {"reads": {}, "deal": {}, "view": "planning"})

    def fake(path, body, timeout):
        seen.update(body=body)
        return {"job_type": "cabling", "lines": []}

    monkeypatch.setattr(c3_type_base, "_request", fake)
    S = NS
    atoms = [
        S(id="w1", atom_type="task", raw_text="Validate the new circuit during business hours", value={}, source_refs=[]),
        S(id="p1", atom_type="pricing_assumption", raw_text="After hours work will be billed at a 50% uplift.",
          value={}, source_refs=[]),
        S(id="r1", atom_type="service_line", raw_text="After Hours | $238.64 per hour", value={}, source_refs=[]),
        S(id="x1", atom_type="exclusion", raw_text="Lift rental or other specialty access equipment.", value={},
          source_refs=[]),
        S(id="o1", atom_type="scope_item", raw_text="Ceiling work above 12 feet", value={},
          source_refs=[S(locator={"section_path": ["SOW", "Out of Scope services include"]})]),
    ]
    brief = {"deal": {}, "sections": {"scope_of_work": [{"id": a.id} for a in atoms]},
             "job_details": [{"text": "After Hours | $238.64 per hour", "cites": ["r1"]},
                             {"text": "Validate the new circuit", "cites": ["w1"]}]}
    q = c3_type_base.quote(atoms, brief, deal_id="d1")
    sent = seen["body"]["deal"]["c3_brief"]
    assert sent["sections"]["scope_of_work"] == [{"id": "w1"}]
    assert [j["cites"] for j in sent["job_details"]] == [["w1"]]
    assert set(sent["not_work"]) == {"p1", "r1", "x1", "o1"}
    # Every line still rides to the server; only the work set changes, and the quote says so.
    assert {r["id"] for r in seen["body"]["deal"]["atoms"]} == {a.id for a in atoms}
    assert q["not_work"]["count"] == 4 and q["not_work"]["ids"] == ["o1", "p1", "r1", "x1"]
    # The day-type and access heads ride along when the read answers them.
    assert {"read:work_window", "read:access_need"} <= set(c3_type_base.QUOTE_LINE_HEADS)


# ---- 6. a site key must be a place; kit lines and parser-made lines name no quote site

def _site(aid, text, keys, art="art_doc", filename="sow.docx"):
    return NS(id=aid, raw_text=text, atom_type="physical_site", entity_keys=list(keys), artifact_id=art,
              review_flags=[], value={}, decision_provenance=None,
              source_refs=[NS(artifact_type="docx", filename=filename, locator={}, extraction_method="docx")])


def test_row_numbers_codes_and_kit_rows_are_no_site_keys():
    from app.core.site_keys_follow_heads import settle

    row = _site("r", "34: Hubbell HJU6W CAT 6,8 POS,UNIV, White 0 0", ["site:34"])
    code = _site("k", "DEPT/CLASS: 290-9420 | Description: 10ft Step Ladder", ["site:290_9420"],
                 art="art_kit", filename="010999 Deal Kit.xlsx")
    div = _site("d", "27, 28.", ["site:27"])
    store = _site("s", "Store #34 - Austin", ["site:34_austin"])
    unkeyed_row = _site("u", "12: Patch cords", [])
    addr = _site("a", "1200 Harbor Road", ["site:1200"])
    types = {x.id: ["physical_site", 0.92] for x in (row, code, div, store, unkeyed_row, addr)}
    c3_type_base.set_deal(None)
    rec = settle([row, code, div, store, unkeyed_row, addr], types)
    assert sorted(rec["kept"]) == ["site:1200", "site:34_austin"]
    assert {d["site"] for d in rec["dropped"]} == {"site:34", "site:290_9420", "site:27"}
    assert unkeyed_row.entity_keys == [], "a row number is never minted as a site key"
    c3_type_base._DEAL["site_keys"] = rec  # as the brief stores it
    sites = c3_type_base.quote_sites([row, code, div, store, unkeyed_row, addr])
    c3_type_base.set_deal(None)
    assert sorted(s["site"] for s in sites) == ["site:1200", "site:34_austin"]


def test_a_parser_made_line_is_no_statement_for_the_quote(monkeypatch):
    seen = {}
    monkeypatch.setenv("SOWSMITH_C3_QUOTE", "on")
    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3")
    monkeypatch.setattr(c3_type_base, "_DEAL", {"reads": {}, "deal": {}, "view": "planning"})
    monkeypatch.setattr(c3_type_base, "_request", lambda p, b, timeout: seen.update(body=b) or {"lines": []})
    atoms = [NS(id="gap", atom_type="open_question", review_flags=["declared_scope"], value={}, source_refs=[],
                raw_text="Customer documents declare 50 locations; 15 identified in the parsed files."),
             NS(id="w1", atom_type="task", review_flags=[], value={}, source_refs=[],
                raw_text="Survey the one office and rack the firewall")]
    c3_type_base.quote(atoms, {"deal": {}, "sections": {"scope_of_work": [{"id": "w1"}, {"id": "gap"}]}})
    rows = {r["id"]: r for r in seen["body"]["deal"]["atoms"]}
    assert rows["gap"]["parser_made"] and rows["gap"]["left_out"]
    assert not rows["w1"].get("left_out")

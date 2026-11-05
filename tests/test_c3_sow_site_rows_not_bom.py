"""A store list's rows are sites, not parts: the customer-BOM rule skips lines the heads type as a place
(010035, 2026-11-05: 168 Goodyear store rows were listed as customer-supplied materials)."""
from __future__ import annotations

from app.core import c3_sow as m


def _f(i, text, type_):
    return {"id": i, "section": "scope_of_work", "text": text, "under": "", "type": type_, "when": "", "supplier": None}


def test_store_rows_are_not_customer_materials():
    fs = [_f("s1", "1139 | 1663 Penfield Road | Rochester | NY | 14625-2549", "physical_site"),
          _f("s2", "4226 | 3209 Kirby Drive | Pearland | TX | 77584-8466", "physical_site"),
          _f("b1", "12 | PA-440 firewall", "bom_line")]
    c = {"customer_materials": []}
    m._customer_bom_rows(c, fs, lambda i: True)
    assert [x["cites"] for x in c["customer_materials"]] == [["b1"]]


def test_parts_rows_still_go_to_the_customer():
    fs = [_f("b1", "12 | PA-440 firewall", "bom_line"), _f("b2", "4: SFP-10G-SR", "")]
    c = {}
    m._customer_bom_rows(c, fs, lambda i: i == "b1")
    assert [x["text"] for x in c["customer_materials"]] == ["12 x PA-440 firewall", "4 x SFP-10G-SR"]

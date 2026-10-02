"""A spreadsheet does not ask for a survey by naming one.

Re-parsed Deal Kits were staged "awaiting site survey", pinned on the
"Deal Kit-1" and "SELL RATES" sheets. Nothing there asked for a survey: a
rate row joined the matrix's "Request" column to a priced "Site Survey"
line, and the Deal Kit listed "Site Survey" as a service with a "Required"
box. The deriver read the joined row as one sentence.
"""
from __future__ import annotations

from pathlib import Path

import openpyxl

from app.core.deal_state import read_deal_state
from app.core.schemas import ArtifactType


class _Ref:
    def __init__(self, sheet):
        self.artifact_type = ArtifactType.xlsx
        self.locator = {"sheet": sheet}
        self.filename = "Deal Kit.xlsx"


class _A:
    def __init__(self, text, sheet=None):
        self.raw_text = text
        self.artifact_id = "art_x"
        self.source_refs = [_Ref(sheet)] if sheet else []


SHEET_ROWS = [
    _A("SELL RATES | Country: Canada | Request: 55 | Site Survey 2 hr. min: 96", "SELL RATES"),
    _A("Service: Site Survey | Required: Yes | Hours: 4", "Deal Kit-1"),
    _A("Site Survey Required?: Yes", "Deal Kit-1"),
    _A("Site Survey | Per site | Please quote per location | 250", "Deal Kit-1"),
]


def test_sheet_rows_naming_a_survey_do_not_stage_the_deal():
    st = read_deal_state(SHEET_ROWS)
    assert st.get("stage") is None, st.get("stage")
    assert st.get("next_step") is None


def test_a_sheet_cell_that_asks_in_a_sentence_still_counts():
    ask = _A("Notes: Customer wants us to schedule a site survey before we can quote.", "Deal Kit-1")
    st = read_deal_state(SHEET_ROWS + [ask])
    assert st.get("stage").value == "awaiting site survey"
    assert st.get("stage").evidence_atoms == [ask]


def test_prose_unchanged():
    st = read_deal_state([_A("Customer wants us to schedule a site survey before we can quote.")])
    assert st.get("stage").value == "awaiting site survey"


def test_compile_deal_kit_with_survey_rate_rows_derives_no_state(tmp_path: Path):
    from app.core.compiler import compile_project

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "SELL RATES"
    ws.append(["Country", "Request", "Site Survey 2 hr. min", "L1 Technician 2 hr. min",
               "L2 Technician 2 hr. min", "Networking L1 2 hr. min"])
    for c, v in (("United States", 50), ("Canada", 55), ("Mexico", 45), ("Brazil", 60)):
        ws.append([c, v, v + 40, v + 10, v + 20, v + 30])
    ws2 = wb.create_sheet("Deal Kit-1")
    ws2.append(["Customer", "OxBlue"])
    ws2.append(["OPPTY #", "010246"])
    ws2.append([])
    ws2.append(["Service", "Required", "Hours", "Rate"])
    ws2.append(["Site Survey", "Yes", 4, 96])
    ws2.append(["Install", "Yes", 40, 85])
    ws2.append(["Budgetary ROM", "Yes", 1, 5000])
    wb.save(tmp_path / "Deal Kit.xlsx")
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    states = [a.raw_text for a in r.atoms if a.atom_type.value == "deal_state"]
    assert not [s for s in states if "survey" in s.lower() and "await" in s.lower()], states

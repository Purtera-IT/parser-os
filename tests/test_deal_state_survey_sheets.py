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


# ── 000132: a priced survey line on a Deal Kit v2 sell-rates sheet ──────────


class _Typed(_A):
    def __init__(self, text, sheet, atom_type="service_line", value=None):
        super().__init__(text, sheet)
        self.atom_type = atom_type
        self.value = value or {}


def test_priced_survey_row_with_a_requirement_sentence_does_not_stage_the_deal():
    # A rate row whose description is a sentence ("required for each
    # location") passed the per-cell test and staged "awaiting site survey"
    # with "next step: site survey" -- both lines from the same evidence.
    rows = [
        _Typed("Sell Rates | Service: Site Survey | Description: Per site survey required for each "
               "location before deployment | Sell Rate: 250", "Sell Rates",
               value={"kind": "rate_card_row", "money_keys": ["money:250"]}),
        _A("Site Survey | Notes: Survey needed before deployment at each new location | 96", "SELL RATES"),
    ]
    st = read_deal_state(rows)
    assert st.get("stage") is None and st.get("next_step") is None, st.lines


def test_compile_deal_kit_v2_sell_rates_with_survey_terms_derives_no_state(tmp_path: Path):
    from app.core.compiler import compile_project

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Deal Kit"
    ws.append(["Customer", "Acme Foods"])
    ws.append(["OPPTY #", "000132"])
    ws2 = wb.create_sheet("Sell Rates")
    ws2.append(["Service", "Description", "Sell Rate", "Unit"])
    ws2.append(["Site Survey", "Per site survey required for each location before deployment", 250, "per site"])
    ws2.append(["L1 Technician", "Onsite technician, 2 hour minimum", 65, "per hour"])
    ws2.append(["L2 Technician", "Network technician, 2 hour minimum", 85, "per hour"])
    wb.save(tmp_path / "Deal Kit v2.xlsx")
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    states = [a.raw_text for a in r.atoms if a.atom_type.value == "deal_state"]
    assert not [s for s in states if "survey" in s.lower()], states
    # The priced survey line itself is still an atom.
    assert any("Site Survey" in a.raw_text and "250" in a.raw_text for a in r.atoms)

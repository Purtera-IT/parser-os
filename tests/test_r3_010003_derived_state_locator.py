"""A derived deal-state line names its evidence, not a cell (010003, O8).

010003's "awaiting site survey" lines were pinned to the Deal Kit's SELL
RATES band: the derived atom was a deep copy of its first evidence atom, so
it carried that row's sheet/row locator and a reader was sent to a rate
cell that says nothing of the kind. Now a row under a rate section of any
sheet is not survey evidence, a sentence outranks a sheet row as the
anchor, and the derived atom's locator says it is derived and from what.
"""
from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from app.core.deal_state import read_deal_state
from app.core.schemas import ArtifactType


class _Ref:
    def __init__(self, sheet, section_path):
        self.artifact_type = ArtifactType.xlsx
        self.locator = {"sheet": sheet, "row": 14, "section_path": section_path}


class _A:
    def __init__(self, text, sheet=None, section_path=None):
        self.raw_text = text
        self.artifact_id = "art_x"
        self.value = {}
        self.source_refs = [_Ref(sheet, section_path or [sheet])] if sheet else []


def test_a_row_under_a_rate_band_of_a_deal_kit_is_not_survey_evidence() -> None:
    row = _A("Site Survey: Per site survey required before each install is scheduled | Rate: 96",
             "Deal Kit", ["Deal Kit", "SELL RATES"])
    assert read_deal_state([row]).get("stage") is None
    notes = _A("Notes: Per site survey required before each install is scheduled",
               "Deal Kit", ["Deal Kit", "Customer Notes"])
    assert read_deal_state([notes]).get("stage").value == "awaiting site survey"


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def test_derived_state_atom_has_no_cell_locator(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    deal = tmp_path / "deal"
    deal.mkdir()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Deal Kit"
    ws.append(["Field", "Value"])
    ws.append(["Customer", "Acme Corp"])
    ws.append(["Notes", "Customer wants us to schedule a site survey before we can quote."])
    wb.save(deal / "A Deal Kit.xlsx")
    (deal / "B call notes.md").write_text(
        "Spoke with Dana at Acme.\n\nThey asked us to schedule a site survey next week before we can quote.\n",
        encoding="utf-8",
    )
    r = compile_project(deal, project_id="p", allow_errors=True, use_cache=False)
    derived = [a for a in r.atoms if str(getattr(a.atom_type, "value", a.atom_type)) == "deal_state"]
    assert derived, [a.raw_text for a in r.atoms]
    for a in derived:
        for ref in a.source_refs:
            assert ref.locator.get("derived") is True, ref.locator
            assert not ({"sheet", "row", "cell", "page", "line_start"} & set(ref.locator)), ref.locator
        assert a.value["evidence_atom_ids"]
    stage = next(a for a in derived if a.value["key"] == "stage")
    # the sentence, not the sheet row, anchors the line
    assert stage.source_refs[0].filename == "B call notes.md", stage.source_refs[0].filename

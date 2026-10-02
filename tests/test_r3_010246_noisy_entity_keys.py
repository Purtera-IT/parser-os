"""Entity keys that named nothing on 010246's Deal Kit.

Same class as #268's device_alias_context work -- a word that is an entity
only in the right sentence:

* a JOB is not a place or a customer: "Field Tech" (a capitalised run ending
  in "tech", like "Virginia Tech") was site:field_tech AND customer:field_tech;
* a numbered staffing slot is not a site, a customer or a person: "Niagara
  Tech #1", "Owner: Niagara Tech #2";
* a vendor surface that is also an ordinary word needs a cue from the
  vendor's world: "Niagara Tech" was vendor:tridium, "est 3 hrs" Edwards EST;
* a person is named by the line, not by its heading: "Project Manager: PK"
  under the sheet title "OxBlue Pumphouse - Project Gantt" was
  stakeholder:gantt_ox.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from openpyxl import Workbook

from app.core.entity_extraction import extract_keys
from app.domain import load_domain_pack

PACK = load_domain_pack("security_camera")


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


@pytest.mark.parametrize("text", [
    "Niagara Tech #1 | Field Tech | Onsite | United States | 40 | 55",
    "Resource: Field Tech | Type: Onsite | Hours: 40 | Rate: 55",
    "Phase: Install | Task: Cable pulls | Technician: Field Tech",
    "Owner: Niagara Tech #2",
    "Engineer (est 3 hrs per site) 99 $9,504",
])
def test_no_noise_keys(text: str) -> None:
    keys = extract_keys(text, pack=PACK)
    noisy = [k for k in keys if k.startswith(("site:", "customer:", "vendor:", "stakeholder:"))]
    assert not noisy, (text, keys)


@pytest.mark.parametrize("text,key", [
    ("Upgrade the Niagara N4 supervisor and add two JACE controllers", "vendor:tridium"),
    ("Replace the Edwards EST3 fire alarm panel", "vendor:edwards_est"),
    ("Install cameras at Virginia Tech", "customer:virginia_tech"),
])
def test_real_names_still_tag(text: str, key: str) -> None:
    assert key in extract_keys(text, pack=PACK), extract_keys(text, pack=PACK)


def test_heading_does_not_name_a_person(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    deal = tmp_path / "deal"
    deal.mkdir()
    wb = Workbook()
    ws = wb.active
    ws.title = "Gantt"
    ws.append(["OxBlue Pumphouse - Project Gantt"])
    ws.append([])
    for k, v in (("Customer", "OxBlue"), ("OPPTY #", "010246"), ("Project Manager", "PK")):
        ws.append([k, v])
    ws.append([])
    ws.append(["Phase", "Task", "Owner", "Hours", "Start", "End"])
    for ph, t, o, h in (("Planning", "Kickoff meeting", "PK", 4), ("Install", "Cable pulls", "Niagara Tech #1", 120),
                        ("Install", "Rack and stack", "Niagara Tech #2", 40)):
        ws.append([ph, t, o, h, dt.date(2025, 3, 3), dt.date(2025, 3, 10)])
    wb.save(deal / "Deal Kit.xlsx")
    r = compile_project(deal, project_id="p", allow_errors=True, use_cache=False)
    bad = sorted({k for a in r.atoms for k in a.entity_keys
                  if k in ("stakeholder:gantt_ox", "stakeholder:niagara_tech", "site:niagara_tech",
                           "customer:niagara_tech", "vendor:tridium", "site:field_tech", "customer:field_tech")})
    assert not bad, bad

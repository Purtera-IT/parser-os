"""A country is a rate dimension, never a stakeholder.

Deal 010246's Deal Kit Gantt (re-run on #268) folded its duplicate rows
under ``stakeholder:united_states``: "Owner: United States" on a Gantt row,
"Lead Technician, Hong Kong" on a rate row passed the name-shape test beside
a role cue, so the country became a person and an entity the rows were
grouped under. A country / state / region is refused wherever a stakeholder
identity is minted (key extraction, key hygiene, structural people).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from openpyxl import Workbook

from app.core.entity_extraction import extract_keys
from app.core.entity_hygiene import filter_entity_keys_for_atom
from app.core.place_names import is_place_name
from app.domain import load_domain_pack


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


@pytest.mark.parametrize("text", [
    "Owner: United States",
    "Lead Technician, United States | Hong Kong | Project Coordinator",
    "Approver: New York | Project Manager",
])
def test_country_text_mints_no_stakeholder_key(text: str) -> None:
    keys = extract_keys(text, pack=load_domain_pack("security_camera"))
    assert not [k for k in keys if k.startswith("stakeholder:")], keys


def test_hygiene_drops_place_stakeholder_keys() -> None:
    class _A:
        raw_text = "Owner: United States"
        value: dict = {}
        source_refs: list = []
        atom_type = "scope_item"

    kept = filter_entity_keys_for_atom(_A(), ["stakeholder:united_states", "stakeholder:hong_kong",
                                              "stakeholder:jordan_lee", "money:55"])
    assert "stakeholder:united_states" not in kept and "stakeholder:hong_kong" not in kept
    assert "stakeholder:jordan_lee" in kept and "money:55" in kept


def test_place_names_are_whole_names_only() -> None:
    assert is_place_name("united_states") and is_place_name("United Arab Emirates")
    assert not is_place_name("georgia_smith") and not is_place_name("jordan_lee")


def test_gantt_country_rows_fold_under_no_stakeholder(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    deal = tmp_path / "deal"
    deal.mkdir()
    wb = Workbook()
    ws = wb.active
    ws.title = "Gantt"
    ws.append(["OxBlue Pumphouse - Project Gantt"])
    ws.append([])
    ws.append(["Phase", "Task", "Owner", "Hours", "Start", "End"])
    for t, h in (("Cable pulls", 120), ("Rack and stack", 40), ("Cable pulls", 120), ("Cable pulls", 120)):
        ws.append(["Install", t, "United States", h, dt.date(2025, 3, 3), dt.date(2025, 3, 10)])
    ws.append([])
    ws.append(["Resource", "Role", "Country", "Hours", "Rate"])
    for _ in range(3):
        ws.append(["Lead Technician", "Project Coordinator", "United States", 40, 55])
    wb.save(deal / "Deal Kit.xlsx")
    r = compile_project(deal, project_id="p", allow_errors=True, use_cache=False)
    for a in list(r.atoms) + list(getattr(r, "suppressed_atoms", None) or []):
        keys = a.get("entity_keys") if isinstance(a, dict) else a.entity_keys
        t = a.get("atom_type") if isinstance(a, dict) else a.atom_type
        t = str(getattr(t, "value", t))
        bad = [k for k in keys or [] if k.startswith("stakeholder:") and is_place_name(k.split(":", 1)[1])]
        assert not bad, (t, bad)
        if t == "stakeholder":
            v = (a.get("value") if isinstance(a, dict) else a.value) or {}
            assert not is_place_name(str(v.get("name") or "")), v

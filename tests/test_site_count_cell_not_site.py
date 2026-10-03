"""A keyed count cell ("Locations: 12") is how many sites, never a site.

Once headed xlsx rows kept their column names (#343), a Deal Kit summary row
read "Locations: 12 | Hours(min): 3 | Total: 36". The "Locations:" label-list
tier of the site catalog read everything after the colon, across the other
columns, as a site name, and the entity backfill minted a physical_site named
"12 Hours min 3 Total 36" -- the deal's only "identified" site. A Location
column that holds a real place ("Location: Kenton, OH | Remote: 15 miles")
read its neighbour cell into the name the same way ("OH Remote 15 miles").
"""

from __future__ import annotations

import io
import contextlib
from pathlib import Path

import openpyxl
from openpyxl.styles import Font

from app.core.schemas import EntityRecord, ReviewStatus
from app.core.site_atom_backfill import backfill_physical_sites_from_entities
from app.core.site_detection import find_authoritative_site_phrases
from app.core.site_plausibility import rejects_as_site

COUNT_ROW = "Locations: 12 | Hours(min): 3 | Total: 36"
PLACE_ROW = "Location: Springfield, IL | Remote: 25 miles"


class _Atom:
    def __init__(self, atom_type: str, text: str, entity_keys=None):
        self.atom_type = atom_type
        self.raw_text = text
        self.text = text
        self.entity_keys = entity_keys or []
        self.source_refs = []
        self.artifact_id = "art1"


def _site_entity(key: str, name: str) -> EntityRecord:
    return EntityRecord(
        id=f"ent_{key}",
        project_id="p1",
        entity_type="site",
        canonical_key=key,
        canonical_name=name,
        aliases=[],
        source_atom_ids=[],
        confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
    )


def _is_site(atom) -> bool:
    return getattr(atom.atom_type, "value", atom.atom_type) == "physical_site"


def test_count_cell_is_not_a_site_catalog_phrase() -> None:
    phrases = find_authoritative_site_phrases([_Atom("scope_item", COUNT_ROW)])
    assert not any("12" in p or "Hours" in p for p in phrases), phrases


def test_location_cell_stops_at_the_next_keyed_cell() -> None:
    rows = [
        _Atom("scope_item", "Location: Kenton, OH | Remote: 15 miles"),
        _Atom("scope_item", "Location: Ripon, WI | Remote: 25 miles"),
    ]
    phrases = find_authoritative_site_phrases(rows)
    assert not any("remote" in p.lower() or "miles" in p.lower() for p in phrases), phrases


def test_shared_site_gate_rejects_a_count_cell_name() -> None:
    key = "site:12_hours_min_3_total_36"
    assert rejects_as_site("12 hours min 3 total 36", COUNT_ROW, key) == "count_not_a_place"
    assert rejects_as_site("", COUNT_ROW, key) == "count_not_a_place"
    # A street number in a Location cell is part of an address, not a count.
    assert rejects_as_site("12 Main St", "Location: 12 Main St | Remote: 5 miles", "site:12_main_st") == ""
    assert rejects_as_site("Springfield, IL", PLACE_ROW, "site:springfield_il") == ""


def test_backfill_skips_count_cell_but_keeps_real_place() -> None:
    count_key = "site:12_hours_min_3_total_36"
    place_key = "site:springfield_il"
    atoms = [
        _Atom("scope_item", COUNT_ROW, [count_key]),
        _Atom("scope_item", PLACE_ROW, [place_key]),
    ]
    entities = [
        _site_entity(count_key, "12 hours min 3 total 36"),
        _site_entity(place_key, "Springfield, IL"),
    ]
    out, n = backfill_physical_sites_from_entities(atoms, entities, project_id="p1")
    sites = [a for a in out if _is_site(a)]
    assert n == 1
    assert [s.value["name"] for s in sites] == ["Springfield, IL"]


def test_headed_count_summary_row_mints_no_site(tmp_path: Path) -> None:
    from app.core.compiler import compile_project

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"
    for row in (("Locations", "Hours(min)", "Total"), (12, 3, 36)):
        ws.append(row)
    for col in "ABC":
        ws[f"{col}1"].font = Font(bold=True)
    wb.save(tmp_path / "Deal Kit.xlsx")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = compile_project(
            tmp_path, project_id="p1", allow_errors=True,
            allow_unverified_receipts=True, use_cache=False,
        )
    row = [a for a in result.atoms if a.raw_text.startswith("Locations: 12")]
    assert row, [a.raw_text for a in result.atoms]
    assert not any(k.startswith("site:") for a in row for k in a.entity_keys)
    assert not [a for a in result.atoms if _is_site(a)]

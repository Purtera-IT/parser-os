"""Site geography + roster-preference regression tests.

Grounded in deal 000043 (Clayton Homes Retail Onsite), which ships two
workbooks that both pass the site-roster gate:

  * ``Clayton Homes CALC.xlsx`` sheet ``TRAVEL`` — a travel-cost model that
    happens to carry Address / City / ST / Zip columns. Its column mapping is
    already correct and is PINNED below so it can never silently regress.
  * ``Exhibit A - Retail Locations.xlsx`` sheets ``In Progress`` / ``Pending`` /
    ``Completed HCs`` — the actual store list, where two real bugs lived:
    a combined ``City/State`` column that landed in the city-only field, and a
    ``Region`` column of internal territory codes ("TEN", "SCA") that was read
    as geography.

Every shape here is generic — a duplicated header, a combined city/state cell,
an internal territory code, a cost sheet competing with a location sheet —
because these are how enterprise rosters ship, not how one customer ships.
"""
from __future__ import annotations

import pytest
from openpyxl import Workbook

from app.core.address_parse import split_city_state_strict
from app.core.schemas import AtomType
from app.parsers.site_roster_extractor import (
    extract_site_roster,
    map_columns_to_fields,
    rank_roster_candidates,
    roster_preference_score,
)
from app.parsers.xlsx_parser import XlsxParser

# ── The real CALC / TRAVEL sheet shape ───────────────────────────
#
# 20 columns, verbatim from the workbook — including the duplicated "City" and
# the trailing blank header, both of which the mapper has to survive.
_TRAVEL_COLUMNS = [
    "HC", "Zone", "Region", "Open/Closed", "Address", "City", "ST", "Zip",
    "Site Name", "City", "State", "Nearest Major Metro", "Metro Population",
    "Distance (Miles)", "Within 25 Miles (Yes/No)", "Remote (Yes/No)",
    "Distance Bucket", "Travel Cost", "Travel Sell", "",
]

#: 38 stores, positional rows (the sheet has duplicate headers, so a dict row
#: would silently collapse two real columns into one).
_TRAVEL_SEED = [
    ("31", "TEN", "4606 Clinton Hwy", "Knoxville", "TN", "37912"),
    ("58", "TEN", "2712 Andrew Johnson Hwy", "Morristown", "TN", "37814"),
    ("268", "TXN", "4401 South Lake Drive", "Texarkana", "TX", "75501"),
    ("291", "TEN", "3110 Winfield Dunn Pkwy", "Kodak", "TN", "37764"),
    ("896", "TXN", "4903 North I-35", "Waco", "TX", "76705"),
    ("1054", "TXN", "5656 South I-45 West", "Corsicana", "TX", "75109"),
    ("2712", "SCA", "16021 Elmore Rd", "Laurinburg", "NC", "28352"),
    ("3301", "SCA", "980 Charlotte Hwy", "Fairview", "NC", "28730"),
]
_TRAVEL_ROWS = [
    [
        hc, "2", region, "Open", street, city, st, zipc,
        f"HC {hc}", city, st, "Nashville", "1300000", "42.5",
        "No", "No", "25-50", "185.00", "240.00", "",
    ]
    for i in range(5)
    for (hc, region, street, city, st, zipc) in (
        (f"{int(h) + i * 10000}", rg, s, c, t, z)
        for (h, rg, s, c, t, z) in _TRAVEL_SEED
    )
][:38]


def test_travel_fixture_is_not_vacuous():
    """Guard the guard: an empty or malformed fixture must fail loudly here."""
    assert len(_TRAVEL_ROWS) == 38
    assert len(_TRAVEL_COLUMNS) == 20
    assert all(len(r) == len(_TRAVEL_COLUMNS) for r in _TRAVEL_ROWS)
    assert len({r[0] for r in _TRAVEL_ROWS}) == 38  # unique HC codes


# ── 1. PIN the CALC / TRAVEL mapping exactly as it behaves today ──


def test_calc_travel_column_mapping_is_pinned():
    """This sheet already maps correctly. Nothing below may change it."""
    field_map = map_columns_to_fields(_TRAVEL_COLUMNS)
    by_header = {_TRAVEL_COLUMNS[i]: f for i, f in sorted(field_map.items())}
    assert by_header == {
        "Address": "street_address",
        "City": "city",
        "ST": "state",
        "Zip": "zip",
        "Site Name": "facility_name",
        "Region": "region",
    }


def test_calc_travel_leftmost_column_wins_a_contested_field():
    """"ST" (col 6) takes ``state``; the later "State" (col 10) falls through.

    The mapper resolves specificity WITHIN a column, then assigns left to right.
    A later, better-worded header does not steal a field already claimed — that
    is what keeps ST->state stable on this sheet.
    """
    field_map = map_columns_to_fields(_TRAVEL_COLUMNS)
    assert field_map[6] == "state"
    assert 10 not in field_map
    assert field_map[5] == "city"
    assert 9 not in field_map


def test_calc_travel_fills_city_for_every_row():
    rows = extract_site_roster(columns=_TRAVEL_COLUMNS, rows=_TRAVEL_ROWS)
    assert len(rows) == 38
    assert sum(1 for r in rows if r.city) == 38
    assert sum(1 for r in rows if r.state) == 38
    assert sum(1 for r in rows if r.street_address) == 38
    assert sum(1 for r in rows if r.zip) == 38
    assert sum(1 for r in rows if r.facility_name) == 38


def test_calc_travel_site_id_comes_from_the_hc_column():
    rows = extract_site_roster(columns=_TRAVEL_COLUMNS, rows=_TRAVEL_ROWS)
    assert sum(1 for r in rows if r.site_id) == 38
    assert all(r.site_id.startswith("HC-") for r in rows)
    assert rows[0].site_id == "HC-31"


# ── 2. Specificity: the most specific synonym wins, not the first ─


@pytest.mark.parametrize(
    "header,expected",
    [
        # The two the bug report names.
        ("City/State", "city_state"),
        ("Site Address", "street_address"),
        # Other collisions the same rule resolves.
        ("City, State", "city_state"),
        ("City / State", "city_state"),
        ("Location ID", "site_id"),
        ("Location Name", "facility_name"),
        ("Facility ID", "site_id"),
        ("Store Number", "site_id"),
        ("Building Name", "facility_name"),
        ("Street Address", "street_address"),
        # And the plain readings still work.
        ("City", "city"),
        ("State", "state"),
        ("ST", "state"),
        ("Address", "street_address"),
        ("Site Name", "facility_name"),
        ("Region", "region"),
    ],
)
def test_header_maps_to_its_most_specific_field(header, expected):
    assert map_columns_to_fields([header]) == {0: expected}


def test_city_state_header_does_not_claim_the_city_field():
    """The reported bug: "City/State" used to map to ``city``.

    ``city`` was checked before ``city_state`` in list order, so "Nashville, TN"
    landed in a city-only field and ``state`` stayed empty. Reordering that one
    pair would only have moved the collision; specificity removes it.
    """
    field_map = map_columns_to_fields(["HC", "City/State", "Address"])
    assert field_map[1] == "city_state"
    assert "city" not in field_map.values()


def test_city_state_and_a_real_city_column_both_map():
    """Exhibit A carries BOTH. Neither may block the other."""
    columns = ["HC", "City/State", "Address", "City", "ST", "Zip"]
    field_map = map_columns_to_fields(columns)
    assert {columns[i]: f for i, f in field_map.items()} == {
        "City/State": "city_state",
        "Address": "street_address",
        "City": "city",
        "ST": "state",
        "Zip": "zip",
    }


def test_short_synonym_must_match_on_a_word_boundary():
    """"st" is the state column, not three letters inside another word."""
    assert map_columns_to_fields(["Estimated Cost"]) == {}
    assert map_columns_to_fields(["ST"]) == {0: "state"}
    assert map_columns_to_fields(["St."]) == {0: "state"}


# ── 3. Region is an org grouping, never geography ────────────────

_REGION_ONLY_COLUMNS = ["HC", "Site Name", "Region", "Address"]
_REGION_ONLY_ROWS = [
    {"HC": "31", "Site Name": "Homes of Knoxville", "Region": "TEN",
     "Address": "4606 Clinton Hwy"},
    {"HC": "268", "Site Name": "Homes of Texarkana", "Region": "TXN",
     "Address": "4401 South Lake Drive"},
    {"HC": "1054", "Site Name": "Homes of Corsicana", "Region": "SCA",
     "Address": "5656 South I-45 West"},
]


def test_region_header_no_longer_maps_to_city_state():
    field_map = map_columns_to_fields(_REGION_ONLY_COLUMNS)
    assert field_map[2] == "region"
    assert "city_state" not in field_map.values()


def test_region_only_sheet_abstains_on_city_and_state():
    """A territory code is not a place. No city/state is the honest answer.

    "TEN" used to be written into ``city_state`` and then read back out as the
    city — a guess, from a column that never named a city at all.
    """
    rows = extract_site_roster(
        columns=_REGION_ONLY_COLUMNS, rows=_REGION_ONLY_ROWS
    )
    assert len(rows) == 3  # non-vacuous: the sheet IS still a roster
    assert [r.region for r in rows] == ["TEN", "TXN", "SCA"]
    assert all(r.city is None for r in rows)
    assert all(r.state is None for r in rows)
    assert all(r.city_state is None for r in rows)
    # The evidence is not dropped — it just stops pretending to be geography.
    assert all(r.street_address for r in rows)


# ── 4. Combined city/state splits only when unambiguous ──────────


@pytest.mark.parametrize(
    "value,city,state",
    [
        ("Nashville, TN", "Nashville", "TN"),
        ("Nashville, Tennessee", "Nashville", "TN"),
        ("Kansas City, MO", "Kansas City", "MO"),
        ("Seattle / WA", "Seattle", "WA"),
        ("Washington, DC", "Washington", "DC"),
    ],
)
def test_unambiguous_city_state_splits(value, city, state):
    assert split_city_state_strict(value) == (city, state)


@pytest.mark.parametrize(
    "value",
    [
        "Nashville",                # no separator: a label, not a pair
        "Springfield, Springfield",  # right side is not a state
        "Nashville, TN, USA",       # more than one separator
        "TEN",                      # a territory code
        "Nashville, T",             # not a state code
        "Nashville, TN 37912",      # shape carries more than a state
        "",
        None,
    ],
)
def test_ambiguous_city_state_abstains(value):
    assert split_city_state_strict(value) == (None, None)


def test_bare_state_names_the_state_and_no_city():
    assert split_city_state_strict("TN") == (None, "TN")
    assert split_city_state_strict("Tennessee") == (None, "TN")


#: Exhibit A's shape, minus the separate City / ST columns, so the ONLY route
#: to a city or a state is the combined ``City/State`` cell. The street values
#: deliberately carry no city or ZIP of their own — nothing but the split can
#: fill these fields.
_EXHIBIT_COLUMNS = ["HC", "FB Page Name", "Region", "City/State", "Address"]
_EXHIBIT_ROWS = [
    {"HC": "31", "FB Page Name": "Clayton Homes of Knoxville", "Region": "TEN",
     "City/State": "Knoxville, TN", "Address": "4606 Clinton Hwy"},
    {"HC": "896", "FB Page Name": "Clayton Homes of Waco", "Region": "TXN",
     "City/State": "Waco, TX", "Address": "4903 North I-35"},
    {"HC": "1054", "FB Page Name": "Clayton Homes of Corsicana", "Region": "TXN",
     "City/State": "Corsicana", "Address": "5656 South I-45 West"},
]


def test_combined_city_state_column_splits_and_keeps_the_raw():
    rows = extract_site_roster(columns=_EXHIBIT_COLUMNS, rows=_EXHIBIT_ROWS)
    assert len(rows) == 3  # non-vacuous
    assert (rows[0].city, rows[0].state) == ("Knoxville", "TN")
    assert (rows[1].city, rows[1].state) == ("Waco", "TX")
    # The raw combined value survives untouched alongside the split.
    assert rows[0].city_state == "Knoxville, TN"
    assert rows[1].city_state == "Waco, TX"
    # Region never becomes geography.
    assert [r.region for r in rows] == ["TEN", "TXN", "TXN"]


def test_city_only_cell_in_a_combined_column_stays_unsplit():
    rows = extract_site_roster(columns=_EXHIBIT_COLUMNS, rows=_EXHIBIT_ROWS)
    assert rows[2].city_state == "Corsicana"   # raw preserved
    assert rows[2].city is None                 # but not promoted to a city
    assert rows[2].state is None


# ── 5. Roster preference is a RANKING, never a filter ────────────

_CALC_CANDIDATE = {
    "filename": "Clayton Homes CALC.xlsx",
    "sheet_name": "TRAVEL",
    "columns": _TRAVEL_COLUMNS,
}
_EXHIBIT_CANDIDATE = {
    "filename": "Exhibit A - Retail Locations.xlsx",
    "sheet_name": "Pending",
    "columns": ["HC", "FB Page Name", "Region", "City/State", "Address", "City", "ST", "Zip"],
}
_RATES_CANDIDATE = {
    "filename": "Clayton Homes CALC.xlsx",
    "sheet_name": "COST RATES",
    "columns": ["Role", "Rate"],
}


def test_exhibit_a_outranks_the_travel_cost_sheet():
    """The real filenames, ranked. This is the whole point of the signal."""
    ranked = rank_roster_candidates([_CALC_CANDIDATE, _EXHIBIT_CANDIDATE])
    by_sheet = {c["sheet_name"]: c for c in ranked}
    assert by_sheet["Pending"]["roster_rank"] == 1
    assert by_sheet["Pending"]["roster_preferred"] is True
    assert by_sheet["TRAVEL"]["roster_rank"] == 2
    assert by_sheet["TRAVEL"]["roster_preferred"] is False
    assert by_sheet["Pending"]["roster_score"] > by_sheet["TRAVEL"]["roster_score"]


def test_ranking_drops_nothing():
    """A preference signal must never remove evidence."""
    candidates = [_CALC_CANDIDATE, _EXHIBIT_CANDIDATE, _RATES_CANDIDATE]
    ranked = rank_roster_candidates(candidates)
    assert len(ranked) == len(candidates)
    assert [c["sheet_name"] for c in ranked] == [c["sheet_name"] for c in candidates]
    # Even the worst candidate keeps every field it arrived with.
    for original, out in zip(candidates, ranked):
        for key, value in original.items():
            assert out[key] == value


def test_ranking_is_order_invariant():
    forward = rank_roster_candidates(
        [_CALC_CANDIDATE, _EXHIBIT_CANDIDATE, _RATES_CANDIDATE]
    )
    backward = rank_roster_candidates(
        [_RATES_CANDIDATE, _EXHIBIT_CANDIDATE, _CALC_CANDIDATE]
    )
    fwd = {c["sheet_name"]: (c["roster_score"], c["roster_rank"]) for c in forward}
    bwd = {c["sheet_name"]: (c["roster_score"], c["roster_rank"]) for c in backward}
    assert fwd == bwd
    assert len(fwd) == 3  # non-vacuous


def test_ranking_is_idempotent():
    once = rank_roster_candidates([_CALC_CANDIDATE, _EXHIBIT_CANDIDATE])
    twice = rank_roster_candidates(once)
    assert once == twice


def test_a_lone_cost_sheet_ranks_first_but_is_not_preferred():
    """Winning a field of one bad candidate does not make it a good roster.

    The travel sheet is the only roster in its own workbook, so it takes rank 1
    there — but its name argues against it, so it is not held out as THE roster.
    It still keeps every atom either way.
    """
    ranked = rank_roster_candidates([_CALC_CANDIDATE])
    assert ranked[0]["roster_rank"] == 1
    assert ranked[0]["roster_score"] < 0
    assert ranked[0]["roster_preferred"] is False


def test_tied_candidates_share_a_rank_and_both_are_preferred():
    a = {"filename": "Site List.xlsx", "sheet_name": "Sheet1"}
    b = {"filename": "Site List.xlsx", "sheet_name": "Sheet1"}
    ranked = rank_roster_candidates([a, b])
    assert [c["roster_rank"] for c in ranked] == [1, 1]
    assert all(c["roster_preferred"] for c in ranked)


def test_preference_markers_are_generic_not_customer_specific():
    """"retail locations" is roster vocabulary; the customer name is not."""
    assert roster_preference_score(filename="Retail Locations.xlsx") > 0
    assert roster_preference_score(filename="Store List.csv") > 0
    assert roster_preference_score(filename="Travel Cost Model.xlsx") < 0
    # A bare customer name carries no preference either way.
    assert roster_preference_score(filename="Clayton Homes.xlsx") == 0.0


def test_name_markers_outweigh_the_structural_bonus():
    """A cost sheet with an address column must not out-rank a location list."""
    cost_with_address = roster_preference_score(
        filename="Travel Cost.xlsx",
        sheet_name="TRAVEL",
        columns=["Site Name", "Address", "City", "ST"],
    )
    bare_location_list = roster_preference_score(filename="Retail Locations.xlsx")
    assert bare_location_list > cost_with_address


# ── 5b. …and the ranking reaches the atoms, still dropping nothing ──


def _site_atoms(path):
    out = XlsxParser().parse_artifact("proj", "art", path)
    atoms = out if isinstance(out, list) else out.atoms
    return [a for a in atoms if a.atom_type == AtomType.physical_site]


def _roster_locators(atoms):
    return [
        ref.locator
        for a in atoms
        for ref in a.source_refs
        if "roster_score" in (ref.locator or {})
    ]


def _write_two_roster_sheets(path):
    """One workbook, two competing rosters: a location list and a cost model."""
    wb = Workbook()
    good = wb.active
    good.title = "Retail Locations"
    good.append(["HC", "FB Page Name", "Region", "City/State", "Address"])
    for row in _EXHIBIT_ROWS:
        good.append([row[c] for c in _EXHIBIT_COLUMNS])
    cost = wb.create_sheet("TRAVEL")
    cost.append(_TRAVEL_COLUMNS)
    for row in _TRAVEL_ROWS:
        cost.append(row)
    wb.save(path)


def test_workbook_ranking_is_stamped_on_atom_provenance(tmp_path):
    path = tmp_path / "Site Roster and Travel Cost.xlsx"
    _write_two_roster_sheets(path)
    atoms = _site_atoms(path)
    assert atoms  # non-vacuous

    by_sheet: dict[str, list[dict]] = {}
    for locator in _roster_locators(atoms):
        by_sheet.setdefault(str(locator.get("sheet")), []).append(locator)
    assert set(by_sheet) == {"Retail Locations", "TRAVEL"}

    assert all(loc["roster_rank"] == 1 for loc in by_sheet["Retail Locations"])
    assert all(loc["roster_preferred"] is True for loc in by_sheet["Retail Locations"])
    assert all(loc["roster_rank"] == 2 for loc in by_sheet["TRAVEL"])
    assert all(loc["roster_preferred"] is False for loc in by_sheet["TRAVEL"])


def test_ranking_never_drops_an_atom_from_the_losing_sheet(tmp_path):
    """The down-ranked travel sheet keeps all 38 of its sites."""
    path = tmp_path / "Site Roster and Travel Cost.xlsx"
    _write_two_roster_sheets(path)
    atoms = _site_atoms(path)
    per_sheet: dict[str, int] = {}
    for locator in _roster_locators(atoms):
        sheet = str(locator.get("sheet"))
        per_sheet[sheet] = per_sheet.get(sheet, 0) + 1
    assert per_sheet["Retail Locations"] == len(_EXHIBIT_ROWS)
    assert per_sheet["TRAVEL"] == len(_TRAVEL_ROWS)


def test_region_reaches_the_atom_value_and_never_becomes_city(tmp_path):
    path = tmp_path / "Site Roster and Travel Cost.xlsx"
    _write_two_roster_sheets(path)
    atoms = _site_atoms(path)
    preferred = [
        a for a in atoms
        for ref in a.source_refs
        if (ref.locator or {}).get("sheet") == "Retail Locations"
    ]
    assert len(preferred) == len(_EXHIBIT_ROWS)  # non-vacuous
    values = {a.value["site_id"]: a.value for a in preferred}
    assert values["HC-31"]["region"] == "TEN"
    assert values["HC-31"]["city"] == "Knoxville"
    assert values["HC-31"]["state"] == "TN"
    assert values["HC-31"]["city_state"] == "Knoxville, TN"
    # The row whose combined cell named no state abstains rather than guessing.
    assert values["HC-1054"]["city"] is None
    assert values["HC-1054"]["state"] is None
    assert values["HC-1054"]["city_state"] == "Corsicana"


# ── 6. Extraction is order-invariant and idempotent ──────────────


def _identity(rows):
    return sorted(
        (r.site_id, r.facility_name, r.street_address, r.city, r.state,
         r.region, r.city_state, r.zip)
        for r in rows
    )


def test_extraction_is_order_invariant():
    forward = extract_site_roster(columns=_TRAVEL_COLUMNS, rows=_TRAVEL_ROWS)
    reversed_rows = extract_site_roster(
        columns=_TRAVEL_COLUMNS, rows=list(reversed(_TRAVEL_ROWS))
    )
    assert len(forward) == 38  # non-vacuous
    assert _identity(forward) == _identity(reversed_rows)


def test_extraction_is_idempotent():
    once = extract_site_roster(columns=_TRAVEL_COLUMNS, rows=_TRAVEL_ROWS)
    twice = extract_site_roster(columns=_TRAVEL_COLUMNS, rows=_TRAVEL_ROWS)
    assert once == twice
    assert len(once) == 38  # non-vacuous


def test_exhibit_shape_extraction_is_order_invariant():
    forward = extract_site_roster(columns=_EXHIBIT_COLUMNS, rows=_EXHIBIT_ROWS)
    backward = extract_site_roster(
        columns=_EXHIBIT_COLUMNS, rows=list(reversed(_EXHIBIT_ROWS))
    )
    assert len(forward) == 3  # non-vacuous
    assert _identity(forward) == _identity(backward)

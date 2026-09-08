"""A site roster pasted as a table into a HubSpot note is read as a roster.

Deal 010310: a PM pasted a five-column, three-row site table into a note.
HubSpot's HTML->text conversion collapsed every tag to a space, so it reached
parser-os as one 448-byte line. The deal published FOUR sites for three rows --
one real site split across two, all four named from city+province+postal
fragments -- and the site named "Malport", the only name that is not also a
city and so the only one that cannot be re-derived from geography, was lost.

The extractor was never wrong. ``site_roster_extractor`` already maps every one
of those headers; it was simply unreachable from a note.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from app.parsers.hubspot_note_parser import HubspotNoteParser, parse_hubspot_note_text
from app.parsers.note_site_roster import (
    find_delimited_table,
    site_entity_key,
    site_roster_from_note_lines,
)

ANOVA_LINES = [
    "Site Name\tAddress\tCity\tProvince\tPostal Code",
    "Brampton\t55 Devon rd\tBrampton\tON\tL6T 5B6",
    "Malport\t7675 Torbram Rd\tMississauga\tON\tL4T 3L8",
    "Mississauga\t7505 Bramalea Rd\tMississauga\tON\tL5S 1C4",
    "Just need a quick quote for 5 tanks per site, UTM-Huba RLS install in Canada.",
]


def _note(body_lines: list[str]) -> str:
    return "\n".join(
        [
            "HubSpot Note: Site Name",
            "HubSpot Note ID: 116573225011",
            "Date: 2026-09-08T18:53:58.144Z",
            "Author: Megan Blevins",
            "Author-Email: megan@purtera-it.com",
            "",
        ]
        + body_lines
    )


def _parse(body_lines: list[str]):
    d = Path(tempfile.mkdtemp()) / "010310-hs-note-116573225011-Site Name.txt"
    d.write_text(_note(body_lines))
    return HubspotNoteParser().parse_artifact_full("proj", "art", d)


def _sites(out):
    return [a for a in out.atoms if a.atom_type.value == "physical_site"]


def test_finds_the_table_and_leaves_the_prose_line_out():
    columns, rows, header_index = find_delimited_table(ANOVA_LINES)
    assert header_index == 0
    assert columns == ["Site Name", "Address", "City", "Province", "Postal Code"]
    assert len(rows) == 3, "the trailing prose line is not a table row"


def test_roster_has_one_row_per_declared_site():
    roster_rows, _cols, _rows = site_roster_from_note_lines(ANOVA_LINES)
    assert len(roster_rows) == 3


def test_site_names_come_from_the_site_name_column():
    roster_rows, _c, _r = site_roster_from_note_lines(ANOVA_LINES)
    names = sorted(r.facility_name for r in roster_rows)
    assert names == ["Brampton", "Malport", "Mississauga"]


def test_malport_survives():
    """The regression in one assertion.

    "Malport" is the only site name here that is not also its city, so it is the
    only one that cannot be reconstructed from geography. Losing it is what made
    the old roster look plausible while being wrong.
    """
    roster_rows, _c, _r = site_roster_from_note_lines(ANOVA_LINES)
    malport = [r for r in roster_rows if r.facility_name == "Malport"]
    assert len(malport) == 1
    assert malport[0].street_address == "7675 Torbram Rd"
    assert malport[0].city == "Mississauga"


def test_one_row_never_becomes_two_sites():
    """7675 Torbram Rd, Mississauga L4T published as BOTH "mississauga on l4t"
    and "torbram rd mississauga on l4t" -- two sites from one row."""
    out = _parse(ANOVA_LINES)
    keys = [k for a in _sites(out) for k in a.entity_keys]
    assert len(keys) == len(set(keys)), f"duplicate site keys: {keys}"
    assert len(keys) == 3


def test_end_to_end_atoms_carry_full_geography():
    out = _parse(ANOVA_LINES)
    sites = _sites(out)
    assert len(sites) == 3
    for a in sites:
        v = a.value
        assert v.get("address"), f"{v.get('name')} has no address"
        assert v.get("city"), f"{v.get('name')} has no city"
        assert v.get("state"), f"{v.get('name')} has no state"
        assert v.get("zip"), f"{v.get('name')} has no postal code"


def test_entity_key_is_the_declared_name_not_geography():
    out = _parse(ANOVA_LINES)
    keys = sorted(k for a in _sites(out) for k in a.entity_keys)
    assert keys == ["site:brampton", "site:malport", "site:mississauga"]


def test_a_flattened_note_yields_no_roster_and_does_not_crash():
    """The pre-fix artifact shape. No table to find -- abstain, never guess."""
    flat = (
        "Site Name Address City Province Postal Code Brampton 55 Devon rd "
        "Brampton ON L6T 5B6 Malport 7675 Torbram Rd Mississauga ON L4T 3L8 "
        "Mississauga 7505 Bramalea Rd Mississauga ON L5S 1C4 Just need a quick "
        "quote for 5 tanks per site."
    )
    out = _parse([flat])
    assert _sites(out) == []


def test_prose_note_is_untouched():
    out = _parse(["Customer called about the install.", "They want it next week."])
    assert _sites(out) == []


def test_body_lines_is_additive_and_body_still_flattens():
    parsed = parse_hubspot_note_text(_note(ANOVA_LINES))
    assert parsed["body_lines"] == ANOVA_LINES
    assert "\n" not in parsed["body"], "body stays flattened for prose consumers"


def test_a_non_roster_table_is_not_a_roster():
    """The gate is site_roster_extractor's, not ours."""
    rows = [
        "Item\tQty\tUnit Price",
        "UTM sensor\t5\t420.00",
        "Bracket\t5\t18.00",
        "Cable\t15\t3.50",
    ]
    roster_rows, _c, _r = site_roster_from_note_lines(rows)
    assert roster_rows == []


def test_single_column_lines_are_not_a_table():
    assert find_delimited_table(["one", "two", "three"]) is None


def test_site_entity_key_abstains_when_nothing_names_the_row():
    class _Empty:
        site_id = ""
        facility_name = ""
        street_address = ""

    assert site_entity_key(_Empty()) == ""

"""The SOW Locations table gives one site, with its city line (010003, O7).

The 010003 SOW's Locations table ("Facility" | "Address", the address cell
wrapping its "City, ST ZIP" line under the street) still produced
"facility: NYC Office | address: 40 10th Ave Fl 4" after the all-caps fix:

* drawn with rules, fitz's site-roster fallback emitted a second copy of the
  row built from the street alone (it deduped on site id, and the roster has
  none);
* unruled with a mixed-case city line, the region read as two side-by-side
  boxes (the left/right width test hinged on the city line's case) and no
  site came out at all.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.schemas import AtomType

fitz = pytest.importorskip("fitz")


def _build(path: Path, *, city: str, ruled: bool) -> None:
    doc = fitz.open()
    p = doc.new_page(width=612, height=792)
    p.insert_text((36, 60), "Statement of Work", fontsize=12, fontname="hebo")
    p.insert_text((36, 80), "This Statement of Work is made between CDW Direct, LLC and Customer.", fontsize=10)
    p.insert_text((36, 104), "Locations", fontsize=12, fontname="hebo")
    p.insert_text((36, 124), "Facility", fontsize=9, fontname="hebo")
    p.insert_text((200, 124), "Address", fontsize=9, fontname="hebo")
    p.insert_text((36, 138), "NYC Office", fontsize=9)
    p.insert_text((200, 138), "40 10th Ave Fl 4", fontsize=9)
    p.insert_text((200, 150), city, fontsize=9)
    if ruled:
        for y in (114, 128, 155):
            p.draw_line((30, y), (450, y))
        for x in (30, 195, 450):
            p.draw_line((x, 114), (x, 155))
    p.insert_text((36, 190), "Project Management", fontsize=12, fontname="hebo")
    p.insert_text((36, 210), "CDW will provide project management for the duration of the project.", fontsize=10)
    doc.save(str(path))
    doc.close()


@pytest.mark.parametrize("ruled", [False, True])
@pytest.mark.parametrize("city", ["NEW YORK, NY 10014", "New York, NY 10014"])
def test_locations_table_gives_one_site_with_its_city(tmp_path: Path, city: str, ruled: bool) -> None:
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    pdf = tmp_path / "Signed SOW.pdf"
    _build(pdf, city=city, ruled=ruled)
    out = OrbitBriefPdfParser().parse(pdf)
    atoms = list(getattr(out, "atoms", out))
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1, [(a.atom_type.value, a.raw_text) for a in atoms]
    assert city in sites[0].raw_text and "40 10th Ave Fl 4" in sites[0].raw_text
    assert (sites[0].value["state"], sites[0].value["zip"]) == ("NY", "10014")

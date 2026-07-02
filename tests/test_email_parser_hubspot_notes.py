from __future__ import annotations

from app.core.schemas import AtomType
from app.parsers.email_parser import EmailParser


def test_hubspot_note_txt_emits_physical_site_for_compact_state_zip(tmp_path):
    p = tmp_path / "010058-hs-note-111645120815-GECKO ROBOTICS.txt"
    p.write_text(
        "\n".join(
            [
                "HubSpot Note: GECKO ROBOTICS",
                "Date: 2026-06-24T12:16:07.000Z",
                "Author: Patrick Kelly",
                "",
                "GECKO ROBOTICS 100 S COMMONS STE 145 PITTSBURGH, PA15212-5359",
            ]
        ),
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact("p", "art_note", p)
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1
    site = sites[0]
    assert site.value["street_address"] == "100 S COMMONS STE 145"
    assert site.value["city"] == "PITTSBURGH"
    assert site.value["state"] == "PA"
    assert site.value["zip"] == "15212"
    assert site.value["aliases"] == ["GECKO ROBOTICS"]
    assert "site:pittsburgh_pa_15212" in site.entity_keys

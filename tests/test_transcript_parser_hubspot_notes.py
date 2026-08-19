from __future__ import annotations

from app.core.schemas import AtomType
from app.parsers.transcript_parser import TranscriptParser


def test_txt_note_transcript_parser_emits_physical_site_for_hubspot_address(tmp_path):
    p = tmp_path / "010058-hs-note-111645120815-GECKO ROBOTICS.txt"
    p.write_text(
        "HubSpot Note: GECKO ROBOTICS\n"
        "GECKO ROBOTICS 100 S COMMONS STE 145 PITTSBURGH, PA15212-5359\n",
        encoding="utf-8",
    )
    atoms = TranscriptParser().parse_artifact("p", "art_note", p)
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1
    assert sites[0].value["street_address"] == "100 S COMMONS STE 145"
    assert sites[0].value["city"] == "PITTSBURGH"
    assert sites[0].value["state"] == "PA"
    assert sites[0].value["zip"] == "15212"
    assert sites[0].value["aliases"] == ["GECKO ROBOTICS"]

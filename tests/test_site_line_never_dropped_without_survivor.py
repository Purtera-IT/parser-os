"""An address line typed physical_site is never dropped with no survivor.

Shape of deal 010353's intake request (synthetic text, real structure): a
markdown "## Job site" section holds one line "<number> <route>, <city>, <ST>
<ZIP>". Typed physical_site, it carried no site id, so `_dedupe_physical_site_atoms`
treated it as a ghost and dropped it outright; the only site kept was a
sentence fragment naming the same street without the ZIP. The suppression
ledger showed no survivor and the ZIP left the compile. Its only entity key was
a part number: the route "US-224" has a SKU's shape.

The compile's geo pass fills the line's city/state/ZIP from its own text, so it
counts as a located site -- and a located site with no id was the ghost case.
"""
from __future__ import annotations

from app.core.entity_extraction import _emit_part_numbers
from app.core.schemas import (ArtifactType, AtomType, AuthorityClass, EvidenceAtom,
                              ReviewStatus, SourceRef)
from app.core.semantic_dedup import _dedupe_physical_site_atoms

LINE = "4410 US-35, Cedarville, OH 45314"


def _site(aid: str, text: str, value: dict) -> EvidenceAtom:
    return EvidenceAtom(
        id=aid, project_id="d1", artifact_id="art_md", atom_type=AtomType.physical_site,
        raw_text=text, normalized_text=text.lower(), value=value, entity_keys=[],
        source_refs=[SourceRef(id=f"src_{aid}", artifact_id="art_md",
                               artifact_type=ArtifactType.txt, filename="INTAKE_REQUEST.md",
                               locator={"line_start": 13}, extraction_method="markdown_line_block",
                               parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=0.6,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t")


def _fragment() -> EvidenceAtom:
    text = "1 × Camera system(s) at Ridge Yard located at 4410 US-35, Cedarville, OH"
    sid = "1_camera_system_s_at_ridge_yard_located_at_4410_us_35_cedarville_oh"
    return _site("atm_fragment", text, {"kind": "physical_site", "id": sid, "site_id": sid,
                                        "name": "Cedarville", "address": "4410 US-35",
                                        "city": "Cedarville", "state": "OH"})


def _line() -> EvidenceAtom:
    # As the compile hands it over: the geo pass filled the location fields
    # from its own text, but nothing gave it a site id.
    return _site("atm_line", LINE, {"text": LINE, "section_path": ["Ridge Yard", "Job site"],
                                    "block_kind": "paragraph", "address": "4410 US-35",
                                    "city": "Cedarville", "state": "OH", "zip": "45314"})


def test_the_only_copy_of_the_zip_is_not_dropped():
    out = {a.id: a for a in _dedupe_physical_site_atoms([_fragment(), _line()])}
    assert "atm_line" in out, "the address line with the ZIP was dropped with no survivor"
    assert out["atm_line"].raw_text == LINE
    assert "atm_fragment" in out


def test_a_ghost_whose_details_a_site_states_still_goes():
    ghost = _site("atm_ghost", "Ridge Yard", {"text": "Ridge Yard"})
    out = {a.id for a in _dedupe_physical_site_atoms([_fragment(), ghost])}
    assert "atm_ghost" not in out


def test_a_route_in_a_street_address_is_not_a_part_number():
    assert _emit_part_numbers(LINE) == set()
    assert _emit_part_numbers("Site: 15733 US-224, Findlay") == set()
    assert _emit_part_numbers("Mount on SR-9 North of town") == set()
    # SKUs keep their keys, including one that happens to look like a route.
    assert _emit_part_numbers("Order 2 x C9300-48P and CW9166I-B") == {
        "part_number:c9300_48p", "part_number:cw9166i_b"}
    assert _emit_part_numbers("Replace the SR-2000 controller") == {"part_number:sr_2000"}

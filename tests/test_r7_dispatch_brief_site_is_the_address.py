"""A dispatch-brief sentence's site is its address, not its equipment list.

Shape of a live intake (synthetic names): "Please provide a quote ... for the
installation of 1 × Camera system(s) — <model>; 1 × Solar unit(s) — <desc>;
1 × Mount / mast — <desc> at <Site> located at <number> <route>, <City>, ST
<ZIP>." The street was taken from the FIRST number in the sentence, so the
only physical_site atom read "1 × Camera system(s) — ... located at
<address>" -- equipment and address in one atom. The equipment stays in the
sentence's own atom; the site atom is the address, ZIP and all.
"""
from __future__ import annotations

from app.core.address_parse import find_us_addresses_in_text
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.site_geo_fallback import geo_fallback_sites

BRIEF = (
    "Please provide a quote and earliest availability for the installation of "
    "1 × Camera system(s) — Falcon PTZ 4MP Camera; 1 × Solar unit(s) — 300 W solar "
    "unit mounted on a 25 foot crank mast; 1 × Mount / mast — 25 foot crank mast at "
    "Harbor Point located at 4821 OH-15, Ottawa, OH 45875."
)


def test_the_street_starts_at_the_house_number_after_the_last_clause_break():
    [addr] = find_us_addresses_in_text(BRIEF)
    assert addr.street_address == "4821 OH-15"
    assert (addr.city, addr.state, addr.zip) == ("Ottawa", "OH", "45875")
    assert addr.aliases == ("Harbor Point",)


def test_a_plain_address_and_a_company_lead_in_are_unchanged():
    [a] = find_us_addresses_in_text("GECKO ROBOTICS 100 S COMMONS STE 145 PITTSBURGH, PA 15212")
    assert a.street_address == "100 S COMMONS STE 145" and a.aliases == ("GECKO ROBOTICS",)
    [b] = find_us_addresses_in_text("Ship to: 72 Madison Avenue | New York, NY 10016")
    assert b.street_address.startswith("72 Madison Avenue")


def test_a_time_before_the_address_is_not_the_street():
    [a] = find_us_addresses_in_text("Meet at 10:30 at 500 Main St, Austin, TX 78701")
    assert a.street_address == "500 Main St" and a.aliases == ()


def test_the_geo_fallback_site_is_the_address_alone():
    ref = SourceRef(id="src_1", artifact_id="art_md", artifact_type=ArtifactType.txt, filename="INTAKE.md",
                    locator={"line_start": 35, "line_end": 35, "block_kind": "paragraph"},
                    extraction_method="markdown_line_block", parser_version="t")
    atom = EvidenceAtom(
        id="atm_brief", project_id="p", artifact_id="art_md", atom_type=AtomType.quantity,
        raw_text=BRIEF, normalized_text=BRIEF.lower(), value={}, entity_keys=[], source_refs=[ref],
        authority_class=AuthorityClass.customer_current_authored, confidence=0.91,
        review_flags=[], review_status=ReviewStatus.auto_accepted, parser_version="t",
    )
    [site] = geo_fallback_sites([atom], project_id="p")
    assert site.raw_text == "4821 OH-15, Ottawa, OH 45875"
    assert "×" not in site.raw_text and site.value["zip"] == "45875"

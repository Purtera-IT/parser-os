"""An empty site catalog must gate harder, not stop gating."""
from app.core.entity_extraction import extract_keys
from app.domain import get_active_domain_pack

PACK = get_active_domain_pack()
VSAT = (
    "This Statement of Work outlines the scope, deliverables, and responsibilities "
    "for the removal of VSAT (Very Small Aperture Terminal) equipment from "
    "designated rooftops."
)
REAL = "Install 12 APs at Wind Creek Atmore, 100 Hospitality Ln, Atmore, AL 36502."


def _sites(text, **kw):
    return sorted(k for k in extract_keys(text, pack=PACK, **kw) if k.startswith("site:"))


def test_empty_catalog_emits_no_sites():
    # The detector read every atom and vouched for nothing. That is a finding,
    # not missing information — so nothing may be asserted as a site.
    assert _sites(VSAT, authoritative_sites=set()) == []
    assert _sites(REAL, authoritative_sites=set()) == []


def test_a_populated_catalog_still_keeps_its_sites():
    assert _sites(REAL, authoritative_sites={"wind creek atmore"}) == ["site:wind_creek_atmore"]


def test_a_populated_catalog_still_rejects_a_phrase_outside_it():
    # The phantom this whole gate exists for: an acronym expansion lifted out
    # of a prose sentence.
    assert _sites(VSAT, authoritative_sites={"wind creek atmore"}) == []


def test_no_catalog_at_all_keeps_prior_behaviour():
    # None means no catalog was computed — callers that never build one must
    # not silently lose their site keys.
    assert _sites(VSAT) != []

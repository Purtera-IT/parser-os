"""A detected site whose address arrived as one lumped string is still a site
nobody can route to. These cover the enrichment that makes it locatable.

Shapes are taken verbatim from the OPTBOT iMac-refresh SOW, where the summary
table gives ``699 Broad St, Ste 1200`` and the access-window table two pages
later gives ``Augusta, GA 30901`` for the same site.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.core.site_geo_fallback import enrich_site_geo


def _site(**value):
    return SimpleNamespace(atom_type="physical_site", value=value,
                           raw_text=value.get("_text", ""), source_refs=[])


def _atom(text, atom_type="constraint"):
    return SimpleNamespace(atom_type=atom_type, value={}, raw_text=text,
                           source_refs=[])


def test_enriches_from_the_sites_own_address():
    s = _site(site_id="AUG-DC-06", name="Augusta Data Center Annex",
              address="699 Broad Street, Suite 1200, Augusta, GA 30901")
    assert enrich_site_geo([s]) == 1
    assert (s.value["city"], s.value["state"], s.value["zip"]) == (
        "Augusta", "GA", "30901")


def test_enriches_from_a_sibling_atom_naming_only_that_site():
    s = _site(site_id="AUG-DC-06", name="Augusta Data Center Annex",
              address="699 Broad St, Ste 1200, Augusta")
    sibling = _atom(
        "| **AUG-DC-06** | Augusta Data Center Annex, 699 Broad Street, "
        "Suite 1200, Augusta, GA 30901 | **Mon-Fri 08:00-16:00 only** |")
    assert enrich_site_geo([s, sibling]) == 1
    assert s.value["state"] == "GA" and s.value["zip"] == "30901"


def test_a_paragraph_naming_two_sites_enriches_neither():
    """Which of the two does the address belong to? Unknowable — so guessing
    is worse than leaving the field blank for the PM to fill."""
    a = _site(site_id="AUG-DC-06", name="Augusta Data Center Annex")
    b = _site(site_id="MAC-TRN-07", name="Macon Training Center")
    both = _atom("Sites AUG-DC-06 and MAC-TRN-07 both stage out of "
                 "the depot at Macon, GA 31201.")
    assert enrich_site_geo([a, b, both]) == 0
    assert "city" not in a.value and "city" not in b.value


def test_never_overwrites_what_an_extractor_already_established():
    s = _site(site_id="ATL-WEST-02", city="Atlanta", state="GA", zip="30318",
              address="976 Brady Ave NW, Savannah, GA 31401")
    assert enrich_site_geo([s]) == 0
    assert (s.value["city"], s.value["zip"]) == ("Atlanta", "30318")


def test_rejects_a_two_letter_token_that_is_not_a_state():
    s = _site(site_id="X1", address="12 Long Wharf, New Haven, XX 06511")
    assert enrich_site_geo([s]) == 0
    assert "state" not in s.value


def test_no_physical_sites_is_a_no_op():
    assert enrich_site_geo([_atom("Augusta, GA 30901")]) == 0

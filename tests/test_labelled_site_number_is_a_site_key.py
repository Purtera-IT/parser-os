"""A labelled site-number field names the place, whatever else the row calls it.

010447 (Computacenter / The Home Depot, "145 Locations" in its own subject line) published 380
``physical_site`` lines carrying 370 distinct sites, and the quote priced **366 sites / 3660 crew
hours**. The deal's own documents arrive as two sheets: one keys each store by its number, the other
by its name. Both state the number outright in a labelled field --

    Site No.: dc 5841 | Site Location: Hialeah - Miami DFC | State: FL | Location: USA

-- and 127 store numbers carried two keys each (``site:dc5841`` from one sheet,
``site:hialeah_miami_dfc`` from the other). Keying the number lets the machinery that already
exists do the rest: ``quote_sites``' "keys that share a line are one site", and the roster's own
code identity in ``site_identity``. No new join, no similarity.

Measured over the four deals compiled on 2026-11-05: 010447 370 -> 152 sites (stated 145); 010358
(101 sites), 010452 and 010171 unchanged.

The label carries the weight. A bare number in a row is not a site id, so a quantity, a part number
or a square footage can never become a place.
"""
from __future__ import annotations

import pytest

from app.core.entity_extraction import _emit_sites


def _sites(text: str) -> set[str]:
    return {k for k in _emit_sites(text) if k.startswith("site:")}


@pytest.mark.parametrize("text,key", [
    ("Site No.: dc 5841 | Site Location: Hialeah - Miami DFC | State: FL", "site:dc5841"),
    ("Site No.: dc 5883 | Site Location: Keasbey BDC | State: NJ", "site:dc5883"),
    ("Site No.: dc 5852a | net-new (y/n): Yes | replace/upgrade (y/n): No", "site:dc5852a"),
    ("Site ID: HOU-0012 | City: Houston", "site:hou_0012"),
    ("Location #: 2207 | Region: Midwest", "site:location_2207"),
])
def test_a_labelled_site_number_becomes_a_site_key(text, key):
    assert key in _sites(text)


@pytest.mark.parametrize("text", [
    "Qty: 4471 cables delivered",
    "Bldg SqFt: 42648 | Floors: 3",
    "Firewall Model: PA-3410 | quantity: 2",
    "Circuit Survey Hrs: 1.25 | WLAN Survey Hrs: 3.91184",
    "Net 30 terms | PO: 4500123456",
])
def test_a_number_that_is_not_a_labelled_site_number_is_not_a_place(text):
    assert _sites(text) == set() or not any(
        k.split(":", 1)[1].strip("_").isdigit() for k in _sites(text))


def test_the_two_sheets_of_one_store_share_a_key():
    """This is the whole point: the number is the identity both sheets agree on."""
    by_number = _sites("Site No.: dc 5841 | net-new (y/n): Yes | Firewall Model: PA-3410")
    by_name = _sites("Site No.: dc 5841 | Site Location: Hialeah - Miami DFC | State: FL")
    assert by_number & by_name, "the stated number has to be the shared key"
    assert "site:dc5841" in by_number & by_name


def test_a_bare_number_keeps_its_label_so_it_cannot_collide():
    """"Store #: 4471" is a place; a count of 4471 is not. The key says which."""
    got = _sites("Store #: 4471 | Open date: 2/14")
    assert "site:store_4471" in got
    assert "site:4471" not in got


def test_different_stores_stay_different():
    a = _sites("Site No.: dc 5841 | Site Location: Hialeah - Miami DFC")
    b = _sites("Site No.: dc 5883 | Site Location: Keasbey BDC")
    assert not ({k for k in a if k.startswith("site:dc")} & {k for k in b if k.startswith("site:dc")})

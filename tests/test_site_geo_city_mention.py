"""A site named for its city gets that city's state from a mention elsewhere (live 000061)."""
from types import SimpleNamespace

from app.core.site_geo_fallback import enrich_site_geo


def _atom(atom_type, text, value=None):
    return SimpleNamespace(atom_type=atom_type, raw_text=text, value=value or {}, entity_keys=[], source_refs=[])


def test_site_named_for_its_city_is_placed_from_a_city_state_mention():
    site = _atom("physical_site", "Yeah, it's a warehouse, but in the corner they have an office section with wall.",
                 {"name": "highland park warehouse office", "kind": "physical_site"})
    email = _atom("deal_metadata", "we would love to discuss the details you sent over about Highland Park, MI.")
    assert enrich_site_geo([site, email]) == 1
    assert (site.value["city"], site.value["state"]) == ("Highland Park", "MI")


def test_full_state_name_and_lead_in_words_are_understood():
    site = _atom("physical_site", "office", {"name": "highland park warehouse office"})
    said = _atom("raw_utterance", "This client has an Office In Highland Park, Michigan, and they.")
    enrich_site_geo([site, said])
    assert (site.value.get("city"), site.value.get("state")) == ("Highland Park", "MI")


def test_no_fill_when_the_name_does_not_carry_the_city_or_places_disagree():
    site = _atom("physical_site", "HQ", {"name": "north campus"})
    other = _atom("deal_metadata", "Shipping from Highland Park, MI.")
    assert enrich_site_geo([site, other]) == 0
    assert "city" not in site.value

    torn = _atom("physical_site", "x", {"name": "springfield office"})
    a = _atom("deal_metadata", "Springfield, IL is one option")
    b = _atom("deal_metadata", "or Springfield, MO")
    enrich_site_geo([torn, a, b])
    assert not torn.value.get("state")


def test_never_overwrites_an_established_city():
    site = _atom("physical_site", "x", {"name": "highland park office", "city": "Detroit", "state": "MI"})
    enrich_site_geo([site, _atom("deal_metadata", "Highland Park, MI")])
    assert site.value["city"] == "Detroit"

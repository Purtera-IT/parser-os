"""A place the documents name is a site candidate; the lines around it decide.

2026-09-15, untaught pool: 010283 listed nine cities under "Locations / Cities"
and two became sites; 010270 named "the Huntsville, Alabama facility" four
times and got none; 010307's inventory sheet had a Location column of thirteen
places and got none. "Heading to Dallas, Texas" is travel, a signature's
"Austin, TX" is a party's address -- those must stay mentions.
"""
from types import SimpleNamespace

import pytest

from app.core import decide as decide_mod, semantic_role, site_geo_fallback as sg
from app.core.decide import Decision


def _a(atom_type, text, value=None, artifact_id="art_note", **extra):
    return SimpleNamespace(atom_type=atom_type, raw_text=text, value=dict(value or {}), artifact_id=artifact_id,
                           source_refs=[], entity_keys=[], **extra)


@pytest.fixture(autouse=True)
def _known_places(monkeypatch):
    import app.core.geo_reference as geo
    known = {("el segundo", "CA"), ("kent", "WA"), ("seattle", "WA"), ("huntsville", "AL"), ("dallas", "TX"),
             ("austin", "TX"), ("los angeles", "CA"), ("englewood", "CO")}
    monkeypatch.setattr(geo, "is_known_place", lambda city, state=None: (str(city).lower(), str(state or "").upper()) in known)
    decide_mod.set_store(None)
    yield


def _judge(answers):
    calls = []

    def fake(text, candidates, *, instruction, context="", **_):
        calls.append((text, context))
        return answers.get(text, (None, 0.0))
    return fake, calls


def test_a_list_of_cities_becomes_sites_and_an_existing_one_is_skipped(monkeypatch):
    fake, calls = _judge({"El Segundo, CA": ("job_site", 0.93), "Seattle, WA": ("job_site", 0.9), "Englewood, CO": ("job_site", 0.9)})
    monkeypatch.setattr(semantic_role, "classify_role", fake)
    atoms = [
        _a("scope_item", "Locations"), _a("scope_item", "Cities"),
        _a("scope_item", "El Segundo, CA"), _a("scope_item", "Kent, WA"), _a("scope_item", "Seattle, WA"),
        _a("scope_item", "Essentially these are sites in which we could do one of the above"),
        _a("physical_site", "Kent, WA", {"name": "Kent, WA", "city": "Kent", "state": "WA"}),
        _a("scope_item", "Englewood, CO", artifact_id="art_other"),
    ]
    out = sg.geo_mention_sites(atoms, project_id="deal-1")
    assert sorted(s.value["name"] for s in out) == ["El Segundo, CA", "Englewood, CO", "Seattle, WA"]
    seg = next(s for s in out if s.value["city"] == "El Segundo")
    assert seg.value["state"] == "CA" and seg.value["inferred"] is True and "geo_mention_site" in seg.review_flags
    assert seg.value["geo_mention_source"] == "llm" and seg.value["geo_mention_confidence"] == 0.93
    assert "site:el_segundo_ca" in seg.entity_keys
    # the judge saw the list around the line, from the same document only
    text, ctx = next(c for c in calls if c[0] == "El Segundo, CA")
    assert "Locations" in ctx and "Cities" in ctx and "> El Segundo, CA" in ctx and "these are sites" in ctx
    assert "Englewood" not in ctx
    assert all(c[0] != "Kent, WA" for c in calls)


def test_travel_signatures_and_unknown_places_stay_mentions(monkeypatch):
    fake, calls = _judge({"Dallas, TX": ("mention_only", 0.9), "Huntsville, AL": ("job_site", 0.91)})
    monkeypatch.setattr(semantic_role, "classify_role", fake)
    atoms = [
        _a("scope_item", "Doing great. Traveling. Heading to Dallas, Texas. So going through an airport right now"),
        _a("stakeholder", "Enterprise Account Executive | HQ: 3828 Pecana Trail, Austin, TX 78749"),
        _a("scope_item", "Nor-Cal"),
        _a("dependency", "you're quoting a recurring onsite Smart Hands support agreement for the Huntsville, Alabama facility only."),
        _a("scope_item", "Meraki MS switch and a Cisco AP"),
    ]
    out = sg.geo_mention_sites(atoms, project_id="deal-2")
    assert [s.value["name"] for s in out] == ["Huntsville, AL"]
    assert out[0].value["mention"] == "Huntsville, Alabama"
    assert sorted(c[0] for c in calls) == ["Dallas, TX", "Huntsville, AL"]


def test_a_location_column_and_a_city_without_a_comma(monkeypatch):
    fake, calls = _judge({"Toronto": ("job_site", 0.9), "Los Angeles, CA": ("job_site", 0.88)})
    monkeypatch.setattr(semantic_role, "classify_role", fake)
    atoms = [
        _a("entity", "Location Toronto", {"entity_type": "location", "name": "Toronto"}, artifact_id="art_xlsx"),
        _a("raw_table_row", "Toronto | MDF 3rd Floor | Rack 2 | Vertiv", artifact_id="art_xlsx"),
        _a("scope_item", "Los Angeles CA"),
    ]
    out = sg.geo_mention_sites(atoms, project_id="deal-3")
    assert sorted(s.value["name"] for s in out) == ["Los Angeles, CA", "Toronto"]
    tor = next(s for s in out if s.value["name"] == "Toronto")
    assert tor.value["city"] == "Toronto" and tor.value["state"] is None
    assert "MDF 3rd Floor" in next(c[1] for c in calls if c[0] == "Toronto")


def test_a_taught_place_needs_no_model_and_the_model_needs_confidence(monkeypatch):
    class _Store:
        def resolve(self, *, relation, text, candidates, scope, **_):
            return Decision(verdict="job_site", confidence=0.97, source="store", correction_id="corr_pm") if text == "Huntsville, AL" else None
        def few_shot(self, **_): return []
    prev = decide_mod.get_store(); decide_mod.set_store(_Store())
    fake, calls = _judge({"Dallas, TX": ("job_site", 0.6)})
    monkeypatch.setattr(semantic_role, "classify_role", fake)
    try:
        out = sg.geo_mention_sites([_a("scope_item", "support for the Huntsville, AL facility"), _a("scope_item", "Dallas, TX")], project_id="deal-4")
    finally:
        decide_mod.set_store(prev)
    assert [s.value["name"] for s in out] == ["Huntsville, AL"]
    assert out[0].value["geo_mention_source"] == "store" and out[0].value["geo_mention_correction_id"] == "corr_pm"
    assert [c[0] for c in calls] == ["Dallas, TX"]


def test_cap_and_nothing_to_do(monkeypatch):
    monkeypatch.setenv("SOWSMITH_GEO_MENTION_MAX", "1")
    fake, _ = _judge({"El Segundo, CA": ("job_site", 0.9), "Seattle, WA": ("job_site", 0.9)})
    monkeypatch.setattr(semantic_role, "classify_role", fake)
    assert len(sg.geo_mention_sites([_a("scope_item", "El Segundo, CA"), _a("scope_item", "Seattle, WA")], project_id="d")) == 1
    assert sg.geo_mention_sites([_a("scope_item", "no place here")], project_id="d") == []
    assert sg.geo_mention_sites([], project_id="d") == []


def test_a_document_that_lists_several_places_is_judged_once(monkeypatch):
    """010283: nine cities under Locations / Cities. Judged one by one the same
    list came back job_site 0.80, abstain, job_site 0.80; judged as the list it
    is, once, every city follows."""
    calls = []

    def fake(text, candidates, *, instruction, context="", **_):
        calls.append(text)
        return ("job_site", 0.86) if text.startswith("3 places:") else (None, 0.0)
    monkeypatch.setattr(semantic_role, "classify_role", fake)
    atoms = [_a("scope_item", "Locations"), _a("scope_item", "El Segundo, CA"), _a("scope_item", "Seattle, WA"), _a("scope_item", "Englewood, CO")]
    out = sg.geo_mention_sites(atoms, project_id="deal-5")
    assert sorted(s.value["name"] for s in out) == ["El Segundo, CA", "Englewood, CO", "Seattle, WA"]
    assert calls == ["3 places: El Segundo, CA; Seattle, WA; Englewood, CO"]
    assert all(s.value["geo_mention_judged_as"] == "list" for s in out)


def test_a_second_address_in_the_same_city_is_its_own_site(monkeypatch):
    """010294: origin 5161 Lankershim is a site; the destination 5200 Lankershim,
    same city, was not. The address parser wants a ZIP; the line has none."""
    import app.core.geo_reference as geo
    monkeypatch.setattr(geo, "is_known_place", lambda city, state=None: True)
    fake, calls = _judge({"5200 Lankershim Blvd Ste 200, North Hollywood, CA": ("job_site", 0.9)})
    monkeypatch.setattr(semantic_role, "classify_role", fake)
    origin = _a("physical_site", "Origin Address", {"name": "Origin Address", "street_address": "5161 Lankershim Blvd ste 350", "city": "North Hollywood", "state": "CA"})
    line = _a("dependency", "Point of Origin: 5161 Lankershim Blvd Ste 350 North Hollywood, CA Point of Destination: 5200 Lankershim Blvd Ste 200 North Hollywood, CA")
    out = sg.geo_mention_sites([origin, line], project_id="deal-6")
    assert [s.value["name"] for s in out] == ["5200 Lankershim Blvd Ste 200, North Hollywood, CA"]
    assert out[0].value["street_address"] == "5200 Lankershim Blvd Ste 200" and out[0].value["city"] == "North Hollywood"
    assert calls and calls[0][0] == "5200 Lankershim Blvd Ste 200, North Hollywood, CA"

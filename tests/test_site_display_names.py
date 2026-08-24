"""A site the PM cannot name is a site the PM cannot talk about.

Every fixture here is a trimmed copy of a real ``envelope.json`` pulled
from blob — see ``tests/fixtures/site_naming/_build_fixtures.py`` for the
download commands. The shapes are the worker's, not mine: rows live under
``site_readiness.sites``, the row names itself in ``site``, and the atom's
structured payload arrives as ``structured`` rather than ``value``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.site_naming import recover_site_display_names

FIXTURES = Path(__file__).parent / "fixtures" / "site_naming"


def load(name: str) -> dict:
    with (FIXTURES / f"{name}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def recover(name: str) -> tuple[list[dict], dict[str, str]]:
    env = load(name)
    return env["sites"], recover_site_display_names(
        sites=env["sites"], atoms=env["atoms"], documents=env["documents"],
    )


# ── Clayton: the regression guard ────────────────────────────────────────
# 437 roster sites, every one already named. A rename here is a production
# regression — this deal is how we found out that 437 sites had collapsed
# to 3 the last time site handling changed.

def test_clayton_437_named_sites_are_untouched():
    sites, recovered = recover("6bc2dc0a")
    assert len(sites) == 437
    assert sum(1 for s in sites if s.get("facility_name")) == 437
    assert recovered == {}


def test_clayton_does_no_work_when_every_site_is_named():
    """The early return is the guard: a fully-anchored deal never reaches
    the scan, so no amount of atom text can rename one of its sites."""
    env = load("6bc2dc0a")
    assert recover_site_display_names(
        sites=env["sites"],
        atoms=[{"text": "Clayton Homes of Nowhere is the real site"}],
        documents=[],
    ) == {}


# ── Tier 1: the document's own words ─────────────────────────────────────

def test_prose_site_is_named_from_the_sentence_that_produced_it():
    """0cc36784. The document says "their Prudential Center office in
    Boston, MA"; the emitter slugged that to ``site:prudential_center_
    office`` and dropped the phrase. The PM saw the slug."""
    sites, recovered = recover("0cc36784")
    assert recovered == {"site:prudential_center_office": "Prudential Center office"}
    # The site that WAS anchored keeps the name the roster gave it.
    hq = next(s for s in sites if s["site"] == "site:hq")
    assert hq["facility_name"] == "HQ"


def test_facility_name_recovered_for_a_site_named_only_in_a_lead_sentence():
    """03783c65. "Wind Creek Atmore is soliciting proposals..." is the
    only place the facility is named."""
    _, recovered = recover("03783c65")
    assert recovered == {"site:wind_creek_atmore": "Wind Creek Atmore"}


def test_city_and_zip_only_sites_stay_unnamed():
    """Same deal. ``site:atmore_36502`` and ``site:chicago_60661`` are
    genuinely city/ZIP-only — there is no facility name in the documents,
    so inventing one would be worse than leaving the field blank."""
    sites, recovered = recover("03783c65")
    for key in ("site:atmore_36502", "site:chicago_60661"):
        assert key not in recovered
        row = next(s for s in sites if s["site"] == key)
        assert row["facility_name"]  # already named by the geo pass


# ── Tier 2: an identity string the compile already established ───────────

def test_collapsed_site_is_named_from_the_workbook_cell():
    """17dd11ae — the deal that must improve. ``site:azs_1`` is what
    survived collapsing ``site:maricopa_county_iron_mountain_data_centers_
    azs_1_scottsdale``; no atom text contains "AZS" at all, so tier 1
    cannot see it. The workbook cell can."""
    sites, recovered = recover("17dd11ae")
    assert recovered == {
        "site:azs_1": "Iron Mountain Data Centers AZS-1 - Scottsdale",
    }
    # Every site on the deal is now named — one by the roster, one here.
    named = sum(1 for s in sites if s.get("facility_name")) + len(recovered)
    assert named == len(sites) == 2


def test_the_fused_two_site_string_loses_to_the_specific_one():
    """The pool also holds "Maricopa County Iron Mountain Data Centers Azs
    1 Scottsdale", which names two sites at once. Shortest-wins is what
    keeps the PM from seeing a site's neighbour in its name."""
    _, recovered = recover("17dd11ae")
    assert "Maricopa" not in recovered["site:azs_1"]


# ── Guards on what may become a name ─────────────────────────────────────

def test_a_key_echo_is_not_a_name():
    """A single token that slugifies back to the key ("AZS-1" for
    ``site:azs_1``) is the identifier spelled differently. Handing it back
    would look like a fix while telling the PM nothing new."""
    assert recover_site_display_names(
        sites=[{"site": "site:azs_1", "aliases": []}],
        atoms=[{"text": "Racked servers staged at AZS-1 before pickup."}],
        documents=[],
    ) == {}
    # Same for the spaced spelling, and for a roster code like HC 238 —
    # one worded token plus a number is still just the identifier.
    assert recover_site_display_names(
        sites=[{"site": "site:azs_1", "aliases": []},
               {"site": "site:hc_238", "aliases": []}],
        atoms=[{"text": "Pickup at AZS 1 and HC 238 the same week."}],
        documents=[],
    ) == {}


def test_an_uncorroborated_longer_string_is_not_borrowed():
    """Tier 2 requires one of the row's own aliases to vouch for the
    candidate. Without that, any sentence containing the slug's tokens
    could name the site."""
    assert recover_site_display_names(
        sites=[{"site": "site:azs_1", "aliases": []}],
        atoms=[{"structured": {"aliases": ["Some Other Depot AZS-1 Annex"]}}],
        documents=[],
    ) == {}
    assert recover_site_display_names(
        sites=[{"site": "site:azs_1",
                "aliases": ["site:maricopa_county_some_other_depot_azs_1_annex"]}],
        atoms=[{"structured": {"aliases": ["Some Other Depot AZS-1 Annex"]}}],
        documents=[],
    ) == {"site:azs_1": "Some Other Depot AZS-1 Annex"}


@pytest.mark.parametrize("name", ["17dd11ae", "0cc36784", "03783c65", "6bc2dc0a"])
def test_recovery_never_renames_an_already_named_row(name):
    env = load(name)
    already_named = {s["site"] for s in env["sites"] if s.get("facility_name")}
    recovered = recover_site_display_names(
        sites=env["sites"], atoms=env["atoms"], documents=env["documents"],
    )
    assert already_named.isdisjoint(recovered)

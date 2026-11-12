"""With the heads settling a deal's sites, the site list (Scope tab, SOW) is their roster: a parser fill-in the heads
never read makes no row (000036, 2026-11-05: a geo-fallback line put CDW's own office in the SOW, the quote had none)."""
from __future__ import annotations

from app.core.orbitbrief_core import build_site_readiness


class _Atom:
    def __init__(self, i, keys, value=None, flags=()):
        self.id, self.atom_type, self.entity_keys = i, "physical_site", keys
        self.value, self.review_flags = value or {"id": keys[0][5:]}, list(flags)
        self.text = self.raw_text = ""
        self.artifact_id, self.confidence, self.authority_class = "art", 0.9, "machine_extractor"


def _sites(atoms, roster):
    return {r["site"] for r in build_site_readiness(atoms=atoms, edges=[], heads_roster=roster)["sites"]}


def test_an_empty_roster_lists_no_site():
    atoms = [_Atom("a1", ["site:new_york_ny_10016"], flags=["geo_fallback_site"])]
    assert _sites(atoms, []) == set()


def test_roster_sites_stay_and_others_go():
    atoms = [_Atom("a1", ["site:houston_77002"]), _Atom("a2", ["site:new_york_ny_10016"], flags=["geo_fallback_site"])]
    roster = [{"site": "site:houston_77002", "keys": ["site:houston_77002"], "atom_ids": ["a1"]}]
    assert _sites(atoms, roster) == {"site:houston_77002"}


def test_no_roster_keeps_the_old_list():
    atoms = [_Atom("a1", ["site:houston_77002"])]
    assert _sites(atoms, None) == {"site:houston_77002"}

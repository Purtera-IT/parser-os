"""A site minted after the geo pass still gets the place the document names.

Live 000061 (compile 9a6aacfc, 2026-09-15): the roster was empty after dedup,
so site_atom_backfill minted "highland park warehouse office" from the call --
long after site_geo_fallback had run over a deal with no sites. The brief
carried the site with no city or state while the transcript said "Highland
Park, Michigan" in the next breath. The compiler now runs the same enrichment
again on what backfill minted; this covers the pass that makes that work.
"""
from types import SimpleNamespace

from app.core.site_atom_backfill import _mint_physical_site
from app.core.site_geo_fallback import enrich_site_geo


def _said(text):
    return SimpleNamespace(atom_type="raw_utterance", value={"speaker": "Chase"}, raw_text=text, source_refs=[])


def test_a_late_minted_site_takes_the_place_the_call_named():
    anchor = _said("Yeah, it's a warehouse, but in the corner they have an office section with wall.")
    minted = _mint_physical_site(
        project_id="p1", site_key="site:highland_park_warehouse_office",
        display_name="highland park warehouse office", geo={}, source_atom=anchor, reason="entity_backfill",
    )
    mention = _said("This client has an office in Highland Park, Michigan, and they.")
    assert enrich_site_geo([anchor, minted, mention]) == 1
    assert (minted.value["city"], minted.value["state"]) == ("Highland Park", "MI")


def test_a_place_the_document_never_names_stays_open():
    anchor = _said("They have an office in the corner of the warehouse.")
    minted = _mint_physical_site(
        project_id="p1", site_key="site:warehouse_office", display_name="warehouse office",
        geo={}, source_atom=anchor, reason="entity_backfill",
    )
    assert enrich_site_geo([anchor, minted]) == 0
    assert "city" not in minted.value or not minted.value.get("city")

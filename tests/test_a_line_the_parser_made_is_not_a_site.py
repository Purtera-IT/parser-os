"""A place on a line the parser MADE is ours, not the document's.

010452's roster carried a JPMorgan lockbox letterhead -- "24386 IT ServicePoint 4 Chase Metrotech
Center 7th, Floor East Brooklyn, NY 11245" -- on a Green Bay job. The geo fallback minted a site
from it, and `document_scope` then resolved the whole SOW to that place:
`site:floor_east_brooklyn_ny_11245` sits on 146 atoms ("Date: 4/6/2026", "Version: 2", "2.0 | Scope
of Work") while the job's own `site:green_bay` sits on 4.

No head separates the two lines. Asked all 110 line heads on them, 42 answered, and the only two
that differ are `jdg:image` (0.523) and `read:crew_size` (0.658) -- neither about places. Every
place head agrees: read:address_level says `full` on both (0.933 / 0.971), read:site `present`
0.9999 on both, jdg:site_role `job_site` on both, jdg:geo_mention `mention_only` on both.

They are right. Both lines ARE full addresses, and nothing in the WORDS of a lockbox letterhead
distinguishes it from a work address. What distinguishes them is that one line was invented and the
other was read. That is provenance, not judgment, so it is not asked of a head.

Measured over the four deals: 010171 (1 roster line), 010447 (380), 010358 (103) have no made-up
roster line at all, and 010452 has exactly one.
"""
import types

from app.core import c3_type_base as c3

LETTERHEAD = "24386 IT ServicePoint 4 Chase Metrotech Center 7th, Floor East Brooklyn, NY 11245"
GREEN_BAY = "Address: 2283 Sachs Ct., Green Bay, WI 54313"


def _atom(aid, text, keys, flags=()):
    return types.SimpleNamespace(id=aid, atom_type="physical_site", raw_text=text, text=text,
                                 entity_keys=list(keys), review_flags=list(flags), value=None)


def _roster(atoms, reads=None):
    c3._DEAL.clear()
    c3._DEAL["site_keys"] = {"by": "heads:physical_site"}
    c3._DEAL["reads"] = reads or {}
    return c3.quote_sites(atoms)


def test_the_geo_fallback_letterhead_is_not_a_site():
    a = _atom("a1", LETTERHEAD, ["site:floor_east_brooklyn_ny_11245"], ["geo_fallback_site"])
    assert _roster([a]) == []


def test_a_geo_mention_site_is_not_a_site_either():
    a = _atom("a1", LETTERHEAD, ["site:floor_east_brooklyn_ny_11245"], ["geo_mention_site"])
    assert _roster([a]) == []


def test_the_real_work_address_beside_it_still_prices():
    real = _atom("a1", GREEN_BAY, ["site:green_bay"], ["site_entity_backfill"])
    made = _atom("a2", LETTERHEAD, ["site:floor_east_brooklyn_ny_11245"], ["geo_fallback_site"])
    assert [s["site"] for s in _roster([real, made])] == ["site:green_bay"]


def test_an_ordinary_site_line_is_untouched():
    a = _atom("a1", "1 Main St, Akron, OH 44301", ["site:akron_oh_44301"])
    assert [s["site"] for s in _roster([a])] == ["site:akron_oh_44301"]


def test_the_receipt_says_the_parser_made_the_line():
    a = _atom("a1", LETTERHEAD, ["site:floor_east_brooklyn_ny_11245"], ["geo_fallback_site"])
    rec = []
    c3._DEAL.clear()
    c3._DEAL["site_keys"] = {"by": "heads:physical_site"}
    c3._DEAL["reads"] = {}
    assert c3.quote_sites([a], record=rec) == []
    assert rec and rec[0]["why"] == "the parser made this line, it did not read it"


def test_a_party_line_is_not_rescued_by_the_role_head():
    """010358's extra sites were "Scott Morgan | 1361 N Glenville Dr | scott@traversa.net", typed
    stakeholder and keyed site:scott_morgan. jdg:site_role answers job_site on nearly every line."""
    a = types.SimpleNamespace(id="a1", atom_type="stakeholder", entity_keys=["site:scott_morgan"],
                              review_flags=[], value=None,
                              raw_text="Scott Morgan | 1361 N Glenville Dr | scott@traversa.net",
                              text="Scott Morgan | 1361 N Glenville Dr | scott@traversa.net")
    assert _roster([a], {"a1": {"jdg:site_role": ["job_site", 0.99]}}) == []

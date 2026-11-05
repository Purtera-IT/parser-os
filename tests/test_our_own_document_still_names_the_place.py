"""A work address in OUR document still names the job site, and a minted letterhead does not.

010452's one place is "Address: 2283 Sachs Ct., Green Bay, WI 54313", which sits in our own SOW.
The quoting read holds our documents out so the quote cannot read hours, crew or price off them --
right for those, wrong for the PLACE, which is a fact about the customer's premises our document
merely repeats. The exemption for that existed and asked for an `address:` entity key. That line
has none: it is keyed `site:green_bay` and `site:2283_sachs_ct_green_bay_wi_54313`. So it never
fired, the site stayed `seen_only_in_ours`, and the deal priced site_count 0 through four builds.
"""
import types

from app.core import c3_type_base as c3

GREEN_BAY = "2283 Sachs Ct., Green Bay, WI 54313"
LETTERHEAD = "24386 IT ServicePoint 4 Chase Metrotech Center 7th, Floor East Brooklyn, NY 11245"


def _atom(aid, text, keys, flags=(), value=None):
    return types.SimpleNamespace(id=aid, atom_type="physical_site", raw_text=text, text=text,
                                 entity_keys=list(keys), review_flags=list(flags), value=value)


def _roster(atoms, reads, out):
    c3._DEAL.clear()
    c3._DEAL["site_keys"] = {"by": "heads:physical_site"}
    c3._DEAL["view"] = "quoting"
    c3._DEAL["view_out_why"] = out
    c3._DEAL["reads"] = reads
    return c3.quote_sites(atoms, left_out=set(c3.out_of_quote()))


def test_stated_place_reads_a_street_and_zip():
    assert c3._stated_place(GREEN_BAY) is True
    assert c3._stated_place("Site No.: dc 5509 | net-new (y/n): Yes") is True
    assert c3._stated_place("Install Palo Alto pair only - USA - 37 locations") is False
    assert c3._stated_place("") is False


def test_our_own_document_still_names_the_work_site():
    a = _atom("a1", GREEN_BAY, ["site:green_bay", "site:2283_sachs_ct_green_bay_wi_54313"],
              flags=["site_entity_backfill"])
    got = _roster([a], {"a1": {"jdg:site_role": ["job_site", 0.9918]}}, {"a1": "ours"})
    assert [s["site"] for s in got] == ["site:green_bay"]
    assert got[0]["from_ours"] == "ours"


def test_a_letterhead_the_geo_fallback_made_is_not_a_job_site():
    """The role head calls it job_site at 0.987. What keeps it out is that the line was MADE."""
    a = _atom("a2", LETTERHEAD, ["site:floor_east_brooklyn_ny_11245"], flags=["geo_fallback_site"])
    assert _roster([a], {"a2": {"jdg:site_role": ["job_site", 0.987]}}, {"a2": "ours"}) == []


def test_a_line_value_marks_inferred_is_not_a_job_site():
    a = _atom("a3", GREEN_BAY, ["site:green_bay"], value={"inferred": True})
    assert _roster([a], {"a3": {"jdg:site_role": ["job_site", 0.99]}}, {"a3": "ours"}) == []


def test_the_real_site_comes_through_beside_the_letterhead():
    real = _atom("a1", GREEN_BAY, ["site:green_bay"], flags=["site_entity_backfill"])
    letter = _atom("a2", LETTERHEAD, ["site:floor_east_brooklyn_ny_11245"], flags=["geo_fallback_site"])
    got = _roster([real, letter],
                  {"a1": {"jdg:site_role": ["job_site", 0.9918]},
                   "a2": {"jdg:site_role": ["job_site", 0.987]}},
                  {"a1": "ours", "a2": "ours"})
    assert [s["site"] for s in got] == ["site:green_bay"]


def test_a_line_stating_no_place_is_not_exempted_however_sure_the_head_is():
    a = _atom("a4", "We will provide a new Access Point", ["site:access_point"])
    assert _roster([a], {"a4": {"jdg:site_role": ["job_site", 0.9474]}}, {"a4": "ours"}) == []


def test_an_address_key_still_works_as_it_did():
    a = _atom("a5", "the usual place", ["site:somewhere", "address:1_main_st"])
    got = _roster([a], {"a5": {"jdg:site_role": ["job_site", 0.99]}}, {"a5": "ours"})
    assert [s["site"] for s in got] == ["site:somewhere"]


def test_a_line_the_keep_head_dropped_can_still_name_the_job_site():
    """010171: col:admission dropped BOTH site lines (0.833 and 0.620) and the roster came back
    empty, so the deal priced site_count 0 with its one address sitting in the corpus.

    col:admission answers "should this text be an atom at all -- a real fact about the work, or
    wreckage, boilerplate, small talk or a torn fragment" (ml/c3/schema.py). A job site's address IS
    a real fact about the work, so that drop is an error against the head's own rubric -- and the
    head has 0 labels and grades 0.725. Either way the brief's keep decision is about what the brief
    SAYS; where the work happens is a fact about the customer's premises regardless.
    """
    a = _atom("a1", "4250 Executive Square, Suite 900, La Jolla, CA 92037",
              ["site:la_jolla_ca_92037"], flags=["email_note_physical_site"])
    c3._DEAL.clear()
    c3._DEAL["site_keys"] = {"by": "heads:physical_site"}
    c3._DEAL["reads"] = {"a1": {"jdg:site_role": ["job_site", 0.9568]}}
    got = c3.quote_sites([a], left_out={"a1"})
    assert [s["site"] for s in got] == ["site:la_jolla_ca_92037"]


def test_a_dropped_line_that_names_no_place_stays_dropped():
    """The exemption is not "the keep head is overridden": the line must state a place."""
    a = _atom("a2", "Thanks, talk soon!", ["site:thanks"])
    c3._DEAL.clear()
    c3._DEAL["site_keys"] = {"by": "heads:physical_site"}
    c3._DEAL["reads"] = {"a2": {"jdg:site_role": ["job_site", 0.92]}}
    assert c3.quote_sites([a], left_out={"a2"}) == []


def test_a_dropped_line_the_geo_fallback_invented_stays_dropped():
    a = _atom("a3", LETTERHEAD, ["site:floor_east_brooklyn_ny_11245"], flags=["geo_fallback_site"])
    c3._DEAL.clear()
    c3._DEAL["site_keys"] = {"by": "heads:physical_site"}
    c3._DEAL["reads"] = {"a3": {"jdg:site_role": ["job_site", 0.9885]}}
    assert c3.quote_sites([a], left_out={"a3"}) == []

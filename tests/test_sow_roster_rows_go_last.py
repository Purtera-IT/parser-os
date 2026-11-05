"""A roster row goes last in the writer's facts whatever the head types it.

The writer gets the site list as SITES; the rows filling its fact budget before the deal's own lines
is how 010035 lost its schedule (1,072 store rows under scope pushed out "all sites completed by
Thanksgiving 2026 ... Mon thru Sat 7am to 7pm"). The ordering keyed on
("physical_site", "site_attribute") and missed `site_infrastructure` -- what is AT a site rather than
the site itself -- which is how 141 of 010447's 416 facts are typed. Its first 381 facts were store
rows and the deal's own 22 scope lines came after them.
"""
from app.core.c3_sow import _STATED_SITE_NO, facts


def _env(atoms):
    rows = [{"id": a["id"], "text": a["raw_text"], "type": a["atom_type"]} for a in atoms]
    return {"atoms": atoms, "c3_brief": {"sections": {"scope_of_work": rows}}}


def _atom(aid, atom_type, text):
    return {"id": aid, "atom_type": atom_type, "raw_text": text, "text": text,
            "entity_keys": [], "review_flags": []}


def test_the_pattern_is_shared_with_the_roster():
    assert _STATED_SITE_NO.search("Site No.: dc 5841 | net-new (y/n): No")
    assert not _STATED_SITE_NO.search("Install Palo Alto pair AND decommission existing")


def test_a_site_infrastructure_roster_row_goes_after_the_deals_own_lines():
    atoms = [_atom("r1", "site_infrastructure", "Site No.: dc 5841 | net-new (y/n): No"),
             _atom("r2", "site_infrastructure", "Site No.: dc 5123 | net-new (y/n): Yes"),
             _atom("s1", "scope_item", "Install Palo Alto pair AND decommission existing Palo Alto")]
    got = [f["id"] for f in facts(_env(atoms))]
    assert got.index("s1") < got.index("r1"), got
    assert got.index("s1") < got.index("r2"), got


def test_a_row_stating_its_number_goes_last_even_typed_as_scope():
    atoms = [_atom("r1", "scope_item", "Site No.: dc 5841 | net-new (y/n): No"),
             _atom("s1", "scope_item", "Install Palo Alto pair AND decommission existing Palo Alto")]
    got = [f["id"] for f in facts(_env(atoms))]
    assert got.index("s1") < got.index("r1"), got


def test_a_physical_site_row_still_goes_last_as_before():
    atoms = [_atom("r1", "physical_site", "1 Main St, Akron, OH 44301"),
             _atom("s1", "scope_item", "Mount three displays in the main conference room")]
    got = [f["id"] for f in facts(_env(atoms))]
    assert got.index("s1") < got.index("r1"), got


def test_a_deal_with_no_roster_rows_keeps_its_order():
    atoms = [_atom("s1", "scope_item", "Mount three displays in the main conference room"),
             _atom("s2", "scope_item", "Mainly Microsoft Teams but perhaps an option for Zoom")]
    got = [f["id"] for f in facts(_env(atoms))]
    assert got == ["s1", "s2"], got

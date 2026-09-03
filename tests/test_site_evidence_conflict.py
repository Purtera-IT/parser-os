"""Two documents, one site, two addresses — report it, don't pick.

Deal 010215 carries Academy of Early Learning at BOTH 600 E Northside Ave
(per-site SOW, Aug 12) and 111 Academy St (locations list, Aug 21). Eight of ten
sites agree exactly; two do not, and in both cases the SOW carries the address of
the school in the preceding SOW.
"""

from __future__ import annotations

from app.core.site_evidence_conflict import (
    find_site_address_conflicts,
    normalize_address,
)

SOW = "SOW Smarthands Marion County SD Academy Of Early Learning.docx"
LIST = "Marion County School District Locations.docx"

REAL = [
    {"name": "Marion County School District Academy of Early Learning",
     "address": "600 E Northside Ave", "source": SOW, "authored_at": "2026-08-12T18:05:00Z"},
    {"name": "Academy of Early Learning",
     "address": "111 Academy St", "source": LIST, "authored_at": "2026-08-21T18:26:44Z"},
    {"name": "Marion County School District Johnakin Middle School",
     "address": "601 Gurley St", "source": "johnakin.docx", "authored_at": "2026-08-12T18:05:00Z"},
    {"name": "Johnakin MS",
     "address": "601 Gurley St", "source": LIST, "authored_at": "2026-08-21T18:26:44Z"},
]


def test_the_real_conflict_is_reported():
    out = find_site_address_conflicts(REAL)
    assert len(out) == 1, "only Academy of Early Learning disagrees"
    c = out[0]
    assert c["address_count"] == 2
    got = [a["address"] for a in c["addresses"]]
    assert "600 E Northside Ave" in got and "111 Academy St" in got


def test_it_names_the_document_behind_each_claim():
    # A conflict a PM cannot trace to two documents is not actionable.
    c = find_site_address_conflicts(REAL)[0]
    assert {a["source"] for a in c["addresses"]} == {SOW, LIST}


def test_the_later_claim_is_visible_as_later():
    c = find_site_address_conflicts(REAL)[0]
    assert c["addresses"][0]["authored_at"] < c["addresses"][1]["authored_at"]


def test_agreeing_sites_are_not_flagged():
    # Johnakin appears twice with the same address under two different names.
    assert all("Johnakin" not in str(c["site"]) for c in find_site_address_conflicts(REAL))


def test_spelling_differences_are_not_conflicts():
    # Crying wolf on "Ave" vs "Avenue" would train a reader to ignore the warning.
    same = [
        {"name": "Easterling Primary", "address": "600 E Northside Ave", "source": "a"},
        {"name": "Easterling Primary School", "address": "600 East Northside Avenue", "source": "b"},
    ]
    assert find_site_address_conflicts(same) == []
    assert normalize_address("600 E Northside Ave") == normalize_address("600 East Northside Avenue")


def test_a_truncated_address_IS_a_conflict():
    # "1123" vs "123" is not a spelling variant; it is a different building.
    diff = [
        {"name": "McCormick", "address": "1123 Sandy Bluff Rd", "source": "a"},
        {"name": "McCormick", "address": "123 Sandy Bluff Rd", "source": "b"},
    ]
    assert len(find_site_address_conflicts(diff)) == 1


def test_unnamed_sites_are_skipped_not_guessed():
    assert find_site_address_conflicts([
        {"name": "", "address": "1 A St", "source": "a"},
        {"name": "", "address": "2 B St", "source": "b"},
    ]) == []


def test_empty_and_missing_inputs_are_safe():
    assert find_site_address_conflicts([]) == []
    assert find_site_address_conflicts(None) == []
    assert find_site_address_conflicts([{"name": "X"}]) == []


def test_two_schools_at_one_address_are_reported():
    """The mirror case: dedup keeps both, so something must say they collide."""
    from app.core.site_evidence_conflict import find_address_collisions

    out = find_address_collisions([
        {"name": "Academy of Early Learning", "address": "600 E Northside Ave",
         "source": "SOW-A", "authored_at": "2026-08-12"},
        {"name": "Easterling Primary School", "address": "600 E Northside Avenue",
         "source": "SOW-B", "authored_at": "2026-08-13"},
        {"name": "Johnakin Middle School", "address": "601 Gurley St",
         "source": "SOW-C", "authored_at": "2026-08-12"},
    ])
    assert len(out) == 1
    assert out[0]["site_count"] == 2
    assert [s["name"] for s in out[0]["sites"]] == [
        "Academy of Early Learning", "Easterling Primary School"
    ]


def test_one_site_at_one_address_is_not_a_collision():
    from app.core.site_evidence_conflict import find_address_collisions

    assert find_address_collisions([
        {"name": "Johnakin Middle School", "address": "601 Gurley St"},
        {"name": "Johnakin Middle School", "address": "601 Gurley Street"},
    ]) == []

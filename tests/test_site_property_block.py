"""Read the site out of a per-site SOW's own header block.

Deal 010215: emails said "10 sites" eight times, ten per-site SOWs arrived, and
the site layer resolved two — because `physical_site` was only emitted for site
ROSTER tables (one site per data row), and a per-site SOW is the opposite shape:
one document, one site, stated as a labelled property block.
"""

from __future__ import annotations

from app.parsers.site_property_block import (
    fields_from_property_row,
    site_from_property_rows,
    site_display_name,
)

# Exactly as the parser now emits them, after the merged-header cell fix.
NAME_ROW = {
    "Requestor Information": "Cost Center/Loc Name",
    "Requestor Information__1": "Marion County School District\nJohnakin Middle School",
    "Requestor Information__2": "Cost Center/Loc #",
    "Requestor Information__3": "94575001",
}
ADDR_ROW = {
    "Requestor Information": "Address Line 1",
    "Requestor Information__1": "601 Gurley St",
    "Requestor Information__2": "Address Line 2",
}
CITY_ROW = {
    "Requestor Information": "City",
    "Requestor Information__1": "Marion",
    "Requestor Information__2": "Country",
    "Requestor Information__3": "USA",
}
STATE_ROW = {
    "Requestor Information": "State",
    "Requestor Information__1": "SC",
    "Requestor Information__2": "Zip Code",
    "Requestor Information__3": "29571",
}


def test_it_reads_the_real_johnakin_block():
    site = site_from_property_rows([NAME_ROW, ADDR_ROW, CITY_ROW, STATE_ROW])
    assert site["address"] == "601 Gurley St"
    assert site["city"] == "Marion"
    assert site["state"] == "SC"
    assert site["zip"] == "29571"
    # The shared account code is cost_center, NOT site_id: all ten SOWs carry
    # 94575001 (the district's), and calling it site_id made dedup collapse ten
    # schools into one.
    assert site["cost_center"] == "94575001"
    assert "Johnakin" in site["name"]


def test_the_address_is_whole_and_untruncated():
    # The email-scraped versions arrived fused and truncated
    # ("601 gurley street 1205 south main street", "1123" -> "123").
    site = site_from_property_rows([ADDR_ROW])
    assert site["address"] == "601 Gurley St"
    assert "1205" not in site["address"], "two addresses must never fuse"


def test_a_label_with_no_value_does_not_swallow_the_next_label():
    # "Address Line 2 | City | Marion" must not read Address Line 2 = "City".
    row = {"a": "Address Line 2", "b": "City", "c": "Marion"}
    out = fields_from_property_row(row)
    assert "address2" not in out
    assert out.get("city") == "Marion"


def test_placeholder_values_are_not_addresses():
    assert fields_from_property_row({"a": "Address Line 1", "b": "N/A"}) == {}
    assert fields_from_property_row({"a": "Address Line 1", "b": "TBD"}) == {}


def test_a_block_with_no_name_or_address_is_not_a_site():
    # City + state alone names no place; emitting it would invent a site.
    assert site_from_property_rows([CITY_ROW, STATE_ROW]) is None


def test_an_address_alone_is_enough():
    assert site_from_property_rows([ADDR_ROW])["address"] == "601 Gurley St"


def test_unrecognised_labels_are_ignored_rather_than_guessed():
    assert fields_from_property_row({"a": "Favourite Colour", "b": "blue"}) == {}


def test_empty_input_is_safe():
    assert fields_from_property_row(None) == {}
    assert fields_from_property_row({}) == {}
    assert site_from_property_rows([]) is None
    assert site_from_property_rows([None]) is None


def test_display_name_prefers_what_the_document_called_it():
    site = site_from_property_rows([NAME_ROW, ADDR_ROW])
    assert "Johnakin" in site_display_name(site)
    assert "\n" not in site_display_name(site)


def test_display_name_falls_back_to_the_address():
    assert site_display_name({"address": "601 Gurley St", "city": "Marion"}).startswith("601 Gurley St")


# ── end to end on the real documents ─────────────────────────────────────

def test_ten_per_site_sows_yield_ten_sites(tmp_path):
    """The whole point: 10 documents describing 10 locations => 10 sites.

    Before this, `physical_site` was only emitted for roster tables (one site
    per data row), so these ten contributed zero and the deal resolved its sites
    by scraping email prose — arriving fused and truncated.
    """
    import glob
    from pathlib import Path
    from app.parsers.docx_parser import DocxParser

    corpus = sorted(glob.glob(
        "/private/tmp/claude-501/-Users-purtera/688e4b18-55b8-4411-9d01-a00049d5ca4f"
        "/scratchpad/sodexo_all/SOW Smarthands*.docx"
    ))
    if len(corpus) < 10:
        import pytest
        pytest.skip("fixture corpus not present on this machine")

    sites = []
    for f in corpus:
        atoms = DocxParser().parse(Path(f))
        got = [a for a in atoms if str(getattr(a, "atom_type", "")).endswith("physical_site")]
        assert len(got) == 1, f"{Path(f).name} should yield exactly one site, got {len(got)}"
        sites.append((got[0].value or {}).get("address"))

    assert len(sites) == 10
    # Addresses the email-scraped path got wrong: truncated and fused.
    assert "1123 Sandy Bluff Rd" in sites, "leading digit must survive"
    assert "601 Gurley St" in sites
    assert not any(s and "1205 S Main St" in s and "Gurley" in s for s in sites), "no fusing"


def test_ten_sites_get_ten_distinct_keys():
    """site_readiness keys on value["id"]; a shared key silently merges sites.

    All ten 010215 SOWs carry Cost Center/Loc # 94575001 — the DISTRICT's number,
    not the school's. Keying on it collapses ten schools into one, which loses
    exactly as much as dropping nine of them.
    """
    import glob
    from pathlib import Path
    from app.parsers.docx_parser import DocxParser

    corpus = sorted(glob.glob(
        "/private/tmp/claude-501/-Users-purtera/688e4b18-55b8-4411-9d01-a00049d5ca4f"
        "/scratchpad/sodexo_all/SOW Smarthands*.docx"
    ))
    if len(corpus) < 10:
        import pytest
        pytest.skip("fixture corpus not present on this machine")

    keys, sids, centres = set(), set(), set()
    for f in corpus:
        atoms = DocxParser().parse(Path(f))
        v = ([a for a in atoms if str(getattr(a, "atom_type", "")).endswith("physical_site")][0].value or {})
        keys.add(v.get("id"))
        sids.add(v.get("site_id"))
        centres.add(v.get("cost_center"))

    assert len(keys) == 10, "each school must key distinctly"
    assert len(sids) == 10, "site_id must identify the SITE, since dedup reads it first"
    assert len(centres) == 1, "the cost centre really is shared — that is why it cannot be the key"


def test_two_schools_at_one_address_stay_two_sites():
    from app.parsers.site_property_block import site_key
    a = site_key({"name": "Marion County School District Marion High School", "address": "1205 S Main St"})
    b = site_key({"name": "Marion County School District Marion Intermediate School", "address": "1205 S Main St"})
    assert a != b, "a shared address does not make two schools one site"


def test_the_cost_centre_is_only_a_fallback():
    from app.parsers.site_property_block import site_key
    assert site_key({"cost_center": "94575001"}) == "loc_94575001"
    # A name always wins over the shared cost centre.
    assert site_key({"name": "Palmetto MS", "cost_center": "94575001"}) == "loc_palmetto_ms"


# ── universality: it must work for a vendor we have never seen ────────────

def test_a_vendor_with_completely_different_labels_still_works():
    """No 'Cost Center/Loc Name', no 'Address Line 1' — nothing we whitelisted.

    A label vocabulary is one vendor wide. The next customer writes "Site" and
    "Location", or leaves cells unlabelled, and a whitelist finds nothing —
    which looks exactly like a document containing nothing.
    """
    rows = [
        {"0": "Site", "1": "Riverside Distribution Centre"},
        {"0": "Location", "1": "4820 Camino Del Rio"},
        {"0": "Town", "1": "San Diego", "2": "Prov", "3": "CA"},
        {"0": "Postcode", "1": "92108"},
    ]
    site = site_from_property_rows(rows)
    assert site is not None
    assert site["address"] == "4820 Camino Del Rio"
    assert site["zip"] == "92108"
    assert site["state"] == "CA"
    assert "Riverside" in site["name"]


def test_no_labels_at_all():
    # An unlabelled block still yields the place.
    rows = [{"0": "Harbour Point Logistics Hub"}, {"0": "1180 Peachtree St"}, {"0": "GA", "1": "30309"}]
    site = site_from_property_rows(rows)
    assert site["address"] == "1180 Peachtree St"
    assert site["state"] == "GA" and site["zip"] == "30309"


def test_a_shared_account_code_no_longer_collapses_sites():
    """The 010215 failure from the other side.

    A vendor whose blocks all carry one account number used to key on it, so
    every site merged into one. The address is the identity when no name exists.
    """
    from app.parsers.site_property_block import site_key
    a = site_key({"cost_center": "ACCT-100", "address": "601 Gurley St"})
    b = site_key({"cost_center": "ACCT-100", "address": "747 Millers Rd"})
    assert a != b, "two addresses under one account code are two sites"


def test_the_account_code_is_the_last_resort_only():
    from app.parsers.site_property_block import site_key
    # Nothing identifies the place itself — collapsing is then correct, because
    # the block has not told us there is more than one.
    assert site_key({"cost_center": "ACCT-100"}) == "loc_acct_100"

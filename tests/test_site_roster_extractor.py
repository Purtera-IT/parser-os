"""Unit tests for the site_roster table extractor.

End-to-end PDF tests live in scripts/test_site_roster_corpus.py
(18 mock PDFs + the regen doc 08). These narrow tests pin the
behavior of looks_like_site_roster / extract_site_roster / the
site-code suffix gate without needing PDFs.
"""
from __future__ import annotations

from app.parsers.site_roster_extractor import (
    extract_site_roster,
    looks_like_site_roster,
    map_columns_to_fields,
)
from app.core.entity_extraction import _emit_sites, _site_code_suffix_ok


# ── _site_code_suffix_ok (Fix A) ─────────────────────────────────


def test_suffix_gate_accepts_function_dash_digits():
    # ATL-HQ-01: last="01", prev="HQ" (in allowlist) -> accept
    assert _site_code_suffix_ok("01", prev_segment="HQ") is True
    assert _site_code_suffix_ok("02", prev_segment="WEST") is True
    assert _site_code_suffix_ok("03", prev_segment="AIR") is True
    assert _site_code_suffix_ok("05", prev_segment="CP") is True


def test_suffix_gate_accepts_digits_dash_digits():
    # ATL-047-04: last="04", prev="047" (3-digit street #) -> accept
    assert _site_code_suffix_ok("04", prev_segment="047") is True


def test_suffix_gate_rejects_bare_digit_last_segment():
    # Pure digits with no prev_segment should NOT be a valid suffix
    # — the function is part of a larger code, not a standalone ID.
    assert _site_code_suffix_ok("01", prev_segment=None) is False


def test_suffix_gate_unchanged_for_normal_suffixes():
    # HQ, WEST, DC1 — pre-existing allowlist behavior preserved
    assert _site_code_suffix_ok("HQ") is True
    assert _site_code_suffix_ok("WEST") is True
    assert _site_code_suffix_ok("DC1") is True


# ── _emit_sites picks up the canonical Marriott IDs ─────────────


def test_emit_sites_captures_all_marriott_canonical_ids():
    text = (
        "Sites in scope: ATL-HQ-01, ATL-WEST-02, ATL-AIR-03, "
        "ATL-047-04, ATL-CP-05."
    )
    keys = _emit_sites(text)
    assert "site:atl_hq_01" in keys
    assert "site:atl_west_02" in keys
    assert "site:atl_air_03" in keys
    assert "site:atl_047_04" in keys
    assert "site:atl_cp_05" in keys


def test_emit_sites_still_rejects_junk_codes():
    text = "Junk: MOCK-OPTBOT-ATL, DEV-TEST-99, MSA-2026-001."
    keys = _emit_sites(text)
    # None of the junk codes should leak through
    for k in keys:
        assert "mock" not in k and "dev_test" not in k and "msa_2026" not in k


def test_emit_sites_rejects_connector_codes():
    text = "Use RJ-45 connectors, USB-C cables, BLE-5 radios."
    keys = _emit_sites(text)
    assert all("rj_45" not in k for k in keys)
    assert all("usb_c" not in k for k in keys)
    assert all("ble_5" not in k for k in keys)


# ── map_columns_to_fields ────────────────────────────────────────


def test_map_columns_canonical_headers():
    cols = ["Site ID", "Facility name", "Street address", "MDF / IDF",
            "Access window", "Escort owner"]
    m = map_columns_to_fields(cols)
    assert m[0] == "site_id"
    assert m[1] == "facility_name"
    assert m[2] == "street_address"
    assert m[3] == "mdf_idf"
    assert m[4] == "access_window"
    assert m[5] == "escort_owner"


def test_map_columns_handles_synonyms():
    # "Store #" -> site_id, "Location" -> street_address (closer match
    # than facility_name), "Address" -> would prefer street_address but
    # location took it first.
    cols = ["Store #", "Location", "Address"]
    m = map_columns_to_fields(cols)
    assert m[0] == "site_id"
    # Location matches the "location" keyword in street_address group;
    # the second Address column then has no remaining street_address
    # slot and isn't picked. That's acceptable — we just need the
    # primary signals (id + address).
    assert any(v == "street_address" for v in m.values())


def test_map_columns_explicit_declaration_positional_fallback():
    # Headers don't match any canonical pattern but explicit_declaration=True
    # falls back positionally.
    cols = ["Code", "Where", "Notes"]
    m = map_columns_to_fields(cols, explicit_declaration=True)
    # Code, Where, Notes -> name maps to facility_name, notes maps to notes;
    # Code falls through to positional site_id, "Where" falls through to
    # positional street_address.
    assert m.get(0) == "site_id"
    # Remaining columns absorbed positionally
    assert "facility_name" in m.values() or "street_address" in m.values()


# ── looks_like_site_roster ───────────────────────────────────────


def test_looks_like_site_roster_via_columns():
    cols = ["Site ID", "Facility", "Address"]
    rows = [{"Site ID": "X01", "Facility": "F", "Address": "1 Main St"}]
    assert looks_like_site_roster(columns=cols, rows=rows) is True


def test_looks_like_site_roster_via_explicit_declaration():
    cols = ["Code", "Name", "Where"]
    rows = [{"Code": "HQ", "Name": "Main", "Where": "123 Main"}]
    txt = "kind=physical_site for all rows below."
    assert looks_like_site_roster(columns=cols, rows=rows, surrounding_text=txt) is True


def test_looks_like_site_roster_via_row_shape():
    # No useful headers, but 3+ rows have site-shaped IDs in their
    # leftmost cells.
    cols = ["", "", ""]
    rows = [
        {"": "ATL-HQ-01", "_1": "...", "_2": "..."},
        {"": "ATL-WEST-02", "_1": "...", "_2": "..."},
        {"": "ATL-AIR-03", "_1": "...", "_2": "..."},
    ]
    # The dict keys don't help; pass through extract path
    assert looks_like_site_roster(columns=cols, rows=rows) is True


def test_does_not_classify_pricing_table_as_roster():
    cols = ["Item", "Quantity", "Unit price", "Total"]
    rows = [{"Item": "Switch", "Quantity": "5", "Unit price": "$1,000", "Total": "$5,000"}]
    assert looks_like_site_roster(columns=cols, rows=rows) is False


# ── extract_site_roster ─────────────────────────────────────────


def _rosterized(cols, rows, **kw):
    return extract_site_roster(columns=cols, rows=rows, **kw)


def test_extract_standard_table():
    cols = ["Site ID", "Facility", "Address"]
    rows = [
        {"Site ID": "NYC-HQ-01", "Facility": "Acme NY",  "Address": "1 Park Ave, NY"},
        {"Site ID": "BOS-MAIN-02","Facility":"Acme BOS", "Address": "100 Federal St, BOS"},
    ]
    out = _rosterized(cols, rows)
    assert len(out) == 2
    assert out[0].site_id == "NYC-HQ-01"
    assert out[0].facility_name == "Acme NY"
    assert out[0].street_address.startswith("1 Park Ave")


def test_extract_collapses_pdf_wrap_in_site_id():
    cols = ["Site ID", "Facility", "Address"]
    rows = [
        {"Site ID": "NYC-WEST-0\n2", "Facility": "Acme", "Address": "addr"},
    ]
    out = _rosterized(cols, rows)
    assert out[0].site_id == "NYC-WEST-02"


def test_extract_restores_underscores_from_spaces():
    # PDF rendering of "ATL_HQ_01" sometimes shows up as "ATL HQ 01"
    # because the underscores render as low-baseline pixels that get
    # detected as a separate text run. Test the underscore-restoration.
    cols = ["Code", "Name"]
    rows = [
        {"Code": "ATL HQ 01", "Name": "Atlanta HQ"},
    ]
    out = _rosterized(cols, rows, surrounding_text="kind=physical_site")
    assert any(r.site_id == "ATL_HQ_01" for r in out)


def test_extract_skips_header_row_duplicates():
    cols = ["Site ID", "Facility", "Address"]
    rows = [
        {"Site ID": "Site ID", "Facility": "Facility", "Address": "Address"},  # dup
        {"Site ID": "NYC-HQ-01", "Facility": "Acme NY", "Address": "1 Park"},
    ]
    out = _rosterized(cols, rows)
    assert len(out) == 1
    assert out[0].site_id == "NYC-HQ-01"


def test_extract_skips_pricing_table():
    cols = ["Item", "Quantity", "Unit price"]
    rows = [{"Item": "Switch", "Quantity": "5", "Unit price": "$1,000"}]
    assert _rosterized(cols, rows) == []


def test_extract_2_column_minimum():
    cols = ["Site ID", "Facility"]
    rows = [
        {"Site ID": "LON-HQ-01", "Facility": "Acme London"},
        {"Site ID": "LON-WEST-02", "Facility": "Acme Hammersmith"},
    ]
    out = _rosterized(cols, rows)
    assert len(out) == 2
    assert out[0].street_address is None


def test_extract_handles_extra_unknown_columns():
    cols = ["Site ID", "Facility", "Address", "Risk class", "Owner"]
    rows = [
        {"Site ID": "MIA-HQ-01", "Facility": "Miami", "Address": "100 Biscayne",
         "Risk class": "Tier 1", "Owner": "Jane Roe"},
    ]
    out = _rosterized(cols, rows)
    assert len(out) == 1
    # Unknown cols land in extras
    extras = dict(out[0].extra_fields)
    assert "Risk class" in extras or "Owner" in extras


def test_extract_phone_and_email():
    cols = ["Site ID", "Facility", "Address", "Phone", "Email"]
    rows = [
        {"Site ID": "AUS-HQ-01", "Facility": "Acme Austin",
         "Address": "100 Congress Ave, Austin TX",
         "Phone": "512-555-0100", "Email": "ops@acme.test"},
    ]
    out = _rosterized(cols, rows)
    assert out[0].phone == "512-555-0100"
    assert out[0].email == "ops@acme.test"

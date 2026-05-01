from app.core.item_identity import (
    canonical_item_identity,
    enrich_value_with_identity,
    is_primary_vendor_quantity,
    merge_parser_value_identity,
    normalize_inclusion_status,
)


def assert_key(text, expected):
    result = canonical_item_identity({"description": text})
    assert result is not None, text
    assert result.canonical_key == expected, (text, result)


def test_rj45_synonyms():
    for text in ["RJ45 terminations", "RJ-45", "data jack", "comm outlet", "work area outlet", "ethernet jack", "network port"]:
        assert_key(text, "rj45")


def test_cat6_shielding_distinct():
    assert_key("Cat6 UTP cable drops", "cat6_utp")
    assert_key("unshielded category 6 runs", "cat6_utp")
    assert_key("Cat6 STP cable drops", "cat6_stp")
    assert_key("shielded Cat6 runs", "cat6_stp")
    assert_key("Cat6A UTP drops", "cat6a_utp")
    assert_key("Shielded Cat6A", "cat6a_stp")


def test_cat6a_does_not_become_cat6():
    result = canonical_item_identity({"description": "Category 6A cable"})
    assert result is not None
    assert result.canonical_key == "cat6a"


def test_power_scope_pollution_not_poe():
    result = canonical_item_identity({"description": "4 20 amp power locations"})
    assert result is not None
    assert result.canonical_key == "power"
    assert result.scope_pollution_candidate is True
    poe = canonical_item_identity({"description": "PoE switch power over ethernet"})
    assert poe is not None
    assert poe.canonical_key == "poe"


def test_multi_interpretation_for_data_drop():
    results = canonical_item_identity({"description": "68 data drops including jacks and cable"}, allow_multi=True)
    keys = {r.canonical_key for r in results}
    assert "data_drop" in keys
    assert "rj45" in keys


def test_inclusion_statuses():
    assert normalize_inclusion_status("Yes") == "included"
    assert normalize_inclusion_status("Not Included") == "excluded"
    assert normalize_inclusion_status("By Others") == "excluded"
    assert normalize_inclusion_status("Alt 1") == "optional"
    assert normalize_inclusion_status("Allowance") == "allowance"
    assert normalize_inclusion_status("TBD") == "tbd"


def test_enrich_value_with_identity():
    value = enrich_value_with_identity({"description": "Cable certification report exports", "included": "No"})
    assert value["normalized_item"] == "certification_testing"
    assert value["inclusion_status"] == "excluded"


def test_is_primary_vendor_quantity_respects_optional_and_blob():
    assert is_primary_vendor_quantity(
        {"quantity": 10, "inclusion_status": "included", "quantity_status": "parsed"},
        raw_text="Cat6 cable",
    )
    assert not is_primary_vendor_quantity(
        {"quantity": 10, "inclusion_status": "optional", "quantity_status": "parsed"},
        raw_text="optional alt line",
    )
    assert not is_primary_vendor_quantity(
        {"quantity": 5, "inclusion_status": "unknown", "quantity_status": "parsed", "notes": "Not included"},
        raw_text="",
    )


def test_merge_parser_value_identity_keeps_protected_normalized_item():
    base = {"description": "power run", "normalized_item": "cat6_utp", "quantity": 12}
    merged = merge_parser_value_identity(dict(base), raw_text="Cat6 UTP runs")
    assert merged["normalized_item"] == "cat6_utp"

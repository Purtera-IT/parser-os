"""Unit tests for the sheet-role classifier (atom-emission gate)."""

from __future__ import annotations

from app.parsers.sheet_classifier import (
    SheetRole,
    classify_sheet,
)


def test_empty_sheet_is_suppressed():
    c = classify_sheet("Summary", [[None, None], ["", ""]])
    assert c.role is SheetRole.EMPTY
    assert c.suppress is True


def test_reference_name_do_not_edit_suppressed():
    rows = [["L0", "EUC"], ["L1", "DC"]]
    c = classify_sheet("Helper - Do not Edit", rows)
    assert c.role is SheetRole.REFERENCE
    assert c.suppress is True


def test_sell_rates_name_suppressed_even_with_typo():
    rows = [["PS-L1-ENG-LABOR-X", "Per Hour"], ["PS-TRAVEL", "Per Site"]]
    c = classify_sheet("SELLL RATES", rows)
    assert c.role is SheetRole.RATE_CARD
    assert c.suppress is True


def test_cost_rates_name_suppressed():
    c = classify_sheet("COST RATES", [["PS-RENTAL", "Per Day"]])
    assert c.suppress is True


def test_rate_code_content_suppressed_without_name_hint():
    # Renamed helper tab, but content is dense rate codes / skill levels.
    rows = [
        ["PS-TRAVEL-EXPENSE", "L0"],
        ["PS-L1-ENG-LABOR-ONSITE", "L1 EUC"],
        ["PS-L2-ENG-LABOR-ONSITE", "L2 DC"],
        ["PS-PROJMGMT-REMOTE", "Per Hour"],
        ["PS-MATERIALS-EQUIP", "Per Device"],
    ]
    c = classify_sheet("Tab3", rows)
    assert c.role is SheetRole.RATE_CARD
    assert c.suppress is True


def test_financial_summary_suppressed():
    rows = [
        ["Deal Summary", "", "Overall Deal Kit"],
        ["OPPTY #", "126", "Total Deal Revenue", "21560"],
        ["Sales Rep", "Dan", "Total Deal Cost", "15660"],
        ["Customer", "DCW", "Total Deal Margin", "5900"],
        ["Billing Type", "T&M", "Margin %", "0.27"],
    ]
    c = classify_sheet("Deal Kit", rows)
    assert c.role is SheetRole.FINANCIAL_SUMMARY
    assert c.suppress is True


def test_catalog_with_empty_order_qty_suppressed():
    header = ["ID #", "Material Description", "OEM", "Order QTY", "USA Cost $", "USA Sell $"]
    rows = [header]
    for i in range(1, 15):
        rows.append([str(i), f"CAT6 item {i}", "CommScope", "", "10.0", "12.0"])
    c = classify_sheet("Materials", rows)
    assert c.role is SheetRole.CATALOG
    assert c.suppress is True


def test_catalog_with_filled_order_qty_is_scope():
    # Same catalog columns but order quantities ARE populated -> real BOM.
    header = ["ID #", "Material Description", "OEM", "Order QTY", "USA Cost $", "USA Sell $"]
    rows = [header]
    for i in range(1, 15):
        rows.append([str(i), f"CAT6 item {i}", "CommScope", str(i * 2), "10.0", "12.0"])
    c = classify_sheet("Materials", rows)
    assert c.role is SheetRole.SCOPE
    assert c.suppress is False


def test_instructions_only_suppressed():
    rows = [
        ["INSTRUCTIONS TO BIDDERS"],
        ["Submit pricing in the attached Excel template."],
    ]
    c = classify_sheet("Sheet", rows)
    assert c.role is SheetRole.INSTRUCTIONS
    assert c.suppress is True


def test_real_scope_table_not_suppressed():
    rows = [
        ["Site", "Room", "Device", "Qty"],
        ["Banks HS", "101", "65in TV", "3"],
        ["Banks HS", "102", "65in TV", "2"],
    ]
    c = classify_sheet("Scope", rows)
    assert c.role is SheetRole.SCOPE
    assert c.suppress is False


def test_real_bom_with_part_and_qty_not_suppressed():
    rows = [
        ["Part Number", "Description", "Qty", "Unit Price"],
        ["WS-C3850", "Switch", "5", "1200"],
        ["AIR-AP", "Access Point", "10", "400"],
    ]
    c = classify_sheet("BOM", rows)
    assert c.suppress is False


def test_data_header_overrides_rate_code_density():
    # Even if a few rate codes appear, a real data header keeps it SCOPE.
    rows = [
        ["Site", "Device", "Qty", "Labor Code"],
        ["A", "TV", "3", "PS-L1-ENG-LABOR"],
        ["B", "TV", "2", "PS-L1-ENG-LABOR"],
    ]
    c = classify_sheet("Install Plan", rows)
    assert c.suppress is False

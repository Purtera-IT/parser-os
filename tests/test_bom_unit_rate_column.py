"""BOM unit cost reads from a rate/price column, never from a type column.

The pattern "unit $" normalized to the bare word "unit", so the first column
holding that word won. On a fee table "Unit Type | Unit Rate | Billable Units |
Subtotal" that is "Unit Type", and the first number in its text ("... (3 sites
x 10 days)") became the unit cost instead of the rate.
"""

from __future__ import annotations

import pytest

from app.core.table_schema_registry import (
    _col_matches,
    emit_atoms_for_schema,
    identify_schema,
)

FEE_COLS = ["Unit Type", "Unit Rate", "Billable Units", "Subtotal"]


def _bom(columns, row):
    atoms = emit_atoms_for_schema(
        schema_name="bom", columns=columns, row=row, row_idx=1, table_idx=0,
        project_id="p", artifact_id="a", filename="f.docx",
    )
    return [a for a in atoms if a.atom_type.value == "bom_line"][0].value


def test_unit_type_text_is_not_the_unit_cost():
    v = _bom(FEE_COLS, ["Field Tech - Per Day (3 sites x 10 days)", "$450.00", "30", "$13,500.00"])
    assert v["unit_cost"] == 450.0
    assert v["qty"] == 30


def test_rate_with_no_digit_in_the_type_column_is_still_read():
    v = _bom(FEE_COLS, ["Field Tech - Per Hour", "$95.00", "", ""])
    assert v["unit_cost"] == 95.0


def test_fee_table_still_routes_as_bom():
    assert identify_schema(FEE_COLS) == "bom"


@pytest.mark.parametrize("col", ["Unit Rate", "Unit Price", "Unit Cost", "Rate", "Price"])
def test_each_rate_or_price_column_is_the_unit_cost(col):
    v = _bom(["SKU", "Item Type", col, "Qty"], ["X-1", "Kit 7", "$12.50", "4"])
    assert v["unit_cost"] == 12.5


@pytest.mark.parametrize("col", ["Total Price", "Tax Rate"])
def test_a_total_or_tax_column_is_not_the_unit_cost(col):
    v = _bom(["SKU", "Qty", col], ["X-1", "4", "$50.00"])
    assert v["unit_cost"] is None


def test_dollar_pattern_needs_a_dollar_in_the_header():
    assert not _col_matches("Unit Type", ("unit $",))
    assert _col_matches("Unit $", ("unit $",))
    v = _bom(["SKU", "Qty", "Unit $"], ["X-1", "4", "7.25"])
    assert v["unit_cost"] == 7.25

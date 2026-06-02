"""Commercial-sheet routing: pricing is captured, not dropped, and
never masquerades as scope.

Estimating workbooks carry rate cards, master price catalogs and
deal-financials tabs. Those are not customer scope, but they hold
pricing the PM needs. The sheet-role router sends them to typed
commercial atoms (``commercial_total`` / ``pricing_assumption``) with
``money:`` entity keys — out of ``scope_truth``, into the OrbitBrief
``pricing_clarity`` surface. Pure backing data (helper / dropdown /
empty / cover) is still dropped.
"""

from __future__ import annotations

from openpyxl import Workbook

from app.core.schemas import AtomType, AuthorityClass
from app.parsers.sheet_classifier import (
    SheetDestination,
    SheetRole,
    classify_sheet,
)
from app.parsers.xlsx_parser import XlsxParser


def _atoms(path):
    out = XlsxParser().parse_artifact("proj", "art", path)
    return out if isinstance(out, list) else out.atoms


# ── classifier: role → destination mapping ──────────────────────────


def test_role_destination_mapping() -> None:
    from app.parsers.sheet_classifier import SheetClassification

    cases = {
        SheetRole.SCOPE: SheetDestination.SCOPE,
        SheetRole.RATE_CARD: SheetDestination.COMMERCIAL,
        SheetRole.CATALOG: SheetDestination.COMMERCIAL,
        SheetRole.FINANCIAL_SUMMARY: SheetDestination.COMMERCIAL,
        SheetRole.EMPTY: SheetDestination.DROP,
        SheetRole.INSTRUCTIONS: SheetDestination.DROP,
        SheetRole.REFERENCE: SheetDestination.DROP,
    }
    for role, expected in cases.items():
        c = SheetClassification(role=role, suppress=role is not SheetRole.SCOPE,
                                reason="t", confidence=1.0)
        assert c.destination is expected, role


# ── deal-financials summary (label → value pairs) ───────────────────


def test_financial_summary_emits_commercial_totals(tmp_path) -> None:
    path = tmp_path / "Deal_Kit.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Deal Kit"
    ws.append(["OPPTY #", 126, "Total Deal Revenue", 21560])
    ws.append(["Sales Rep", "Dan", "Total Deal Cost", 15660])
    ws.append(["Customer", "DCW", "Total Deal Margin", 5900])
    ws.append(["Billing Type", "T&M", "Margin % on Total Deal", 0.27])
    wb.save(path)

    atoms = _atoms(path)
    # Never scope.
    assert not [a for a in atoms if a.atom_type == AtomType.scope_item]
    totals = [a for a in atoms if a.atom_type == AtomType.commercial_total]
    assert totals
    keys = {k for a in atoms for k in (a.entity_keys or [])}
    # Headline deal economics captured as money atoms.
    assert {"money:21560", "money:15660", "money:5900"} <= keys
    # The 27% margin ratio is NOT money.
    assert "money:0" not in keys
    # Tagged as vendor/internal pricing, flagged for review.
    assert all(a.authority_class == AuthorityClass.vendor_quote for a in totals)


# ── master catalog (money-keyword columns) ──────────────────────────


def test_catalog_emits_pricing_assumptions(tmp_path) -> None:
    path = tmp_path / "Deal_Kit.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Master Price List"
    ws.append(["ID #", "Material Description", "OEM", "Order QTY", "USA Cost $"])
    ws.append([1, "CAT6 Plenum cable 1000ft", "CommScope", None, 661.48])
    ws.append([2, "CAT6 Non-plenum 1000ft", "CommScope", None, 338.71])
    wb.save(path)

    atoms = _atoms(path)
    assert not [a for a in atoms if a.atom_type == AtomType.scope_item]
    pricing = [a for a in atoms if a.atom_type == AtomType.pricing_assumption]
    assert pricing
    keys = {k for a in atoms for k in (a.entity_keys or [])}
    assert "money:661" in keys and "money:339" in keys


# ── pure backing data is still dropped ──────────────────────────────


def test_helper_sheet_drops_to_nothing(tmp_path) -> None:
    path = tmp_path / "Deal_Kit.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Helper - Do not Edit"
    ws.append(["L0"])
    ws.append(["L1"])
    ws.append(["L2"])
    wb.save(path)

    assert _atoms(path) == []

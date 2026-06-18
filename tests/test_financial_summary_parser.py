"""Structured deal-financials / P&L extraction in the xlsx parser.

A deal-kit financial summary is a 2-D label→value grid, not a row table.
The generic commercial emitter mashed unrelated cells together
(``OPPTY # | 126 | Total Deal Revenue | 21560``); the structured
extractor must instead recover a clean deal header + per-category P&L,
and fall back to the generic emitter when the grid isn't a P&L.
"""

from __future__ import annotations

from app.parsers.xlsx_parser import XlsxParser
from app.core.schemas import ArtifactType


def _emit(rows, sheet_name="Deal Kit"):
    p = XlsxParser()
    return p._emit_financial_summary_rows(
        project_id="p",
        artifact_id="a",
        artifact_type=ArtifactType.xlsx,
        filename="Deal Kit.xlsx",
        sheet_name=sheet_name,
        rows=rows,
    )


def _merge_header(atoms):
    """Merge the per-field ``deal_metadata`` atoms (one atom per header row) into
    one (fields, entity_keys) — mirrors build_deal_header's render-time reassembly."""
    fields: dict = {}
    ekeys: set = set()
    for a in atoms:
        if a.atom_type.value != "deal_metadata":
            continue
        for k, v in (a.value.get("fields") or {}).items():
            fields.setdefault(k, v)
        ekeys.update(a.entity_keys or [])
    return fields, ekeys


def _pl_grouped(atoms):
    """Regroup the per-row ``pl_metric`` atoms (one per sheet row) back into
    {category_key: {revenue, cost, margin, margin_pct}} for assertions — first
    value per (category, metric) wins, same as the PM brief's render-time roll-up."""
    pl: dict = {}
    for a in atoms:
        v = a.value if isinstance(a.value, dict) else {}
        if v.get("kind") != "pl_metric":
            continue
        ck, m = v.get("category_key"), v.get("metric")
        if not ck or m not in ("revenue", "cost", "margin", "margin_pct"):
            continue
        slot = pl.setdefault(ck, {"revenue": None, "cost": None, "margin": None, "margin_pct": None})
        if slot.get(m) is None:
            slot[m] = v.get("value")
    return pl


# A compact deal-kit grid: header fields on the left, P&L block below.
_GRID = [
    ["OPPTY #", 126, "Total Deal Revenue", 21560],
    ["Sales Rep", "Dan", "Total Deal Cost", 15660],
    ["Customer", "DCW", "Total Deal Margin", 5900],
    ["Billing Type", "T&M", "Margin % on Total Deal", 0.2737],
    ["Region", "USA", None, None],
    ["Project Financials", None, None, None],
    ["Total Labor Revenue", 21560, "Margin % on Labor", 0.2857],
    ["Total Labor Cost", 15400, None, None],
    ["Total Labor Margin", 6160, None, None],
    ["Total PMO Revenue", 0, None, None],
    ["Total PMO Cost", 260, None, None],
    ["Total PMO Margin", -260, None, None],
    ["Materials Revenue", 0, "Materials Cost", 0],
]


def test_deal_header_extracted():
    atoms = _emit(_GRID)
    f, ekeys = _merge_header(atoms)   # one atom per field; merge for the record
    assert f["opportunity_id"] == "126"
    assert f["sales_rep"] == "Dan"
    assert f["customer"] == "DCW"
    assert f["billing_type"] == "T&M"
    assert f["region"] == "USA"
    assert "deal:126" in ekeys


def test_pl_categories_extracted_clean():
    atoms = _emit(_GRID)
    pl = _pl_grouped(atoms)
    assert pl["deal"]["revenue"] == 21560
    assert pl["deal"]["cost"] == 15660
    assert pl["deal"]["margin"] == 5900
    assert pl["deal"]["margin_pct"] == 27.37  # fraction → percent
    assert pl["labor"]["margin"] == 6160
    assert pl["labor"]["margin_pct"] == 28.57
    assert pl["pmo"]["cost"] == 260
    assert pl["pmo"]["margin"] == -260


def test_generic_header_fields_captured_beyond_whitelist():
    # Non-standard header labels (not in the canonical vocabulary) must
    # still be captured structurally once the grid is confirmed a deal kit,
    # so no header datum is silently dropped.
    rows = [
        ["OPPTY #", 126, "Total Deal Revenue", 21560],
        ["Customer", "DCW", "Total Deal Cost", 15660],
        ["PO Number", "PO-4471", "Total Deal Margin", 5900],
        ["Account Manager", "Rivera", None, None],
        ["Total Labor Revenue", 21560, None, None],
        ["Total Labor Cost", 15400, None, None],
    ]
    atoms = _emit(rows)
    f, _ = _merge_header(atoms)
    # canonical keys preserved
    assert f["opportunity_id"] == "126"
    assert f["customer"] == "DCW"
    # generic slugged keys captured
    assert f["po_number"] == "PO-4471"
    assert f["account_manager"] == "Rivera"


def test_div_zero_pl_metric_captured_faithfully():
    # A KNOWN P&L metric whose formula failed (#DIV/0!) is meaningful content —
    # it tells the PM the metric is undefined (e.g. margin% on $0 revenue) — so it
    # is captured as a faithful, flagged atom, never silently dropped. (An Excel
    # error in an arbitrary HEADER field is still rejected — see the sweep test.)
    rows = [
        ["Total Deal Revenue", 1000, "Margin % on Total Deal", "#DIV/0!"],
        ["Total Deal Cost", 600, None, None],
        ["Total Deal Margin", 400, None, None],
        ["Total Labor Revenue", 1000, None, None],
        ["End User", "TBD", None, None],
    ]
    atoms = _emit(rows)
    # the #DIV/0! margin row is emitted faithfully (the raw error string, flagged)
    err = [
        a for a in atoms
        if isinstance(a.value, dict) and a.value.get("kind") == "pl_metric"
        and a.value.get("category_key") == "deal" and a.value.get("metric") == "margin_pct"
    ]
    assert len(err) == 1
    assert err[0].value.get("value") == "#DIV/0!"
    assert err[0].value.get("formula_error") == "#DIV/0!"
    assert "xlsx_parser:formula_error" in err[0].review_flags
    assert "#DIV/0!" in err[0].raw_text


def test_sweep_rejects_excel_errors_and_heading_values():
    # The structural header sweep must not mint junk fields: an Excel error
    # literal (#DIV/0!) is a failed formula, and a multi-word non-numeric
    # phrase is a section title, not an atomic field value. A real atomic
    # value next to a non-canonical label is still captured (control).
    rows = [
        ["OPPTY #", 126, "Total Deal Revenue", 21560],
        ["Customer", "DCW", "Total Deal Cost", 15660],
        ["PO Number", "PO-4471", "Total Deal Margin", 5900],
        ["Summary", "Overall Deal Kit Summary Net Margin", None, None],
        ["Status", "#DIV/0!", None, None],
        ["Total Labor Revenue", 21560, None, None],
        ["Total Labor Cost", 15400, None, None],
    ]
    atoms = _emit(rows)
    f, _ = _merge_header(atoms)
    # atomic non-canonical value captured
    assert f["po_number"] == "PO-4471"
    # heading-like multi-word value rejected
    assert "summary" not in f
    # Excel error literal rejected
    assert "status" not in f
    assert "#DIV/0!" not in f.values()


def test_non_pl_grid_falls_back_to_commercial_emitter():
    # A money table with no P&L vocabulary and no deal-header fields must
    # NOT be swallowed by the structured extractor — it falls back so the
    # money rows still surface.
    rows = [
        ["Role", "Country", "Rate"],
        ["Tech 1", "USA", 55],
        ["Tech 2", "USA", 55],
        ["PM", "USA", 50],
    ]
    atoms = _emit(rows, sheet_name="Gantt Financials")
    # Fallback path emits commercial atoms (rollup + rows), none typed as
    # deal_metadata or pl_line.
    assert atoms, "fallback should still emit money atoms"
    assert not any(a.value.get("kind") == "pl_line" for a in atoms)
    assert not any(a.atom_type.value == "deal_metadata" for a in atoms)

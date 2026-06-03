"""PM render sections built from structured commercial atoms.

The xlsx parser now emits a ``deal_metadata`` header atom and per-category
``commercial_total`` P&L atoms (``value.kind == "pl_line"``), plus folded
materials rollups. These builders assemble the PM-facing deal_header /
deal_financials / bill_of_materials views without re-parsing text.
"""

from __future__ import annotations

from app.core.orbitbrief_core import (
    build_bill_of_materials,
    build_deal_financials,
    build_deal_header,
)
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom(atom_type, *, value=None, entity_keys=None, text="x", aid="art_x"):
    return EvidenceAtom(
        id=f"atm_{abs(hash((str(atom_type), text, repr(value)))) % (10**12):012x}",
        project_id="p",
        artifact_id=aid,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value=value or {},
        entity_keys=entity_keys or [],
        source_refs=[
            SourceRef(
                id="src_1",
                artifact_id=aid,
                artifact_type=ArtifactType.xlsx,
                filename="Deal Kit.xlsx",
                locator={},
                extraction_method="test",
                parser_version="t",
            )
        ],
        receipts=[],
        authority_class=AuthorityClass.vendor_quote,
        confidence=0.8,
        confidence_raw=0.8,
        calibrated_confidence=0.8,
        review_status=ReviewStatus.needs_review,
        review_flags=[],
        parser_version="t",
    )


def _header_atom(fields):
    return _atom(
        AtomType.deal_metadata,
        value={"kind": "deal_header", "fields": fields, "sheet_name": "Deal Kit"},
        entity_keys=["deal:126", "customer:dcw"],
    )


def _pl_atom(category, category_key, *, revenue=None, cost=None, margin=None, margin_pct=None):
    return _atom(
        AtomType.commercial_total,
        value={
            "kind": "pl_line",
            "category": category,
            "category_key": category_key,
            "revenue": revenue,
            "cost": cost,
            "margin": margin,
            "margin_pct": margin_pct,
            "sheet_name": "Deal Kit",
        },
    )


# ───────────────────────── deal_header ─────────────────────────


def test_deal_header_merges_fields():
    atoms = [
        _header_atom(
            {
                "opportunity_id": "126",
                "sales_rep": "Dan",
                "customer": "DCW",
                "billing_type": "T&M",
                "region": "USA",
            }
        )
    ]
    h = build_deal_header(atoms=atoms)
    assert h["present"] is True
    assert h["field_count"] == 5
    assert h["fields"]["opportunity_id"] == "126"
    assert h["fields"]["customer"] == "DCW"
    assert len(h["source_atom_ids"]) == 1


def test_deal_header_first_wins_across_atoms():
    atoms = [
        _header_atom({"customer": "DCW", "region": "USA"}),
        _header_atom({"customer": "OTHER", "sales_rep": "Dan"}),
    ]
    h = build_deal_header(atoms=atoms)
    # First non-empty value per field wins.
    assert h["fields"]["customer"] == "DCW"
    assert h["fields"]["region"] == "USA"
    assert h["fields"]["sales_rep"] == "Dan"


def test_deal_header_absent_when_no_metadata():
    h = build_deal_header(atoms=[_pl_atom("Deal", "deal", revenue=100)])
    assert h["present"] is False
    assert h["fields"] == {}


# ──────────────────────── deal_financials ──────────────────────


def test_pl_lines_ordered_and_totals_prefer_deal_line():
    atoms = [
        _pl_atom("PMO", "pmo", revenue=0, cost=260, margin=-260),
        _pl_atom("Labor", "labor", revenue=21560, cost=15400, margin=6160, margin_pct=28.57),
        _pl_atom("Deal", "deal", revenue=21560, cost=15660, margin=5900, margin_pct=27.37),
    ]
    f = build_deal_financials(atoms=atoms)
    assert f["present"] is True
    assert f["category_count"] == 3
    # Deal first, then labor, then pmo (per _PL_CATEGORY_ORDER).
    assert [l["category_key"] for l in f["lines"]] == ["deal", "labor", "pmo"]
    # Totals come from the explicit grand-total deal line.
    assert f["totals"]["revenue"] == 21560
    assert f["totals"]["cost"] == 15660
    assert f["totals"]["margin"] == 5900
    assert f["totals"]["margin_pct"] == 27.37


def test_totals_summed_when_no_deal_line():
    atoms = [
        _pl_atom("Labor", "labor", revenue=1000, cost=600, margin=400),
        _pl_atom("PMO", "pmo", revenue=0, cost=100, margin=-100),
    ]
    f = build_deal_financials(atoms=atoms)
    assert f["totals"]["revenue"] == 1000
    assert f["totals"]["cost"] == 700
    assert f["totals"]["margin"] == 300
    assert f["totals"]["margin_pct"] == 30.0


def test_financials_absent_when_no_pl_lines():
    f = build_deal_financials(atoms=[_header_atom({"customer": "DCW"})])
    assert f["present"] is False
    assert f["lines"] == []


# ─────────────────────── bill_of_materials ─────────────────────


def _materials_rollup():
    return _atom(
        AtomType.pricing_assumption,
        value={
            "kind": "pricing_rollup",
            "sheet_name": "Materials",
            "sheet_role": "catalog",
            "line_count": 2,
            "rows": [
                {"row": 2, "label": "Cat6 cable", "cells": ["Cat6 cable", 0.5], "money_keys": ["money:0_5"]},
                {"row": 3, "label": "RJ45 jack", "cells": ["RJ45 jack", 1.2], "money_keys": ["money:1_2"]},
            ],
        },
    )


def _rate_card_rollup():
    return _atom(
        AtomType.commercial_total,
        value={
            "kind": "pricing_rollup",
            "sheet_name": "SELL RATES",
            "sheet_role": "rate_card",
            "line_count": 3,
            "rows": [
                {"row": 2, "label": "Tech 1", "cells": ["Tech 1", 55], "money_keys": ["money:55"]},
            ],
        },
    )


def test_bom_surfaces_materials_rows():
    bom = build_bill_of_materials(atoms=[_materials_rollup()])
    assert bom["present"] is True
    assert bom["section_count"] == 1
    assert bom["total_lines"] == 2
    labels = [r["label"] for r in bom["sections"][0]["rows"]]
    assert "Cat6 cable" in labels


def test_bom_excludes_rate_card_rollups():
    bom = build_bill_of_materials(atoms=[_rate_card_rollup()])
    assert bom["present"] is False
    assert bom["sections"] == []


def test_bom_includes_materials_but_not_rate_card_when_mixed():
    bom = build_bill_of_materials(atoms=[_materials_rollup(), _rate_card_rollup()])
    assert bom["section_count"] == 1
    assert bom["sections"][0]["sheet_name"] == "Materials"

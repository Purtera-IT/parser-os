"""Integration: material identity roster vs vendor quantity contradict edges on real COPPER pack (optional path)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.core.compiler import compile_project
from app.core.schemas import AuthorityClass, PacketFamily


COPPER_ROOT = Path(
    os.environ.get(
        "COPPER_VALIDATION_ROOT",
        r"c:\Users\lilli\Downloads\purtera_copper_low_voltage_public_validation_packs"
        r"\purtera_copper_low_voltage_validation_packs\real_data_cases",
    )
)
CASE_DIR = COPPER_ROOT / "COPPER_001_SPRING_LAKE_AUDITORIUM" / "artifacts"


@pytest.mark.skipif(not (CASE_DIR / "extracted").is_dir(), reason="COPPER_001 artifacts not present")
def test_copper_drop_schedule_vs_vendor_material_contradiction_edges() -> None:
    project_dir = CASE_DIR
    result = compile_project(
        project_dir=project_dir,
        project_id="COPPER_001_SPRING_LAKE_AUDITORIUM",
        allow_errors=True,
        allow_unverified_receipts=True,
    )
    mat = [
        e
        for e in result.edges
        if e.edge_type.value == "contradicts"
        and (e.metadata or {}).get("comparison_basis") == "aggregate_roster_vs_summed_vendor_quote"
    ]
    assert len(mat) >= 3
    by = {a.id: a for a in result.atoms}
    seen_identities: set[str] = set()
    for e in mat:
        fa, ta = by.get(e.from_atom_id), by.get(e.to_atom_id)
        assert fa and ta
        assert fa.authority_class == AuthorityClass.approved_site_roster
        assert ta.authority_class == AuthorityClass.vendor_quote
        assert fa.atom_type.value == "quantity"
        # The vendor LINE is the reconciliation node (aeae84e): it carries
        # quantity / normalized_item / comparison_key / inclusion state. The
        # standalone `quantity` atom the quote parser still emits for the same
        # row is the redundant copy and must NOT be the edge endpoint — if it
        # is, that row is being counted twice in the vendor total.
        assert ta.atom_type.value == "vendor_line_item"
        ni = (fa.value or {}).get("normalized_item")
        assert isinstance(ni, str)
        seen_identities.add(ni.lower())
    assert {"rj45", "cat6_utp", "cat6_stp"}.issubset(seen_identities)

    # Vendor totals must be the quote's own numbers, not doubled. Before the
    # row-collapse fix each vendor line was summed twice (RJ45 68 -> 136,
    # Cat6 UTP 60 -> 120, Cat6 STP 8 -> 16) and the doubled figure was
    # rendered to PMs as `vendor_primary_sum` in the packet certificate.
    vendor_totals = {
        (e.metadata or {})["identity"]: (e.metadata or {})["vendor_quantity"]
        for e in mat
    }
    assert vendor_totals["rj45"] == 68.0
    assert vendor_totals["cat6_utp"] == 60.0
    assert vendor_totals["cat6_stp"] == 8.0
    for e in mat:
        md = e.metadata or {}
        assert md["delta"] == md["roster_quantity"] - md["vendor_quantity"]
        # One vendor line per identity in this pack — no duplicate row nodes.
        assert len(md["vendor_atom_ids"]) == len(set(md["vendor_atom_ids"]))
    qc = [p for p in result.packets if p.family == PacketFamily.quantity_conflict]
    assert not any(
        (p.anchor_key or "") == "device:unknown" or (p.anchor_signature and p.anchor_signature.canonical_key == "device:unknown")
        for p in qc
    )
    mat_pkts = [
        p
        for p in result.packets
        if p.anchor_key
        and p.anchor_key.startswith("material:")
        and p.family in (PacketFamily.quantity_conflict, PacketFamily.vendor_mismatch)
    ]
    assert len({p.anchor_key for p in mat_pkts}) >= 3

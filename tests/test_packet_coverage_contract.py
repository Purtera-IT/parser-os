"""v57.16 atom-type -> packet coverage contract.

Every AtomType must be explicitly accounted for: either eligible to anchor
a packet family, or deliberately declared non-anchor. A new schema type
that nobody routes fails this test rather than silently orphaning into the
envelope with no deliverable section (the exact failure mode that left
commercial_total / pricing_assumption stranded for a release).

Also exercises the live wiring: commercial atoms now produce a
commercial_summary packet.

Pure-helper level — no LLM / network.
"""

from __future__ import annotations

from app.core.packetizer import (
    PACKET_ANCHOR_ELIGIBLE,
    PACKET_NON_ANCHOR,
    assert_atom_type_coverage,
    build_packets,
)
from app.core.schemas import (
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    PacketFamily,
    ReviewStatus,
    SourceRef,
)


def test_every_atom_type_is_declared():
    declared = set(PACKET_ANCHOR_ELIGIBLE) | set(PACKET_NON_ANCHOR)
    missing = set(AtomType) - declared
    assert not missing, (
        "These AtomTypes are neither anchor-eligible nor declared non-anchor "
        f"in packetizer.py — route them or mark them non-anchor: "
        f"{sorted(t.value for t in missing)}"
    )


def test_anchor_and_non_anchor_are_disjoint():
    overlap = set(PACKET_ANCHOR_ELIGIBLE) & set(PACKET_NON_ANCHOR)
    assert not overlap, sorted(t.value for t in overlap)


def test_assert_helper_passes_on_current_schema():
    assert_atom_type_coverage()  # must not raise


def test_commercial_atoms_are_anchor_eligible():
    # Regression: the original bug. Both must map to commercial_summary.
    assert PACKET_ANCHOR_ELIGIBLE[AtomType.commercial_total] is PacketFamily.commercial_summary
    assert PACKET_ANCHOR_ELIGIBLE[AtomType.pricing_assumption] is PacketFamily.commercial_summary


def _atom(atom_type, value, money_keys, atom_id):
    return EvidenceAtom(
        id=atom_id,
        project_id="p",
        artifact_id="art",
        atom_type=atom_type,
        raw_text=str(value.get("label", "")),
        normalized_text=str(value.get("label", "")).lower(),
        value=value,
        entity_keys=money_keys,
        source_refs=[
            SourceRef(
                id=f"src_{atom_id}",
                artifact_id="art",
                artifact_type="xlsx",
                filename="Deal_Kit.xlsx",
                locator={"sheet": "Deal Kit"},
                extraction_method="commercial_sheet_routing",
                parser_version="test",
            )
        ],
        receipts=[],
        authority_class=AuthorityClass.vendor_quote,
        confidence=0.7,
        confidence_raw=0.7,
        calibrated_confidence=0.7,
        review_status=ReviewStatus.needs_review,
        review_flags=[],
        parser_version="test",
    )


def test_commercial_total_atoms_produce_a_deal_economics_packet():
    atoms = [
        _atom(
            AtomType.commercial_total,
            {"label": "Total Deal Revenue 21560", "sheet_role": "financial_summary"},
            ["money:21560"],
            "atm_rev",
        ),
        _atom(
            AtomType.commercial_total,
            {"label": "Total Deal Margin 5900", "sheet_role": "financial_summary"},
            ["money:5900"],
            "atm_margin",
        ),
    ]
    packets = build_packets("p", atoms, entities=[], edges=[], attach_metadata=False)
    commercial = [p for p in packets if p.family == PacketFamily.commercial_summary]
    assert len(commercial) == 1
    pkt = commercial[0]
    ids = set(pkt.governing_atom_ids + pkt.supporting_atom_ids)
    assert {"atm_rev", "atm_margin"} <= ids


def test_pricing_rollup_summary_produces_its_own_packet():
    rollup = _atom(
        AtomType.pricing_assumption,
        {
            "is_summary": True,
            "label": "COST RATES: 312 pricing lines, $42-$1,300",
            "sheet_role": "rate_card",
            "rows": [{"row": 85, "label": "Malaysia", "money_keys": ["money:54"], "cells": ["Malaysia", "53.975"]}],
        },
        ["money:42", "money:1300"],
        "atm_rollup",
    )
    packets = build_packets("p", [rollup], entities=[], edges=[], attach_metadata=False)
    commercial = [p for p in packets if p.family == PacketFamily.commercial_summary]
    assert len(commercial) == 1
    assert "atm_rollup" in set(commercial[0].governing_atom_ids + commercial[0].supporting_atom_ids)

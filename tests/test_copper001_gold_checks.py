from __future__ import annotations

from app.eval.gold import copper_001_material_gold_checks


def test_copper_001_gold_checks_synthetic_payload_passes() -> None:
    payload = {
        "atoms": [
            {
                "id": "g1",
                "authority_class": "approved_site_roster",
                "atom_type": "quantity",
                "raw_text": "rj45",
                "value": {"quantity": 72},
            },
            {
                "id": "v1",
                "authority_class": "vendor_quote",
                "atom_type": "quantity",
                "raw_text": "rj45 vendor",
                "value": {"quantity": 68},
            },
        ],
        "packets": [
            {
                "family": "quantity_conflict",
                "anchor_key": "material:rj45",
                "reason": "72 vs 68",
                "governing_atom_ids": ["g1"],
                "supporting_atom_ids": ["g1"],
                "contradicting_atom_ids": ["v1"],
                "certificate": {
                    "existence_reason": "roster_qty=72.0 vendor_primary_sum=68.0 delta=4.0",
                    "contradiction_summary": "identity=rj45 roster_qty=72.0 vendor_primary_sum=68.0 delta=4.0",
                },
            },
            {
                "family": "vendor_mismatch",
                "anchor_key": "material:cat6_utp",
                "reason": "66 vs 60",
                "governing_atom_ids": ["g_utp"],
                "supporting_atom_ids": ["g_utp"],
                "contradicting_atom_ids": ["v_utp"],
                "certificate": {
                    "existence_reason": "66 60",
                    "contradiction_summary": "roster_qty=66 vendor_primary_sum=60",
                },
            },
            {
                "family": "vendor_mismatch",
                "anchor_key": "material:cat6_stp",
                "reason": "6 vs 8",
                "governing_atom_ids": ["g_stp"],
                "supporting_atom_ids": ["g_stp"],
                "contradicting_atom_ids": ["v_stp"],
                "certificate": {
                    "existence_reason": "6 8",
                    "contradiction_summary": "roster_qty=6 vendor_primary_sum=8",
                },
            },
        ],
        "edges": [
            {
                "edge_type": "contradicts",
                "metadata": {"comparison_basis": "aggregate_roster_vs_summed_vendor_quote"},
            }
        ],
        "entities": [],
    }
    payload["atoms"].extend(
        [
            {
                "id": "g_utp",
                "authority_class": "approved_site_roster",
                "atom_type": "quantity",
                "raw_text": "utp",
                "value": {"quantity": 66},
            },
            {
                "id": "v_utp",
                "authority_class": "vendor_quote",
                "atom_type": "quantity",
                "raw_text": "utp",
                "value": {"quantity": 60},
            },
            {
                "id": "g_stp",
                "authority_class": "approved_site_roster",
                "atom_type": "quantity",
                "raw_text": "stp",
                "value": {"quantity": 6},
            },
            {
                "id": "v_stp",
                "authority_class": "vendor_quote",
                "atom_type": "quantity",
                "raw_text": "stp",
                "value": {"quantity": 8},
            },
        ]
    )
    r = copper_001_material_gold_checks(payload)
    assert r["quantity_conflict_rj45_72_68"] is True
    assert r["vendor_mismatch_cat6_utp_66_60"] is True
    assert r["vendor_mismatch_cat6_stp_6_8"] is True
    assert r["overall_pass"] is True

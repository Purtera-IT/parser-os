from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.gold_compare import (
    GoldPacketsFile,
    compare_case_directory,
    compare_gold_to_compile,
    load_gold_packets,
    render_markdown,
    write_comparison_outputs,
)


def _good_compile() -> dict:
    return {
        "atoms": [
            {
                "id": "g1",
                "authority_class": "approved_site_roster",
                "atom_type": "quantity",
                "raw_text": "rj45 roster",
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
                "id": "p_rj",
                "family": "quantity_conflict",
                "anchor_key": "material:rj45",
                "status": "needs_review",
                "reason": "Roster 72 vs vendor 68",
                "governing_atom_ids": ["g1"],
                "supporting_atom_ids": ["g1"],
                "contradicting_atom_ids": ["v1"],
                "certificate": {
                    "existence_reason": "72 68",
                    "contradiction_summary": "roster_qty=72.0 vendor_primary_sum=68.0",
                    "governing_rationale": "approved_site_roster",
                    "authority_path": [{"authority_class": "approved_site_roster", "atom_id": "g1"}],
                    "minimal_sufficient_atom_ids": ["g1", "v1"],
                },
            }
        ],
        "entities": [],
    }


def test_comparator_passes_known_good_material_packet() -> None:
    gold = GoldPacketsFile(
        expected_packets=[
            {
                "id": "q1",
                "family": "quantity_conflict",
                "anchor_key_contains": "material:rj45",
                "must_contain_quantities": [72, 68],
                "expected_governing_authority": "approved_site_roster",
            }
        ],
        forbidden_conditions=[],
    )
    r = compare_gold_to_compile(gold=gold, compile_payload=_good_compile(), case_dir=Path("."))
    assert r.overall_pass
    assert r.expected_results[0]["found"]


def test_comparator_fails_missing_expected_packet() -> None:
    gold = GoldPacketsFile(
        expected_packets=[
            {
                "id": "ghost",
                "family": "vendor_mismatch",
                "anchor_key_contains": "material:ghost",
                "must_contain_quantities": [1, 2],
            }
        ],
        forbidden_conditions=[],
    )
    r = compare_gold_to_compile(gold=gold, compile_payload=_good_compile(), case_dir=Path("."))
    assert not r.overall_pass
    assert not r.expected_results[0]["found"]


def test_comparator_catches_vendor_quote_governs_scope() -> None:
    payload = _good_compile()
    payload["packets"].append(
        {
            "id": "bad_scope",
            "family": "scope_inclusion",
            "anchor_key": "site:west",
            "status": "active",
            "reason": "bad",
            "governing_atom_ids": ["v1"],
            "supporting_atom_ids": ["v1"],
            "contradicting_atom_ids": [],
            "certificate": {"existence_reason": "x", "governing_rationale": "y", "authority_path": []},
        }
    )
    gold = GoldPacketsFile(expected_packets=[], forbidden_conditions=["vendor_quote_governs_scope"])
    r = compare_gold_to_compile(gold=gold, compile_payload=payload, case_dir=Path("."))
    assert not r.overall_pass
    assert any(not x["passed"] for x in r.forbidden_results)


def test_comparator_catches_total_row_entity() -> None:
    payload = _good_compile()
    payload["entities"] = [
        {
            "id": "e1",
            "project_id": "p",
            "entity_type": "site",
            "canonical_key": "total",
            "canonical_name": "Grand Total",
            "aliases": [],
            "source_atom_ids": [],
            "confidence": 0.5,
            "review_status": "auto_accepted",
        }
    ]
    gold = GoldPacketsFile(expected_packets=[], forbidden_conditions=["total_row_becomes_entity"])
    r = compare_gold_to_compile(gold=gold, compile_payload=payload, case_dir=Path("."))
    assert not r.overall_pass


def test_comparator_finds_quantities_from_linked_atoms_only() -> None:
    """Quantities appear only on linked atoms, not in packet.reason."""
    compile_payload = {
        "atoms": [
            {
                "id": "ga",
                "authority_class": "approved_site_roster",
                "atom_type": "quantity",
                "raw_text": "",
                "value": {"quantity": 91.0},
            },
            {
                "id": "vb",
                "authority_class": "vendor_quote",
                "atom_type": "quantity",
                "raw_text": "",
                "value": {"quantity": 72.0},
            },
        ],
        "packets": [
            {
                "id": "pq",
                "family": "quantity_conflict",
                "anchor_key": "device:ip_camera",
                "status": "needs_review",
                "reason": "Quantity conflict between scoped work and vendor line item.",
                "governing_atom_ids": ["ga"],
                "supporting_atom_ids": ["ga"],
                "contradicting_atom_ids": ["vb"],
                "certificate": {
                    "existence_reason": "conflict",
                    "contradiction_summary": None,
                    "governing_rationale": "roster",
                    "authority_path": [],
                    "minimal_sufficient_atom_ids": ["ga", "vb"],
                },
            }
        ],
        "entities": [],
    }
    gold = GoldPacketsFile(
        expected_packets=[
            {
                "id": "atom_qty",
                "family": "quantity_conflict",
                "anchor_key_contains": "device:ip_camera",
                "must_contain_quantities": [91, 72],
                "expected_governing_authority": "approved_site_roster",
            }
        ],
        forbidden_conditions=[],
    )
    r = compare_gold_to_compile(gold=gold, compile_payload=compile_payload, case_dir=Path("."))
    assert r.overall_pass


def test_compare_case_directory_writes_outputs(tmp_path: Path) -> None:
    case = tmp_path / "CASE1"
    (case / "labels").mkdir(parents=True)
    (case / "outputs").mkdir(parents=True)
    gold = GoldPacketsFile(
        expected_packets=[
            {
                "id": "x",
                "family": "quantity_conflict",
                "anchor_key_contains": "material:rj45",
                "must_contain_quantities": [72, 68],
                "expected_governing_authority": "approved_site_roster",
            }
        ],
        forbidden_conditions=[],
    )
    (case / "labels" / "gold_packets.json").write_text(json.dumps(gold.model_dump()), encoding="utf-8")
    (case / "outputs" / "compile_result.json").write_text(json.dumps(_good_compile()), encoding="utf-8")
    result = compare_case_directory(case)
    assert result.overall_pass
    j, m = write_comparison_outputs(case, result)
    assert j.is_file() and m.is_file()
    assert "overall_pass" in render_markdown(result).lower()


def test_load_bundle_gold_fixture() -> None:
    path = Path("copper_001_pipeline_bundle/labels/gold_packets.json")
    if not path.is_file():
        pytest.skip("bundle gold fixture not present")
    g = load_gold_packets(path)
    assert len(g.expected_packets) == 7
    assert len(g.forbidden_conditions) == 6

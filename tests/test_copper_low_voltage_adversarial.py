"""
Universal adversarial regression for copper / low-voltage behavior.

Loads synthetic expectations from tests/fixtures/adversarial/copper_low_voltage/
so future fixes are not validated only against COPPER_001.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from openpyxl import Workbook

from app.core.authority import AuthorityDecision, choose_governing_atoms, compare_atoms
from app.core.graph_builder import build_edges
from app.core.item_identity import canonical_item_identity
from app.core.ids import stable_id
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EdgeType, EvidenceAtom, PacketFamily, ReviewStatus, SourceRef
from app.parsers.quote_parser import normalize_inclusion, parse_quote_quantity
from app.parsers.xlsx_parser import XlsxParser
from test_graph_builder import _atom


def _minimal_source_ref(filename: str = "adversarial_fixture.txt", locator: dict | None = None) -> SourceRef:
    return SourceRef(
        id=stable_id("src", "adv", filename),
        artifact_id="art_adv",
        artifact_type=ArtifactType.txt,
        filename=filename,
        locator=locator or {},
        extraction_method="adversarial_fixture",
        parser_version="test",
    )


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "adversarial" / "copper_low_voltage"
CASES_PATH = FIXTURE_DIR / "adversarial_cases.json"


@pytest.fixture(scope="module")
def adversarial_cases() -> dict:
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "1"
    return data


def test_fixture_pack_exists() -> None:
    assert CASES_PATH.is_file(), f"missing adversarial fixture: {CASES_PATH}"


def test_adversarial_naming_drift_rj45_and_data_drop(adversarial_cases: dict) -> None:
    for text in adversarial_cases["naming_drift_rj45"]:
        r = canonical_item_identity({"description": text})
        assert r is not None, text
        assert r.canonical_key == "rj45", (text, r.canonical_key)
    for text in adversarial_cases["naming_drift_data_drop"]:
        r = canonical_item_identity({"description": text})
        assert r is not None, text
        assert r.canonical_key == "data_drop", (text, r.canonical_key)


def test_adversarial_cat6_variants(adversarial_cases: dict) -> None:
    for row in adversarial_cases["cat6_variants"]:
        text = row["text"]
        want = row["canonical_key"]
        r = canonical_item_identity({"description": text})
        assert r is not None, text
        assert r.canonical_key == want, (text, r.canonical_key, want)


def test_adversarial_quantity_ambiguity(adversarial_cases: dict) -> None:
    for row in adversarial_cases["quantity_parse_cases"]:
        raw = row["qty"]
        exp = row["expect"]
        got = parse_quote_quantity("", raw, "", "")
        for k, v in exp.items():
            assert got.get(k) == v, (raw, k, got.get(k), v)


def test_adversarial_quote_inclusion_strings(adversarial_cases: dict) -> None:
    for row in adversarial_cases["inclusion_cases"]:
        got = normalize_inclusion(row["included"], row.get("notes") or "")
        assert got["inclusion_status"] == row["inclusion_status"], row


def test_adversarial_schedule_totals_xlsx(tmp_path: Path) -> None:
    """Total in label column (Plate ID) emits aggregate totals; notes with 'total' do not skip line."""
    path = tmp_path / "drop_schedule_adversarial.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.append(["Site", "Plate ID", "RJ45", "Cat6 UTP", "Notes"])
    ws.append(["Main", "P-01", "2", "1", ""])
    ws.append(["Main", "P-02", "1", "0", "see total roll-up later"])
    ws.append(["Main", "TOTAL", "3", "1", ""])
    wb.save(path)

    atoms = XlsxParser().parse_artifact("proj_adv", "art_adv", path)
    totals = [a for a in atoms if a.atom_type.value == "quantity" and a.value.get("source_row_type") == "total"]
    assert totals, "expected aggregate total row quantities"
    assert all(a.value.get("aggregate") is True for a in totals)
    assert not any("plate:total" in " ".join(a.entity_keys) for a in atoms if a.atom_type.value == "entity")

    note_rows = [
        a
        for a in atoms
        if a.atom_type.value == "quantity"
        and a.value.get("source_row_type") == "line_item"
        and a.source_refs
        and a.source_refs[0].locator.get("row") == 3
    ]
    assert len(note_rows) >= 1, "row 3 should emit line_item quantities despite notes mentioning total"


def test_adversarial_false_contradictions_and_aggregate_vendor(adversarial_cases: dict) -> None:
    del adversarial_cases
    a1 = _atom(
        "adv_p1",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:adv", "plate:avl_1"],
        quantity=1,
        text="RJ45 plate 1",
        value_extra={"normalized_item": "rj45"},
    )
    a2 = _atom(
        "adv_p2",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:adv", "plate:avl_2"],
        quantity=2,
        text="RJ45 plate 2",
        value_extra={"normalized_item": "rj45"},
    )
    edges_pp = build_edges("proj_adv_plates", [a1, a2], [])
    assert not any(e.edge_type == EdgeType.contradicts for e in edges_pp)

    roster = _atom(
        "adv_r",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:adv", "device:ip_camera"],
        quantity=1,
        text="rev",
        value_extra={"normalized_item": "rj45"},
    )
    revised = _atom(
        "adv_c",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.customer_current_authored,
        entity_keys=["site:adv", "device:ip_camera"],
        quantity=2,
        text="rev2",
        value_extra={"normalized_item": "rj45"},
    )
    edges_same = build_edges("proj_adv_same", [roster, revised], [])
    assert any(e.edge_type == EdgeType.contradicts for e in edges_same)

    s1 = _atom(
        "adv_s1",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:adv2", "device:ip_camera"],
        quantity=50,
        text="m",
        value_extra={"normalized_item": "rj45"},
    )
    s2 = _atom(
        "adv_s2",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.approved_site_roster,
        entity_keys=["site:adv3", "device:ip_camera"],
        quantity=41,
        text="w",
        value_extra={"normalized_item": "rj45"},
    )
    v1 = _atom(
        "adv_v1",
        atom_type=AtomType.quantity,
        authority=AuthorityClass.vendor_quote,
        entity_keys=["part:cam"],
        quantity=72,
        text="v",
        value_extra={"normalized_item": "rj45"},
    )
    edges_agg = build_edges("proj_adv_agg", [s1, s2, v1], [])
    reasons = [e.reason or "" for e in edges_agg if e.edge_type == EdgeType.contradicts]
    assert any(
        "Aggregate scoped quantity 91 does not match vendor quantity 72" in r
        or ("approved_site_roster aggregate 91" in r.lower() and "72" in r)
        for r in reasons
    ), reasons


def test_adversarial_scope_governance_authority(adversarial_cases: dict) -> None:
    del adversarial_cases
    ref = _minimal_source_ref("vendor_quote.xlsx")
    vendor_scope = EvidenceAtom(
        id="adv_vendor_scope",
        project_id="p",
        artifact_id="a",
        atom_type=AtomType.scope_item,
        raw_text="vendor scope line",
        normalized_text="vendor scope line",
        value={"context": "scope"},
        entity_keys=["site:governance_test"],
        source_refs=[ref],
        authority_class=AuthorityClass.vendor_quote,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )
    roster_scope = EvidenceAtom(
        id="adv_roster_scope",
        project_id="p",
        artifact_id="a",
        atom_type=AtomType.scope_item,
        raw_text="roster scope",
        normalized_text="roster scope",
        value={"context": "scope"},
        entity_keys=["site:governance_test"],
        source_refs=[_minimal_source_ref("site_list.xlsx")],
        authority_class=AuthorityClass.approved_site_roster,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )
    winners = choose_governing_atoms(
        [vendor_scope, roster_scope], context={"packet_family": PacketFamily.scope_inclusion}
    )
    assert [w.id for w in winners] == ["adv_roster_scope"]

    customer = EvidenceAtom(
        id="adv_cust_inst",
        project_id="p",
        artifact_id="a",
        atom_type=AtomType.customer_instruction,
        raw_text="customer says",
        normalized_text="customer says",
        value={},
        entity_keys=["site:governance_test"],
        source_refs=[_minimal_source_ref("customer_email.txt")],
        authority_class=AuthorityClass.customer_current_authored,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )
    d = compare_atoms(customer, vendor_scope, context={"packet_family": PacketFamily.scope_inclusion})
    assert isinstance(d, AuthorityDecision)
    assert d.governing_atom_id == "adv_cust_inst"
    assert "vendor quote cannot govern scope changes" in d.reason


def test_adversarial_addendum_beats_rfp_by_timestamp(adversarial_cases: dict) -> None:
    ts = adversarial_cases["governance_timestamps"]
    base_kw = dict(
        project_id="p",
        artifact_id="a",
        atom_type=AtomType.scope_item,
        raw_text="scope",
        normalized_text="scope",
        value={},
        entity_keys=["site:timeline_test"],
        authority_class=AuthorityClass.contractual_scope,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )

    def _ref(filename: str, when: str) -> list[SourceRef]:
        return [
            SourceRef(
                id=stable_id("src", filename, when),
                artifact_id="a",
                artifact_type=ArtifactType.txt,
                filename=filename,
                locator={"date": when},
                extraction_method="test",
                parser_version="test",
            )
        ]

    rfp = EvidenceAtom(
        id="scope_rfp",
        source_refs=_ref("rfp_original.txt", ts["rfp"]),
        **{k: v for k, v in base_kw.items() if k != "project_id"},
        project_id="p",
    )
    addendum = EvidenceAtom(
        id=ts["winner_id"],
        source_refs=_ref("addendum_current_scope.txt", ts["addendum"]),
        **{k: v for k, v in base_kw.items() if k != "project_id"},
        project_id="p",
    )
    winners = choose_governing_atoms([rfp, addendum], context={"packet_family": PacketFamily.scope_inclusion})
    assert winners[0].id == ts["winner_id"]


def test_adversarial_access_canonical(adversarial_cases: dict) -> None:
    for row in adversarial_cases["access_canonical"]:
        r = canonical_item_identity({"description": row["text"]})
        assert r is not None, row
        assert r.canonical_key == row["canonical_key"], row

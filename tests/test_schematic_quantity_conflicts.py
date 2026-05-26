"""PR7 — schematic quantity aggregation, edges, and packets."""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.core.ids import stable_id
from app.core.graph_builder import (
    EDGE_FAMILY_SCHEMATIC_QUANTITY_CONTRADICTION,
    build_edges,
)
from app.core.packetizer import build_packets, _valid_quantity_conflict_group
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EdgeType,
    EvidenceAtom,
    EvidenceEdge,
    ReviewStatus,
    SourceRef,
)
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser


def _build_drawing_with_mismatch(path: Path, ptz_marks: int) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    # Legend declares 7 PTZ cameras
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((300, 110), "7", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)

    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    # But only ptz_marks instances on the body. Place on a 3-column
    # grid so high counts (up to 12) still fit inside the page.
    for i in range(ptz_marks):
        col = i % 3
        row = i // 3
        x = 100 + col * 150
        y = 200 + row * 80
        page.insert_text((x, y), "PTZ", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(path))
    doc.close()


def _parse(path: Path):
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art_id = stable_id("art", str(path))
    return parser.parse_artifact("proj_test", art_id, path, domain_pack=pack), pack


def test_quantity_atoms_emitted_for_detected_and_declared(tmp_path: Path) -> None:
    pdf = tmp_path / "mismatch.pdf"
    _build_drawing_with_mismatch(pdf, ptz_marks=3)
    out, _ = _parse(pdf)
    qty = [a for a in out.atoms if a.atom_type == AtomType.quantity and a.value.get("schematic_target_key")]
    roles = {a.value["schematic_role"] for a in qty}
    assert {"detected", "declared"}.issubset(roles), roles
    detected = [a for a in qty if a.value["schematic_role"] == "detected"][0]
    declared = [a for a in qty if a.value["schematic_role"] == "declared"][0]
    assert detected.value["quantity"] == 3
    assert float(declared.value["quantity"]) == 7.0


def test_schematic_quantity_atoms_carry_bbox_provenance(tmp_path: Path) -> None:
    pdf = tmp_path / "mismatch.pdf"
    _build_drawing_with_mismatch(pdf, ptz_marks=3)
    out, _ = _parse(pdf)
    qty = [a for a in out.atoms if a.atom_type == AtomType.quantity and a.value.get("schematic_target_key")]
    assert qty
    for atom in qty:
        loc = atom.source_refs[0].locator
        assert len(loc["bbox"]) == 4
        assert loc["bbox_units"] == "pdf_points"


def test_graph_builder_emits_schematic_contradiction_edge(tmp_path: Path) -> None:
    pdf = tmp_path / "mismatch.pdf"
    _build_drawing_with_mismatch(pdf, ptz_marks=3)
    out, _ = _parse(pdf)
    edges = build_edges("proj_test", out.atoms, entities=[])
    schematic_edges = [
        e for e in edges
        if e.metadata.get("edge_family") == EDGE_FAMILY_SCHEMATIC_QUANTITY_CONTRADICTION
    ]
    assert schematic_edges, "expected at least one schematic_quantity_contradiction edge"
    e = schematic_edges[0]
    assert e.edge_type == EdgeType.contradicts


def test_matching_counts_produce_no_contradiction(tmp_path: Path) -> None:
    pdf = tmp_path / "match.pdf"
    _build_drawing_with_mismatch(pdf, ptz_marks=7)  # matches declared count
    out, _ = _parse(pdf)
    edges = build_edges("proj_test", out.atoms, entities=[])
    schematic_edges = [
        e for e in edges
        if e.metadata.get("edge_family") == EDGE_FAMILY_SCHEMATIC_QUANTITY_CONTRADICTION
    ]
    assert not schematic_edges


def test_packetizer_certifies_schematic_quantity_conflict(tmp_path: Path) -> None:
    pdf = tmp_path / "mismatch.pdf"
    _build_drawing_with_mismatch(pdf, ptz_marks=3)
    out, _ = _parse(pdf)
    edges = build_edges("proj_test", out.atoms, entities=[])
    packets = build_packets("proj_test", out.atoms, [], edges)
    qc_packets = [p for p in packets if p.family.value == "quantity_conflict"]
    schematic_qc = [
        p for p in qc_packets
        if any(
            (a.value.get("schematic_target_key") if isinstance(a.value, dict) else None)
            for a in out.atoms
            if a.id in (p.contradicting_atom_ids or p.governing_atom_ids)
        )
    ]
    assert schematic_qc, "schematic quantity_conflict packet not certified"


def test_packetizer_still_rejects_single_source_non_schematic_qty_conflict() -> None:
    # Build two same-artifact, same-authority, non-schematic quantity atoms
    # with a synthetic contradicts edge. The existing gate must still reject
    # this as "single source arguing with itself."
    src = SourceRef(
        id="sr1",
        artifact_id="art_a",
        artifact_type=ArtifactType.xlsx,
        filename="bom.xlsx",
        locator={"sheet": "Sheet1", "row": 3, "columns": {"a": "A"}},
        extraction_method="xlsx",
        parser_version="v1",
    )

    def _qty(idx: int, val: int) -> EvidenceAtom:
        return EvidenceAtom(
            id=f"atom_qty_{idx}",
            project_id="p",
            artifact_id="art_a",
            atom_type=AtomType.quantity,
            raw_text=f"qty {val}",
            normalized_text=f"qty {val}",
            value={"quantity": val},
            entity_keys=["device:thing"],
            source_refs=[src],
            authority_class=AuthorityClass.machine_extractor,
            confidence=0.9,
            review_status=ReviewStatus.auto_accepted,
            parser_version="v1",
        )

    a, b = _qty(1, 3), _qty(2, 5)
    edge = EvidenceEdge(
        id="edge_1",
        project_id="p",
        from_atom_id=a.id,
        to_atom_id=b.id,
        edge_type=EdgeType.contradicts,
        reason="manual",
        confidence=0.9,
        metadata={"edge_family": "quantity_contradiction"},
    )
    assert not _valid_quantity_conflict_group([a, b], [edge])


def test_quantity_atom_replay_verifies_under_bbox_locator(tmp_path: Path) -> None:
    pdf = tmp_path / "mismatch.pdf"
    _build_drawing_with_mismatch(pdf, ptz_marks=3)
    out, _ = _parse(pdf)
    from app.core.source_replay import replay_source_ref

    art_id = next(iter({a.artifact_id for a in out.atoms}))
    detected = [
        a for a in out.atoms
        if a.atom_type == AtomType.quantity
        and a.value.get("schematic_role") == "detected"
    ]
    assert detected
    # The detected atom's bbox = the first detection's bbox, so replay
    # should verify the same crop the detector hashed at emission time.
    for atom in detected:
        src = atom.source_refs[0]
        receipt = replay_source_ref(atom, src, {art_id: pdf})
        assert receipt.replay_status == "verified", (
            f"quantity atom replay failed: {receipt.reason}"
        )

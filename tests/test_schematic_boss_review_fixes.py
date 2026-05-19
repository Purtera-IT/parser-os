"""Round-2 boss-review fix coverage.

Each test pins down a specific issue the audit flagged so the bug
class can't quietly come back.  Test naming follows the audit's
phrasing where possible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

fitz = pytest.importorskip("fitz")

from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.core.source_replay import replay_source_ref
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
from app.parsers.schematic_atom_emitters import build_replayable_locator
from app.parsers.schematic_models import BBOX_UNITS_PDF_POINTS


def _build_camera_pdf(path: Path, ptz_marks: int = 3, declared: int = 7) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((300, 110), str(declared), fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)

    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    for i in range(ptz_marks):
        col, row = i % 3, i // 3
        page.insert_text((100 + col * 150, 200 + row * 80), "PTZ", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(path))
    doc.close()


def _parse(pdf: Path):
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    return parser.parse_artifact("p", art, pdf, domain_pack=pack), art


# ─── provenance hardening ───


def test_every_schematic_atom_carries_page_locator(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _build_camera_pdf(pdf)
    out, _ = _parse(pdf)
    schematic = [a for a in out.atoms if a.atom_type.value.startswith("schematic_")]
    assert schematic
    for atom in schematic:
        assert atom.source_refs, atom.id
        loc = atom.source_refs[0].locator
        assert isinstance(loc.get("page"), int), (atom.id, atom.atom_type.value, loc)


def test_legend_atom_carries_bbox_and_crop_hash(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _build_camera_pdf(pdf)
    out, _ = _parse(pdf)
    legend_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_legend]
    assert legend_atoms
    loc = legend_atoms[0].source_refs[0].locator
    assert loc.get("bbox_units") == BBOX_UNITS_PDF_POINTS
    assert len(loc.get("bbox") or []) == 4
    assert loc.get("crop_sha256"), loc


def test_target_set_atom_carries_replayable_locator(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _build_camera_pdf(pdf)
    out, _ = _parse(pdf)
    ts_atoms = [a for a in out.atoms if a.atom_type == AtomType.schematic_detection_target_set]
    assert ts_atoms
    loc = ts_atoms[0].source_refs[0].locator
    assert isinstance(loc.get("page"), int)
    # Page bbox + crop hash should be present because we passed the
    # whole page rect through build_replayable_locator.
    assert len(loc.get("bbox") or []) == 4
    assert loc.get("crop_sha256")


def test_declared_count_atom_carries_crop_hash(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _build_camera_pdf(pdf)
    out, _ = _parse(pdf)
    declared = [
        a for a in out.atoms
        if a.atom_type == AtomType.quantity
        and a.value.get("schematic_role") == "declared"
    ]
    assert declared
    loc = declared[0].source_refs[0].locator
    assert loc.get("crop_sha256"), loc
    assert len(loc.get("bbox") or []) == 4
    assert loc.get("bbox") != [0.0, 0.0, 1.0, 1.0], "declared bbox must not be the fake fallback"


def test_declared_count_replay_verifies(tmp_path: Path) -> None:
    pdf = tmp_path / "drawing.pdf"
    _build_camera_pdf(pdf)
    out, art = _parse(pdf)
    declared = [
        a for a in out.atoms
        if a.atom_type == AtomType.quantity
        and a.value.get("schematic_role") == "declared"
    ]
    assert declared
    paths = {art: pdf}
    for atom in declared:
        receipt = replay_source_ref(atom, atom.source_refs[0], paths)
        assert receipt.replay_status == "verified", receipt.reason


# ─── source_replay clamps bbox ───


def test_replay_clamps_bbox_to_page_bounds(tmp_path: Path) -> None:
    from app.core.schemas import (
        ArtifactType,
        AtomType,
        AuthorityClass,
        EvidenceAtom,
        ReviewStatus,
        SourceRef,
    )

    pdf = tmp_path / "drawing.pdf"
    _build_camera_pdf(pdf)
    art = stable_id("art", str(pdf))
    # Stretch the bbox to extend slightly beyond page width — must clamp.
    src = SourceRef(
        id="sr_clamp",
        artifact_id=art,
        artifact_type=ArtifactType.pdf,
        filename=pdf.name,
        locator={
            "page": 0,
            "bbox": [50.0, 50.0, 99999.0, 99999.0],
            "bbox_units": "pdf_points",
            "crop_sha256": "0" * 64,
        },
        extraction_method="test",
        parser_version="t",
    )
    atom = EvidenceAtom(
        id="atom_clamp",
        project_id="p",
        artifact_id=art,
        atom_type=AtomType.schematic_symbol_detection,
        raw_text="x",
        normalized_text="x",
        value={},
        entity_keys=[],
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        parser_version="t",
    )
    receipt = replay_source_ref(atom, src, {art: pdf})
    # Should be ``failed`` due to hash mismatch, NOT crash with an exception.
    assert receipt.replay_status == "failed"
    assert "hash mismatch" in receipt.reason or "crop" in receipt.reason


def test_replay_rejects_bbox_entirely_off_page(tmp_path: Path) -> None:
    from app.core.schemas import (
        ArtifactType,
        AtomType,
        AuthorityClass,
        EvidenceAtom,
        ReviewStatus,
        SourceRef,
    )

    pdf = tmp_path / "drawing.pdf"
    _build_camera_pdf(pdf)
    art = stable_id("art", str(pdf))
    src = SourceRef(
        id="sr_off",
        artifact_id=art,
        artifact_type=ArtifactType.pdf,
        filename=pdf.name,
        locator={
            "page": 0,
            "bbox": [99000.0, 99000.0, 99500.0, 99500.0],
            "bbox_units": "pdf_points",
            "crop_sha256": "0" * 64,
        },
        extraction_method="test",
        parser_version="t",
    )
    atom = EvidenceAtom(
        id="atom_off",
        project_id="p",
        artifact_id=art,
        atom_type=AtomType.schematic_symbol_detection,
        raw_text="x",
        normalized_text="x",
        value={},
        entity_keys=[],
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        parser_version="t",
    )
    receipt = replay_source_ref(atom, src, {art: pdf})
    assert receipt.replay_status == "failed"
    assert "outside" in receipt.reason.lower()


# ─── legend_orphan ───


def test_legend_orphan_emitted_when_legend_entry_has_zero_detections(tmp_path: Path) -> None:
    """Build a PDF where the legend declares PTZ but the floor plan
    body never paints any PTZ tokens. The parser must emit a
    legend_orphan warning for the load-bearing target.
    """
    pdf = tmp_path / "orphan.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN (NO CAMERAS DRAWN)", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    out, _ = _parse(pdf)
    warnings = [a for a in out.atoms if a.atom_type == AtomType.schematic_warning]
    types = {w.value["warning_type"] for w in warnings}
    assert "legend_orphan" in types, [w.value["warning_type"] for w in warnings]


# ─── unknown_symbol suppression ───


def test_unknown_symbol_ignores_sheet_number_repeats(tmp_path: Path) -> None:
    """A page that repeats its own sheet number five times in the body
    (matrix sheet ref, drawing index, title block, revision block, etc.)
    must not flag the sheet number as an unknown symbol.
    """
    pdf = tmp_path / "noisy.pdf"
    doc = fitz.open()
    # Legend page so the schematic flow fires.
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    # Drawing page — repeats its own sheet number 5 times.
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    for i, x in enumerate([100, 200, 300, 400, 500]):
        page.insert_text((x, 200 + i * 30), "E1.01", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    out, _ = _parse(pdf)
    unknowns = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_warning
        and a.value.get("warning_type") == "unknown_symbol"
    ]
    # The page's own sheet number must NOT be flagged as unknown.
    for w in unknowns:
        assert "E1.01" not in str(w.value.get("detail", "")), w.value


def test_unknown_symbol_ignores_single_letter_grid_bubbles(tmp_path: Path) -> None:
    pdf = tmp_path / "grids.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    # Five grid bubbles (single letters A B C D E) repeated.
    for i, letter in enumerate("ABCDE"):
        for j in range(4):
            page.insert_text((100 + j * 100, 200 + i * 40), letter, fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    out, _ = _parse(pdf)
    unknowns = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_warning
        and a.value.get("warning_type") == "unknown_symbol"
    ]
    flagged_tokens = {
        w.value.get("detail", "").split()[1].strip("'") if w.value.get("detail") else ""
        for w in unknowns
    }
    # Single-letter grid bubbles must not be in there.
    assert not (flagged_tokens & set("ABCDE")), flagged_tokens


# ─── prepass-failure surfacing ───


def test_prepass_exception_surfaces_as_warning_atom(tmp_path: Path) -> None:
    """Force the schematic pre-pass to raise. The parser must emit a
    schematic_warning atom describing the failure rather than silently
    dropping every schematic atom.
    """
    from app.parsers import orbitbrief_pdf as op

    pdf = tmp_path / "drawing.pdf"
    _build_camera_pdf(pdf)

    original = op._run_schematic_pre_pass

    def _boom(**kwargs: Any):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated schematic boom")

    op._run_schematic_pre_pass = _boom  # type: ignore[assignment]
    try:
        out, _ = _parse(pdf)
    finally:
        op._run_schematic_pre_pass = original  # type: ignore[assignment]

    warnings = [a for a in out.atoms if a.atom_type == AtomType.schematic_warning]
    assert warnings, "prepass failure must surface as a warning atom"
    details = {w.value.get("detail", "") for w in warnings}
    assert any("simulated schematic boom" in d for d in details), details


# ─── sheet-number top-left noise regression (round-2 fix) ───


def test_pre_pass_resolves_correct_sheet_on_page_with_reference_noise(tmp_path: Path) -> None:
    """Drawing pages cite OTHER sheet numbers in body text ("see sheet
    T0.01"); the parser must still resolve THIS page to its own
    bottom-right title-block sheet number, not the one it references.
    """
    pdf = tmp_path / "noisy_refs.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    # Top-left mentions T0.01 multiple times.
    page.insert_text((72, 60), "Refer to sheet T0.01 (legend), T0.02 (general notes)", fontsize=10)
    page.insert_text((72, 80), "T0.01 contains the symbol legend.", fontsize=10)
    page.insert_text((500, 740), "E1.04", fontsize=10)  # actual title block
    doc.save(str(pdf))
    doc.close()
    out, _ = _parse(pdf)
    targets = [a for a in out.atoms if a.atom_type == AtomType.schematic_detection_target_set]
    assert targets
    sheets = {a.value.get("sheet_number") for a in targets}
    assert "E1.04" in sheets, f"target set sheets: {sheets}"


# ─── packetizer narrow exception still narrow ───


def test_packetizer_rejects_schematic_atom_without_crop_hash() -> None:
    """The tightened gate should refuse a schematic-shaped quantity pair
    when one side lacks crop_sha256, even though both bbox and
    extraction_method are present.
    """
    from app.core.packetizer import _is_schematic_quantity_group
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

    def _atom(role: str, qty: float, crop: str | None) -> EvidenceAtom:
        loc = {
            "page": 0,
            "bbox": [10.0, 10.0, 30.0, 30.0],
            "bbox_units": "pdf_points",
            "schematic_target_key": "x",
            "schematic_role": role,
        }
        if crop is not None:
            loc["crop_sha256"] = crop
        return EvidenceAtom(
            id=f"atom_{role}_{qty}",
            project_id="p",
            artifact_id="art",
            atom_type=AtomType.quantity,
            raw_text=f"{role} {qty}",
            normalized_text=f"{role} {qty}",
            value={"quantity": qty, "schematic_target_key": "x", "schematic_role": role},
            entity_keys=[],
            source_refs=[
                SourceRef(
                    id=f"sr_{role}",
                    artifact_id="art",
                    artifact_type=ArtifactType.pdf,
                    filename="x.pdf",
                    locator=loc,
                    extraction_method=f"schematic_{role}_count",
                    parser_version="v",
                )
            ],
            authority_class=AuthorityClass.machine_extractor,
            confidence=0.9,
            review_status=ReviewStatus.auto_accepted,
            parser_version="v",
        )

    edge = EvidenceEdge(
        id="e",
        project_id="p",
        from_atom_id="a",
        to_atom_id="b",
        edge_type=EdgeType.contradicts,
        reason="",
        confidence=0.9,
        metadata={"edge_family": "schematic_quantity_contradiction"},
    )
    bad = [_atom("detected", 3, "deadbeef"), _atom("declared", 7, None)]
    assert not _is_schematic_quantity_group(bad, [edge])
    good = [_atom("detected", 3, "deadbeef"), _atom("declared", 7, "cafef00d")]
    assert _is_schematic_quantity_group(good, [edge])


def test_packetizer_rejects_schematic_atom_with_wrong_extraction_method() -> None:
    from app.core.packetizer import _is_schematic_quantity_group
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

    def _atom(role: str, method: str) -> EvidenceAtom:
        return EvidenceAtom(
            id=f"a_{role}_{method}",
            project_id="p",
            artifact_id="art",
            atom_type=AtomType.quantity,
            raw_text="x",
            normalized_text="x",
            value={"quantity": 1, "schematic_target_key": "x", "schematic_role": role},
            entity_keys=[],
            source_refs=[
                SourceRef(
                    id=f"sr_{role}",
                    artifact_id="art",
                    artifact_type=ArtifactType.pdf,
                    filename="x.pdf",
                    locator={
                        "page": 0,
                        "bbox": [0.0, 0.0, 10.0, 10.0],
                        "bbox_units": "pdf_points",
                        "crop_sha256": "deadbeef",
                    },
                    extraction_method=method,
                    parser_version="v",
                )
            ],
            authority_class=AuthorityClass.machine_extractor,
            confidence=0.9,
            review_status=ReviewStatus.auto_accepted,
            parser_version="v",
        )

    edge = EvidenceEdge(
        id="e",
        project_id="p",
        from_atom_id="a",
        to_atom_id="b",
        edge_type=EdgeType.contradicts,
        reason="",
        confidence=0.9,
        metadata={"edge_family": "schematic_quantity_contradiction"},
    )
    bad = [
        _atom("detected", "spreadsheet_row"),  # not schematic_
        _atom("declared", "schematic_declared_count"),
    ]
    assert not _is_schematic_quantity_group(bad, [edge])


# ─── build_replayable_locator helper ───


def test_build_replayable_locator_falls_back_when_page_unavailable() -> None:
    loc = build_replayable_locator(
        page_index=4,
        bbox=(10.0, 20.0, 30.0, 50.0),
        page=None,
    )
    assert loc["page"] == 4
    assert loc["bbox"] == [10.0, 20.0, 30.0, 50.0]
    # No page handle: no crop hash. That's expected — the helper
    # documents the degraded path so callers know it's not replayable.
    assert "crop_sha256" not in loc


def test_build_replayable_locator_returns_page_only_when_no_bbox() -> None:
    loc = build_replayable_locator(page_index=3, bbox=None, page=None)
    assert loc == {"page": 3}

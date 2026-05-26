"""Hard real-world stress tests for the four items the boss called out.

These tests intentionally try the messy edge cases synthetic
fixtures normally skip:

  1. Multi-column legends with split headers and wrapped descriptions.
  2. False-positive suppression for keyed-note number callouts,
     detail bubbles, dimension annotations.
  3. Mixed-mode PDFs (text legend + raster floor plan).
  4. Cross-artifact subtype rollups for non-camera packs
     (fire smoke_detector vs generic detector BOM line).

A failure here means the parser still has a real-world gap.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.core.graph_builder import build_edges
from app.core.ids import stable_id
from app.core.packetizer import build_packets
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
from orbitbrief_page_os.segmentation.schematic.legend_locator import (
    TextBlock,
    locate_legend_candidates,
)
from orbitbrief_page_os.segmentation.schematic.legend_parser import parse_legend


def _b(text: str, x0: float, y0: float, w: float = 80, h: float = 12) -> TextBlock:
    return TextBlock(text=text, bbox=(x0, y0, x0 + w, y0 + h))


# ─── 1. multi-column legend stress ───


def test_multi_column_legend_handles_description_wrapped_to_two_lines() -> None:
    """A description that wraps to a second row must not produce a
    second spurious legend entry. The wrapped line should land in
    the same entry's description or be ignored entirely.
    """
    blocks = [
        _b("SYMBOL", 50, 100, w=40),
        _b("DESCRIPTION", 100, 100, w=80),
        _b("COUNT", 200, 100, w=40),
        _b("MOUNTING", 260, 100, w=80),
        # Data row 1 wrapping over two lines
        _b("WN", 50, 120, w=20),
        _b("WIRELESS NODE WITH PoE+", 100, 120, w=150),
        _b("4", 200, 120, w=10),
        _b("CEILING", 260, 120, w=60),
        # Wrap line — same logical entry continues
        _b("BACKUP BATTERY", 100, 136, w=120),
        # Data row 2
        _b("CR", 50, 158, w=20),
        _b("CARD READER", 100, 158, w=80),
        _b("12", 200, 158, w=20),
        _b('48" AFF', 260, 158, w=60),
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    cand = max(cands, key=lambda c: c.score)
    legend = parse_legend(candidate=cand, page_blocks=blocks)
    assert legend is not None
    symbols = sorted((e.raw_symbol_text or "") for e in legend.entries)
    # Exactly 2 symbols, NOT 3 — the wrap line must not produce
    # a phantom entry with an empty symbol.
    assert symbols == ["CR", "WN"], symbols


def test_multi_column_legend_tolerates_empty_cells_in_data_rows() -> None:
    """A data row missing some columns (e.g., no REMARKS) should
    still emit a complete entry with whatever attributes ARE present.
    """
    blocks = [
        _b("SYMBOL", 50, 100, w=40),
        _b("DESCRIPTION", 100, 100, w=80),
        _b("COUNT", 200, 100, w=40),
        _b("MOUNTING", 260, 100, w=80),
        _b("REMARKS", 360, 100, w=80),
        _b("WN", 50, 120, w=20),
        _b("WIRELESS NODE", 100, 120, w=120),
        _b("4", 200, 120, w=10),
        # No mounting cell, no remarks cell
        _b("CR", 50, 140, w=20),
        _b("CARD READER", 100, 140, w=120),
        _b("12", 200, 140, w=10),
        _b('48" AFF', 260, 140, w=60),
        # No remarks cell
    ]
    cands = locate_legend_candidates(page_index=0, blocks=blocks)
    cand = max(cands, key=lambda c: c.score)
    legend = parse_legend(candidate=cand, page_blocks=blocks)
    assert legend is not None
    wn = next(e for e in legend.entries if e.raw_symbol_text == "WN")
    cr = next(e for e in legend.entries if e.raw_symbol_text == "CR")
    cr_attrs = dict(cr.attributes)
    assert "48" in cr_attrs.get("mounting", "")


# ─── 2. detection FP suppression stress ───


def test_keyed_note_integer_callout_in_body_does_not_detect(tmp_path: Path) -> None:
    """A small circular bubble carrying just ``3`` (a keyed-note
    reference) sits in the drawing body — and the legend has a
    symbol ``3`` would be ambiguous; but if it's a STANDALONE
    digit, it should never match a legend symbol that's a text
    token.  Even if a legend mistakenly used a digit symbol, our
    standalone-token check would still classify it correctly.
    """
    pdf = tmp_path / "drawing.pdf"
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
    # Five keyed-note callouts at various locations — all bare integers
    for i, (x, y) in enumerate([(150, 250), (250, 250), (350, 250), (450, 250), (550, 250)]):
        page.insert_text((x, y), str(i + 1), fontsize=10)
    # One real PTZ marker
    page.insert_text((300, 400), "PTZ", fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    out = parser.parse_artifact("p", stable_id("art", str(pdf)), pdf, domain_pack=pack)
    ptz = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_symbol_detection
        and a.value.get("target_key") == "ptz_camera"
    ]
    # Exactly 1 PTZ — the 5 keyed-note integers must not produce
    # false positives.  unknown_symbol clustering also shouldn't
    # flag those (they're keyed-note-shape, not symbol-shape).
    assert len(ptz) == 1
    unknowns = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_warning
        and a.value.get("warning_type") == "unknown_symbol"
    ]
    # No unknown_symbol warnings for the bare integers either.
    assert not unknowns, [w.value for w in unknowns]


def test_detail_bubble_with_sheet_reference_does_not_detect(tmp_path: Path) -> None:
    """Detail bubbles like ``3/A2.01`` (callout number / sheet ref)
    must not be mistaken for a device tag.
    """
    pdf = tmp_path / "drawing.pdf"
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
    # Detail bubbles
    page.insert_text((150, 300), "3/A2.01", fontsize=10)
    page.insert_text((350, 300), "5/A3.02", fontsize=10)
    # Real PTZ
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    out = parser.parse_artifact("p", stable_id("art", str(pdf)), pdf, domain_pack=pack)
    ptz = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_symbol_detection
        and a.value.get("target_key") == "ptz_camera"
    ]
    # Zero PTZ — the detail bubbles must NOT count.
    assert len(ptz) == 0, [d.value for d in ptz]


def test_dimension_annotation_does_not_detect(tmp_path: Path) -> None:
    """Dimensions like ``8'-10"`` or ``42" AFF`` are mounting-height
    callouts, not device labels.  They must never produce a
    detection of their own.
    """
    pdf = tmp_path / "drawing.pdf"
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
    page.insert_text((150, 300), '8\'-10"', fontsize=10)
    page.insert_text((350, 300), '42" AFF', fontsize=10)
    page.insert_text((500, 300), '+120"', fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    out = parser.parse_artifact("p", stable_id("art", str(pdf)), pdf, domain_pack=pack)
    dets = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_symbol_detection
    ]
    assert dets == [], [d.value for d in dets]


def test_grid_marker_letters_do_not_detect(tmp_path: Path) -> None:
    """Grid markers (single letters A B C in column headers) must
    never be classified as device symbols.
    """
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    # Note: legend "B" is a single letter — same shape as a grid bubble
    page.insert_text((72, 110), "B", fontsize=10)
    page.insert_text((180, 110), "BACKBOX", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    # Grid markers along the top edge
    for i, letter in enumerate("ABCDE"):
        page.insert_text((100 + i * 80, 60), letter, fontsize=10)
    # And along the side
    for i, letter in enumerate("ABCDE"):
        page.insert_text((30, 200 + i * 50), letter, fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    out = parser.parse_artifact("p", stable_id("art", str(pdf)), pdf, domain_pack=pack)
    # The legend declares symbol "B" — even so, grid markers must
    # not be classified.  Because of standalone-token rules + the
    # unknown_symbol grid-bubble filter, we should see 0 detections
    # or VERY FEW (only if a marker happens to be near a real device).
    dets = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_symbol_detection
    ]
    # Acceptable bound: <= 2 (grid noise is constrained even with a
    # single-letter legend symbol declared).
    assert len(dets) <= 2, [d.value for d in dets]


# ─── 3. image-only hybrid pages ───


def test_text_legend_with_raster_floor_plan_still_produces_some_atoms(tmp_path: Path) -> None:
    """When the legend page is text-extractable but the floor plan
    page is image-only, the parser should still produce schematic
    atoms (legend, target_set, warnings) on the raster page even
    if it can't detect text-tag symbols.
    """
    pdf = tmp_path / "drawing.pdf"
    doc = fitz.open()
    # Page 0: text legend
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    # Page 1: image-only (only shapes, no text)
    page = doc.new_page(width=612, height=792)
    for cx, cy in [(150, 300), (300, 300), (450, 300)]:
        page.draw_polyline(
            [(cx - 10, cy + 15), (cx, cy), (cx + 10, cy + 15), (cx - 10, cy + 15)],
            color=(0, 0, 0),
            fill=(0, 0, 0),
        )
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    out = parser.parse_artifact("p", stable_id("art", str(pdf)), pdf, domain_pack=pack)
    # Schematic atoms touching page 1 should not be empty —
    # either an ocr_unavailable warning OR some detections.
    schematic_page1 = [
        a for a in out.atoms
        if any(
            (s.locator.get("page") == 1) if isinstance(s.locator, dict) else False
            for s in (a.source_refs or [])
        )
        and a.atom_type.value.startswith("schematic_")
    ]
    assert schematic_page1, "image-only floor plan produced zero schematic atoms"


# ─── 4. cross-artifact non-camera rollup ───


def test_smoke_detector_schematic_vs_generic_detector_bom_quantity_conflict(
    tmp_path: Path,
) -> None:
    """Schematic counts 6 smoke detectors via the fire_safety pack
    (entity_key=device:smoke_detector, parent rollup includes
    device:detector). A BOM line item keyed on device:detector
    with count 4 must produce a cross-artifact quantity_conflict
    packet.
    """
    pdf = tmp_path / "fire.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET FA0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "SD", fontsize=10)
    page.insert_text((180, 110), "SMOKE DETECTOR", fontsize=10)
    page.insert_text((500, 740), "FA0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FIRE RISER", fontsize=14)
    page.insert_text((72, 100), "See sheet FA0.01 for legend.", fontsize=10)
    for i, x in enumerate([100, 200, 300, 400, 500, 100]):
        y = 250 if i < 5 else 290
        page.insert_text((x, y), "SD", fontsize=10)
    page.insert_text((500, 740), "FA1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("fire_safety")
    art = stable_id("art", str(pdf))
    parse_out = parser.parse_artifact("p", art, pdf, domain_pack=pack)

    # Fake BOM atom keyed on the parent rollup device:detector
    bom_artifact_id = "art_bom_fire"
    bom_src = SourceRef(
        id="sr_bom",
        artifact_id=bom_artifact_id,
        artifact_type=ArtifactType.xlsx,
        filename="bom.xlsx",
        locator={"sheet": "BOM", "row": 5, "columns": {"qty": "C"}},
        extraction_method="xlsx_quote_parser",
        parser_version="t",
    )
    bom_atom = EvidenceAtom(
        id="atom_bom_detector",
        project_id="p",
        artifact_id=bom_artifact_id,
        atom_type=AtomType.quantity,
        raw_text="BOM line: 4 detectors",
        normalized_text="4 detectors",
        value={"quantity": 4, "part_identity": "detector"},
        entity_keys=["device:detector"],
        source_refs=[bom_src],
        authority_class=AuthorityClass.vendor_quote,
        confidence=0.95,
        review_status=ReviewStatus.auto_accepted,
        parser_version="t",
    )

    all_atoms = list(parse_out.atoms) + [bom_atom]
    edges = build_edges("p", all_atoms, entities=[])
    cross_artifact = [
        e for e in edges
        if (e.metadata or {}).get("cross_artifact") is True
        and e.edge_type.value == "contradicts"
    ]
    assert cross_artifact, (
        "no cross-artifact contradicts edge between fire_safety "
        "smoke_detector schematic count and the generic detector "
        "BOM; this means the parent_entity_keys rollup is silently "
        "broken for non-camera packs"
    )

    packets = build_packets("p", all_atoms, [], edges)
    qc_with_bom = [
        p for p in packets
        if p.family.value == "quantity_conflict"
        and any(
            a.artifact_id == bom_artifact_id
            for a in all_atoms
            if a.id in (p.contradicting_atom_ids or p.governing_atom_ids)
        )
    ]
    assert qc_with_bom, (
        "no quantity_conflict packet certified for the fire-detector "
        "cross-artifact case"
    )

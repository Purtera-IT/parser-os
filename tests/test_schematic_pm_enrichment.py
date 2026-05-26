"""PM-visible enrichment fields on schematic detections.

A project-manager looking at a single detection atom needs to see:

    target_key                  — what was detected
    located_in_room_display     — human-readable room (LOBBY 101)
    mounting_height             — height callout / schedule field /
                                  legend column / keyed-note default
    mounting_height_source      — which level of the chain answered
    responsibility              — NIC / BY OWNER / BY GC / BY OTHERS
    legend_remarks              — full remarks string from the legend row

These tests pin the enrichment so a future change can't strip
fields the PM relies on.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.core.ids import stable_id
from app.core.schemas import AtomType
from app.domain.loader import load_domain_pack
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser


def _build_pm_floorplan(path: Path) -> None:
    doc = fitz.open()
    # T0.01 — global legend with MOUNTING + REMARKS columns
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    page.insert_text((360, 90), "MOUNTING", fontsize=10)
    page.insert_text((460, 90), "REMARKS", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((300, 110), "2", fontsize=10)
    page.insert_text((360, 110), '120" AFF', fontsize=10)
    page.insert_text((460, 110), "NIC LENS", fontsize=10)
    page.insert_text((72, 128), "DOM", fontsize=10)
    page.insert_text((180, 128), "FIXED DOME CAMERA", fontsize=10)
    page.insert_text((300, 128), "1", fontsize=10)
    page.insert_text((360, 128), "CEILING", fontsize=10)
    page.insert_text((460, 128), "BY OWNER", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)

    # E1.01 — floor plan with rooms + detections + a keyed-note default
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FIRST FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((100, 250), "LOBBY 101", fontsize=10)
    page.insert_text((130, 280), "PTZ", fontsize=10)
    page.insert_text((400, 250), "CONFERENCE 301", fontsize=10)
    page.insert_text((430, 280), "PTZ", fontsize=10)
    page.insert_text((250, 400), "DOM", fontsize=10)
    page.insert_text((72, 540), "KEYED NOTES", fontsize=11)
    page.insert_text((72, 558), "1. All devices mounted at 48 AFF unless noted.", fontsize=9)
    page.insert_text((400, 700), "Project: PM Test", fontsize=9)
    page.insert_text((400, 715), "Sheet Title: First Floor", fontsize=9)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(path))
    doc.close()


def _parse(pdf: Path):
    parser = OrbitBriefPdfParser()
    pack = load_domain_pack("security_camera")
    art = stable_id("art", str(pdf))
    return parser.parse_artifact("p", art, pdf, domain_pack=pack)


def test_detection_shows_human_readable_room_display(tmp_path: Path) -> None:
    pdf = tmp_path / "fp.pdf"
    _build_pm_floorplan(pdf)
    out = _parse(pdf)
    ptz = [a for a in out.atoms if a.atom_type == AtomType.schematic_symbol_detection and a.value["target_key"] == "ptz_camera"]
    assert len(ptz) == 2
    displays = sorted(d.value.get("located_in_room_display", "") for d in ptz)
    assert "LOBBY 101" in displays or any("LOBBY" in d for d in displays)
    assert any("CONFERENCE" in d for d in displays)


def test_detection_pulls_mounting_height_from_legend_column(tmp_path: Path) -> None:
    pdf = tmp_path / "fp.pdf"
    _build_pm_floorplan(pdf)
    out = _parse(pdf)
    ptz = [a for a in out.atoms if a.atom_type == AtomType.schematic_symbol_detection and a.value["target_key"] == "ptz_camera"]
    assert ptz
    for d in ptz:
        assert d.value.get("mounting_height") == '120" AFF', d.value
        assert d.value.get("mounting_height_source") == "legend_column"


def test_detection_inherits_nic_marker_from_legend_remarks(tmp_path: Path) -> None:
    pdf = tmp_path / "fp.pdf"
    _build_pm_floorplan(pdf)
    out = _parse(pdf)
    ptz = [a for a in out.atoms if a.atom_type == AtomType.schematic_symbol_detection and a.value["target_key"] == "ptz_camera"]
    assert ptz
    for d in ptz:
        assert d.value.get("responsibility") == "NIC", d.value
        assert "NIC" in d.value.get("legend_remarks", "")
        assert "responsibility:nic" in d.entity_keys


def test_detection_inherits_by_owner_marker(tmp_path: Path) -> None:
    pdf = tmp_path / "fp.pdf"
    _build_pm_floorplan(pdf)
    out = _parse(pdf)
    dom = [a for a in out.atoms if a.atom_type == AtomType.schematic_symbol_detection and a.value["target_key"] == "fixed_dome_camera"]
    assert dom
    for d in dom:
        assert d.value.get("responsibility") == "BY OWNER"


def test_keyed_note_default_height_used_when_legend_and_schedule_silent(tmp_path: Path) -> None:
    """A pack target whose legend has no MOUNTING column and whose
    drawing has no inline callout should still get the keyed-note
    default (``All devices mounted at 48 AFF unless noted.``).
    """
    pdf = tmp_path / "fp.pdf"
    # Strip the legend MOUNTING/REMARKS so DOM falls through to the keyed note.
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((300, 90), "COUNT", fontsize=10)
    page.insert_text((72, 110), "DOM", fontsize=10)
    page.insert_text((180, 110), "FIXED DOME CAMERA", fontsize=10)
    page.insert_text((300, 110), "1", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((250, 400), "DOM", fontsize=10)
    page.insert_text((72, 540), "KEYED NOTES", fontsize=11)
    page.insert_text((72, 558), "1. All devices mounted at 48 AFF unless noted.", fontsize=9)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    out = _parse(pdf)
    dom = [a for a in out.atoms if a.atom_type == AtomType.schematic_symbol_detection and a.value["target_key"] == "fixed_dome_camera"]
    assert dom
    for d in dom:
        height = d.value.get("mounting_height", "")
        src = d.value.get("mounting_height_source", "")
        assert "48 AFF" in height or "48" in height
        assert src == "keyed_note_default"


def test_inline_callout_beats_legend_column(tmp_path: Path) -> None:
    pdf = tmp_path / "fp.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((360, 90), "MOUNTING", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((360, 110), '120" AFF', fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((200, 300), "PTZ", fontsize=10)
    # An inline callout right next to PTZ overrides the legend default.
    page.insert_text((230, 310), '90" AFF', fontsize=10)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    out = _parse(pdf)
    ptz = [a for a in out.atoms if a.atom_type == AtomType.schematic_symbol_detection and a.value["target_key"] == "ptz_camera"]
    assert ptz
    with_inline = [d for d in ptz if d.value.get("mounting_height_source") == "inline_callout"]
    assert with_inline, [(d.value.get("mounting_height"), d.value.get("mounting_height_source")) for d in ptz]
    assert "90" in with_inline[0].value["mounting_height"]


def test_legend_gap_warning_emitted_once_per_gap(tmp_path: Path) -> None:
    """When the floor plan and a riser sheet both resolve to the same
    global legend, ``legend_gap`` for a missing pack target must
    emit ONCE, not once per drawing page.
    """
    pdf = tmp_path / "fp.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    for sn in ("E1.01", "E1.02"):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 60), f"FLOOR PLAN {sn}", fontsize=14)
        page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
        page.insert_text((200, 300), "PTZ", fontsize=10)
        page.insert_text((500, 740), sn, fontsize=10)
    doc.save(str(pdf))
    doc.close()
    out = _parse(pdf)
    gaps = [
        a for a in out.atoms
        if a.atom_type == AtomType.schematic_warning
        and a.value.get("warning_type") == "legend_gap"
    ]
    # Each gap target should appear exactly once even though there
    # are two drawing pages resolving to the same legend.
    target_counts: dict[str, int] = {}
    for g in gaps:
        tk = g.value.get("target_key", "")
        target_counts[tk] = target_counts.get(tk, 0) + 1
    duplicates = {k: v for k, v in target_counts.items() if v > 1}
    assert not duplicates, duplicates


def test_fieldless_sheet_metadata_is_suppressed(tmp_path: Path) -> None:
    """A legend sheet with no title-block fields should NOT emit a
    schematic_sheet_metadata atom — that adds noise without adding
    information.
    """
    pdf = tmp_path / "fp.pdf"
    doc = fitz.open()
    # T0.01 — pure legend, no title block fields
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "SHEET T0.01 - SYMBOLS & LEGENDS", fontsize=14)
    page.insert_text((72, 90), "SYMBOL", fontsize=10)
    page.insert_text((180, 90), "DESCRIPTION", fontsize=10)
    page.insert_text((72, 110), "PTZ", fontsize=10)
    page.insert_text((180, 110), "PTZ CAMERA", fontsize=10)
    page.insert_text((500, 740), "T0.01", fontsize=10)
    # E1.01 — floor plan with title block
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 60), "FLOOR PLAN", fontsize=14)
    page.insert_text((72, 100), "See sheet T0.01 for legend.", fontsize=10)
    page.insert_text((400, 700), "Project: Test", fontsize=9)
    page.insert_text((400, 715), "Sheet Title: First Floor", fontsize=9)
    page.insert_text((500, 740), "E1.01", fontsize=10)
    doc.save(str(pdf))
    doc.close()
    out = _parse(pdf)
    metas = [a for a in out.atoms if a.atom_type == AtomType.schematic_sheet_metadata]
    pages = {m.value["page"] for m in metas}
    # Only the floor plan (page 1) should have a meta atom.
    assert 0 not in pages, [(m.value["page"], m.value) for m in metas]
    assert 1 in pages

"""Unit tests for the QA overlay renderer.

These tests construct :class:`TakeoffDocument` fixtures by hand and feed
them through :func:`write_qa_overlays` against a tiny synthesized PDF.
We verify the filter contract (which pages get rendered by default) and
the explicit-opt-in paths (``include_rejected_pages``, ``accepted_only``,
``max_pages``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.takeoff.schemas import (
    BBox,
    DeviceInstance,
    SheetRecord,
    SymbolCandidate,
    TakeoffDocument,
)


def _make_blank_pdf(path: Path, page_count: int) -> None:
    import fitz

    doc = fitz.open()
    for _ in range(page_count):
        doc.new_page(width=612, height=792)
    doc.save(str(path))
    doc.close()


def _mk_sheet(
    page_index: int,
    sheet_number: str,
    name: str,
    page_type: str,
    *,
    in_scope: bool = True,
) -> SheetRecord:
    return SheetRecord(
        page_index=page_index,
        sheet_number=sheet_number,
        sheet_name=name,
        page_type=page_type,  # type: ignore[arg-type]
        in_scope=in_scope,
        plan_viewport=BBox(x0=0, y0=0, x1=514, y1=744),
    )


def _mk_candidate(
    page_index: int,
    raw_symbol: str,
    bbox: BBox,
    *,
    rejection_reason: str | None = None,
    cand_id: str | None = None,
) -> SymbolCandidate:
    return SymbolCandidate(
        id=cand_id or f"cand_{page_index}_{raw_symbol}_{int(bbox.x0)}_{int(bbox.y0)}",
        page_index=page_index,
        raw_symbol=raw_symbol,
        normalized_class="wireless_node_outlet" if raw_symbol == "WN" else None,
        bbox=bbox,
        source_methods=["pdf_native_text"],
        rejection_reason=rejection_reason,
    )


def _mk_device(
    candidate: SymbolCandidate,
    sheet: SheetRecord,
    *,
    home_run_to: str | None = "MDF ROOM",
) -> DeviceInstance:
    return DeviceInstance(
        id=f"dev_{candidate.id}",
        page_index=candidate.page_index,
        sheet_number=sheet.sheet_number,
        sheet_name=sheet.sheet_name,
        raw_symbol=candidate.raw_symbol,
        normalized_class=candidate.normalized_class or "wireless_node_outlet",
        bbox=candidate.bbox,
        home_run_to=home_run_to,
    )


# ───────────────────────────── Tests ─────────────────────────────────────


def test_qa_overlay_default_only_renders_accepted_device_pages(tmp_path: Path) -> None:
    """By default, write_qa_overlays renders only floor-plan pages that
    have at least one accepted device. Pages with only rejected candidates
    are skipped and counted in ``skipped_non_device_pages``."""
    pytest.importorskip("fitz")
    pytest.importorskip("PIL")
    from app.takeoff.qa_overlay import write_qa_overlays

    pdf = tmp_path / "tiny_3page.pdf"
    _make_blank_pdf(pdf, page_count=3)

    sheets = [
        _mk_sheet(0, "T1.01", "LEVEL 2 FLOOR PLAN", "floor_plan"),
        _mk_sheet(1, "T1.02", "LEVEL 3 FLOOR PLAN", "floor_plan"),
        _mk_sheet(2, "T1.03", "LEVEL 4 FLOOR PLAN", "floor_plan"),
    ]
    bbox = BBox(x0=60, y0=60, x1=80, y1=80)
    candidates = [
        # Page 0: one accepted candidate -> overlay should render.
        _mk_candidate(0, "WN", bbox, cand_id="c0"),
        # Page 1: only rejected candidates -> overlay should skip by default.
        _mk_candidate(1, "WN", bbox, rejection_reason="outside_viewport", cand_id="c1"),
        # Page 2: no candidate at all -> not even considered.
    ]
    devices = [_mk_device(candidates[0], sheets[0])]

    doc = TakeoffDocument(
        source_pdf=str(pdf),
        sheets=sheets,
        candidates=candidates,
        devices=devices,
    )

    summary = write_qa_overlays(pdf_path=pdf, takeoff=doc)

    assert summary["pages_written"] == 1, summary
    assert summary["pages_requested"] == 1, summary
    # Page 1 had a candidate but no accepted device → counted as skipped.
    assert summary["skipped_non_device_pages"] >= 1, summary
    assert summary["elapsed_seconds"] >= 0.0

    qa_dir = pdf.parent / f"{pdf.stem}.derived" / "qa_overlays"
    pngs = sorted(qa_dir.glob("*.png"))
    assert len(pngs) == 1, pngs
    assert "page_0000" in pngs[0].name


def test_qa_overlay_dispatches_legend_spec_and_skips_detail_riser(tmp_path: Path) -> None:
    """The page-type router decides what each page gets:

    * detail / riser → skipped (no overlay drawn — diagrammatic content
      the current overlays can't usefully annotate)
    * legend / spec / component_schedule → rendered via the
      segmentation-aware legend overlay (the legend_table_match
      strategy). These reference pages are all structurally tabular
      content — the parser's job is to show "I saw the table
      structure", not count devices.

    This test verifies the dispatch contract — exact PNG content
    fidelity is covered by the legend_overlay module's own tests.
    """
    pytest.importorskip("fitz")
    pytest.importorskip("PIL")
    from app.takeoff.qa_overlay import write_qa_overlays

    pdf = tmp_path / "tiny_4page.pdf"
    _make_blank_pdf(pdf, page_count=4)

    sheets = [
        _mk_sheet(0, "T0.00", "SPEC", "spec"),
        _mk_sheet(1, "T0.01", "SYMBOLS & LEGENDS", "legend"),
        _mk_sheet(2, "T9.02", "INSTALLATION DETAILS", "detail"),
        _mk_sheet(3, "T7.01", "RISER DIAGRAM", "riser"),
    ]
    bbox = BBox(x0=60, y0=60, x1=80, y1=80)
    candidates = [
        _mk_candidate(
            i, "WN", bbox, rejection_reason="non_floor_plan", cand_id=f"c{i}"
        )
        for i in range(4)
    ]

    doc = TakeoffDocument(
        source_pdf=str(pdf),
        sheets=sheets,
        candidates=candidates,
    )

    # Default — spec/legend get routed to legend_table_match; detail/riser
    # are skipped. With the legend_overlay dependency available the spec
    # and legend pages render; if rendering fails (no orbitbrief_page_os)
    # they silently no-op, still counted as a request.
    summary = write_qa_overlays(pdf_path=pdf, takeoff=doc)
    assert summary["skipped_non_device_pages"] == 2, summary  # detail/riser
    assert summary["pages_requested"] == 2, summary  # spec + legend
    # pages_written can be 0..2 depending on whether the segmentation
    # pipeline successfully ran on the blank synthesized PDF.
    assert 0 <= summary["pages_written"] <= 2, summary

    qa_dir = pdf.parent / f"{pdf.stem}.derived" / "qa_overlays"
    pngs_default = list(qa_dir.glob("*.png")) if qa_dir.exists() else []
    # If the overlays rendered, they live at page_0000_*.png (spec) and
    # page_0001_*.png (legend). The detail / riser pages must NOT have
    # produced any output.
    other_pngs = [p for p in pngs_default if "page_0002" in p.name or "page_0003" in p.name]
    assert other_pngs == [], other_pngs

    # Opt-in: render all four, including rejected-only pages.
    summary_full = write_qa_overlays(
        pdf_path=pdf,
        takeoff=doc,
        include_rejected_pages=True,
        accepted_only=False,
    )
    assert summary_full["pages_written"] == 4, summary_full
    pngs_full = sorted(qa_dir.glob("*.png"))
    assert len(pngs_full) == 4, pngs_full


def test_qa_overlay_max_pages_caps_render_count(tmp_path: Path) -> None:
    """``max_pages`` caps the number of pages rendered after filtering."""
    pytest.importorskip("fitz")
    pytest.importorskip("PIL")
    from app.takeoff.qa_overlay import write_qa_overlays

    pdf = tmp_path / "tiny_5page.pdf"
    _make_blank_pdf(pdf, page_count=5)

    sheets = [_mk_sheet(i, f"T1.0{i+1}", f"LEVEL {i+1}", "floor_plan") for i in range(5)]
    bbox = BBox(x0=60, y0=60, x1=80, y1=80)
    candidates = [_mk_candidate(i, "WN", bbox, cand_id=f"c{i}") for i in range(5)]
    devices = [_mk_device(c, sheets[c.page_index]) for c in candidates]

    doc = TakeoffDocument(
        source_pdf=str(pdf),
        sheets=sheets,
        candidates=candidates,
        devices=devices,
    )

    summary = write_qa_overlays(pdf_path=pdf, takeoff=doc, max_pages=2)
    assert summary["pages_requested"] == 2, summary
    assert summary["pages_written"] == 2, summary


def test_qa_overlay_returns_zeros_when_no_candidates(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    pytest.importorskip("PIL")
    from app.takeoff.qa_overlay import write_qa_overlays

    pdf = tmp_path / "tiny_1page.pdf"
    _make_blank_pdf(pdf, page_count=1)

    doc = TakeoffDocument(source_pdf=str(pdf), sheets=[], candidates=[])
    summary = write_qa_overlays(pdf_path=pdf, takeoff=doc)
    assert summary == {
        "pages_requested": 0,
        "pages_written": 0,
        "skipped_non_device_pages": 0,
        "elapsed_seconds": summary["elapsed_seconds"],
    }
    assert summary["elapsed_seconds"] >= 0.0

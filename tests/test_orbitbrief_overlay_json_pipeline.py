"""Integration: OrbitBrief overlay (boxes JSON) → structured extraction JSON on a real PDF."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.parsers.orbitbrief_pdf import overlay_payload_and_extraction

_REPO = Path(__file__).resolve().parents[1]
_SAMPLE_PDF = (
    _REPO
    / "real_data_cases"
    / "COPPER_001_SPRING_LAKE_AUDITORIUM"
    / "CASE_DOSSIER.pdf"
)


def test_overlay_then_json_extraction_on_dossier_pdf() -> None:
    """Pipeline is detect → overlay payload → extract_from_overlay_json; both must look sane."""
    if not _SAMPLE_PDF.is_file():
        pytest.skip(f"Fixture PDF not present: {_SAMPLE_PDF}")

    overlay, doc, _written = overlay_payload_and_extraction(_SAMPLE_PDF, page_index=0)

    # --- Overlay JSON (what you’d persist / debug-draw from) ---
    assert overlay.get("pdf") == str(_SAMPLE_PDF.resolve())
    assert overlay.get("page") == 0
    assert isinstance(overlay.get("image_width"), int) and overlay["image_width"] > 0
    assert isinstance(overlay.get("image_height"), int) and overlay["image_height"] > 0
    assert "debug_stats" in overlay
    boxes = overlay.get("boxes") or []
    assert len(boxes) >= 1, "expected at least one visible box on page 0"
    b0 = boxes[0]
    assert "box_id" in b0 and "rect" in b0 and len(b0["rect"]) == 4
    assert "px_bbox" in b0

    # --- Extraction JSON (structured text + sections) ---
    assert isinstance(doc, dict)
    assert "sections" in doc or "document" in doc or "full_text" in doc
    sections = doc.get("sections") or []
    ft = (doc.get("full_text") or "").strip()
    meta = doc.get("document") or {}
    assert sections or len(ft) > 20 or len(meta) > 0, (
        "expected sections, full_text, or document metadata from extraction"
    )

    # Human-readable trace (run: pytest -s tests/test_orbitbrief_overlay_json_pipeline.py -q)
    print("\n=== OVERLAY (page 0) — summary ===")
    print(
        json.dumps(
            {
                "pdf": overlay["pdf"],
                "page": overlay["page"],
                "image_width": overlay["image_width"],
                "image_height": overlay["image_height"],
                "debug_stats": overlay.get("debug_stats"),
                "box_count": len(boxes),
                "first_box": boxes[0],
            },
            indent=2,
            default=str,
        )
    )
    print("\n=== EXTRACTION — summary ===")
    print(
        json.dumps(
            {
                "document_keys": list(meta.keys()) if meta else [],
                "section_count": len(sections),
                "section_kinds": [s.get("kind") for s in sections if isinstance(s, dict)],
                "full_text_chars": len(ft),
                "full_text_preview": ft[:600] + ("…" if len(ft) > 600 else ""),
            },
            indent=2,
            default=str,
        )
    )


def test_overlay_writes_png_and_json_siblings(tmp_path: Path) -> None:
    """Same raster pass produces PNG (visual boxes) then extraction JSON on disk."""
    if not _SAMPLE_PDF.is_file():
        pytest.skip(f"Fixture PDF not present: {_SAMPLE_PDF}")

    stem = f"{_SAMPLE_PDF.stem}_p0000"
    overlay, doc, written = overlay_payload_and_extraction(
        _SAMPLE_PDF,
        page_index=0,
        overlay_dir=tmp_path,
        file_stem=stem,
    )
    del overlay  # used only to force full pipeline; assertions use paths + doc

    png = tmp_path / f"{stem}.png"
    ov_js = tmp_path / f"{stem}.overlay.json"
    ex_js = tmp_path / f"{stem}.extraction.json"
    ex_md = tmp_path / f"{stem}.extraction.md"
    assert png.is_file() and png.stat().st_size > 10_000
    assert ov_js.is_file() and "boxes" in json.loads(ov_js.read_text(encoding="utf-8"))
    assert ex_js.is_file() and "full_text" in json.loads(ex_js.read_text(encoding="utf-8"))
    assert ex_md.is_file() and len(ex_md.read_text(encoding="utf-8")) > 100
    expected_written = {png.resolve(), ov_js.resolve(), ex_js.resolve(), ex_md.resolve()}
    assert {p.resolve() for p in written} == expected_written

    # Extraction JSON matches write_extraction_artifacts output (full pipeline, not stripped doc)
    assert (doc.get("full_text") or "") == json.loads(ex_js.read_text(encoding="utf-8")).get("full_text", "")

    print(f"\nWrote:\n  {png}\n  {ov_js}\n  {ex_js}")

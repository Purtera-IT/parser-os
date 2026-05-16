"""Render QA overlay PNGs that visualize the takeoff on each page.

Each overlay shows, for a single page:

* the plan viewport (blue rectangle)
* excluded regions / titleblock (gray rectangle)
* accepted device candidates (green) with their raw symbol + zone
* rejected candidates (red) with their rejection reason

Overlays are written under ``<PDF_STEM>.derived/qa_overlays/`` and the
QA pass silently no-ops when Pillow is not available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.takeoff.schemas import BBox, DeviceInstance, SheetRecord, SymbolCandidate, TakeoffDocument


_DEFAULT_ZOOM = 1.5  # 1.5x the native PDF size — readable, modest file size.


def _qa_dir(pdf_path: Path) -> Path:
    return pdf_path.with_name(f"{pdf_path.stem}.derived") / "qa_overlays"


def write_qa_overlays(
    *,
    pdf_path: Path,
    takeoff: TakeoffDocument,
    zoom: float = _DEFAULT_ZOOM,
) -> list[Path]:
    """Render one PNG per page that contains any candidate (accepted or not).

    Returns the list of written PNG paths. If PyMuPDF or Pillow is
    unavailable in the environment, returns an empty list.
    """
    try:
        import fitz  # noqa: F401
        from PIL import Image, ImageDraw  # noqa: F401
    except Exception:
        return []

    pages_with_candidates = {c.page_index for c in takeoff.candidates}
    if not pages_with_candidates:
        return []

    out_dir = _qa_dir(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_index = {s.page_index: s for s in takeoff.sheets}
    device_index: dict[int, list[DeviceInstance]] = {}
    for d in takeoff.devices:
        device_index.setdefault(d.page_index, []).append(d)
    candidate_index: dict[int, list[SymbolCandidate]] = {}
    for c in takeoff.candidates:
        candidate_index.setdefault(c.page_index, []).append(c)

    written: list[Path] = []
    with fitz.open(str(pdf_path)) as doc:  # type: ignore[name-defined]
        for page_index in sorted(pages_with_candidates):
            try:
                png = _render_page_overlay(
                    doc=doc,
                    page_index=page_index,
                    sheet=sheet_index.get(page_index),
                    devices=device_index.get(page_index, []),
                    candidates=candidate_index.get(page_index, []),
                    out_dir=out_dir,
                    zoom=zoom,
                )
                if png is not None:
                    written.append(png)
            except Exception:  # pragma: no cover - never fail the parse
                continue
    return written


def _render_page_overlay(
    *,
    doc: Any,
    page_index: int,
    sheet: SheetRecord | None,
    devices: list[DeviceInstance],
    candidates: list[SymbolCandidate],
    out_dir: Path,
    zoom: float,
) -> Path | None:
    import fitz
    from PIL import Image, ImageDraw

    page = doc[page_index]
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode != "RGB":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    def _scaled(bbox: BBox) -> tuple[float, float, float, float]:
        return (
            bbox.x0 * zoom,
            bbox.y0 * zoom,
            bbox.x1 * zoom,
            bbox.y1 * zoom,
        )

    if sheet is not None:
        if sheet.plan_viewport is not None:
            draw.rectangle(_scaled(sheet.plan_viewport), outline=(20, 60, 200), width=3)
        for region in sheet.excluded_regions or []:
            draw.rectangle(_scaled(region), outline=(160, 160, 160), width=2)

    device_keys = {(d.page_index, round(d.bbox.center()[0], 1), round(d.bbox.center()[1], 1)) for d in devices}
    for c in candidates:
        rect = _scaled(c.bbox)
        is_accepted = c.rejection_reason is None
        color = (40, 180, 60) if is_accepted else (210, 50, 50)
        draw.rectangle(rect, outline=color, width=2)
        label = c.raw_symbol
        if is_accepted:
            # Match candidate -> device for zone annotation.
            key = (c.page_index, round(c.bbox.center()[0], 1), round(c.bbox.center()[1], 1))
            if key in device_keys:
                dev = next((d for d in devices
                            if (d.page_index, round(d.bbox.center()[0], 1), round(d.bbox.center()[1], 1)) == key),
                           None)
                if dev and dev.home_run_to:
                    label = f"{c.raw_symbol} -> {dev.home_run_to}"
        draw.text((rect[0] + 4, max(0, rect[1] - 12)), label, fill=color)

    out_path = out_dir / f"page_{page_index:04d}_takeoff.png"
    img.save(out_path, format="PNG", optimize=True)
    return out_path


__all__ = ["write_qa_overlays"]
